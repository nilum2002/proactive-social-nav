"""
camera_node.py

Publishes the USB webcam as ROS topics:

    <ns>/image_raw              sensor_msgs/Image           (optional, off by default)
    <ns>/image_raw/compressed   sensor_msgs/CompressedImage (JPEG, on by default)
    <ns>/camera_info            sensor_msgs/CameraInfo

Raw is off by default on purpose.  At 640x480 bgr8 / 22 fps raw costs ~20 MB/s,
which fills this Pi's remaining card space in under four minutes of recording;
the same stream as JPEG is under 1 MB/s.  Turn publish_raw on only when a
consumer genuinely needs uncompressed pixels, and prefer not to record it.

Capture runs on its own thread rather than a timer: V4L2 hands back the oldest
buffered frame, so a timer that reads slower than the sensor delivers builds up
a queue and the images drift further behind real time the longer it runs.
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('video_device', 0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('publish_raw', False)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('fourcc', 'MJPG')
        # Sensor streams conventionally run best-effort; set true if you would
        # rather block the publisher than lose a frame.
        self.declare_parameter('reliable_qos', False)
        # Physical mount orientation.  A camera fitted upside down needs a 180
        # rotation, which is not the same as flip_vertical: flipping one axis
        # mirrors the scene (text reads backwards), rotating flips both.
        self.declare_parameter('rotate_deg', 0)     # 0 | 90 | 180 | 270
        # Mirroring, applied after the rotation.  Only for genuine mirroring
        # (e.g. a selfie-style preview), not for a rotated mount.
        self.declare_parameter('flip_horizontal', False)
        self.declare_parameter('flip_vertical', False)

        gp = self.get_parameter
        self.device = gp('video_device').value
        self.frame_id = gp('frame_id').get_parameter_value().string_value
        self.width = int(gp('width').value)
        self.height = int(gp('height').value)
        self.fps = float(gp('fps').value)
        self.publish_raw = bool(gp('publish_raw').value)
        self.publish_compressed = bool(gp('publish_compressed').value)
        self.jpeg_quality = int(gp('jpeg_quality').value)
        self.fourcc = gp('fourcc').get_parameter_value().string_value
        self.flip_h = bool(gp('flip_horizontal').value)
        self.flip_v = bool(gp('flip_vertical').value)

        self.rotate_deg = int(gp('rotate_deg').value) % 360
        if self.rotate_deg not in (0, 90, 180, 270):
            self.get_logger().warn(
                f'rotate_deg={self.rotate_deg} is not one of 0/90/180/270 - ignoring'
            )
            self.rotate_deg = 0
        self._rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(self.rotate_deg)

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=(ReliabilityPolicy.RELIABLE if bool(gp('reliable_qos').value)
                         else ReliabilityPolicy.BEST_EFFORT),
        )

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, 'image_raw', qos) if self.publish_raw else None
        self._compressed_pub = (
            self.create_publisher(CompressedImage, 'image_raw/compressed', qos)
            if self.publish_compressed else None
        )
        self._info_pub = self.create_publisher(CameraInfo, 'camera_info', qos)

        if self._image_pub is None and self._compressed_pub is None:
            raise RuntimeError('publish_raw and publish_compressed are both false - nothing to do')

        self._cap = self._open_capture()

        self._frames = 0
        self._publish_failures = 0
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        self.create_timer(5.0, self._log_status)

        self.get_logger().info(
            f'camera_node streaming /dev/video{self.device} at '
            f'{self.out_width}x{self.out_height} target {self.fps:g} fps  '
            f'rot={self.rotate_deg}deg '
            f'(raw={"on" if self.publish_raw else "off"}, '
            f'jpeg={"on" if self.publish_compressed else "off"} q{self.jpeg_quality})'
        )

    def _open_capture(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'could not open video device {self.device}')

        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Smallest driver queue the backend allows, so read() returns a current
        # frame instead of one from several cycles ago.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self.width, self.height):
            self.get_logger().warn(
                f'device gave {actual_w}x{actual_h}, not the requested '
                f'{self.width}x{self.height}; using the device values'
            )
            self.width, self.height = actual_w, actual_h

        # A quarter turn swaps the published dimensions, and camera_info has to
        # describe the image that actually goes out, not the sensor readout.
        if self.rotate_deg in (90, 270):
            self.out_width, self.out_height = self.height, self.width
        else:
            self.out_width, self.out_height = self.width, self.height

        return cap

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.get_logger().warn('frame grab failed', throttle_duration_sec=5.0)
                continue

            frame = self._orient(frame)

            stamp = self.get_clock().now().to_msg()
            try:
                self._publish(frame, stamp)
            except Exception as exc:
                self._publish_failures += 1
                self.get_logger().error(f'publish failed: {exc}', throttle_duration_sec=5.0)
                continue
            self._frames += 1

    def _orient(self, frame):
        """Rotate for the mount, then mirror. Order matters: rotating a mirrored
        frame is not the same picture as mirroring a rotated one."""
        if self._rotate_code is not None:
            frame = cv2.rotate(frame, self._rotate_code)

        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)

        return frame

    def _publish(self, frame, stamp):
        if self._compressed_pub is not None:
            ok, buf = cv2.imencode(
                '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if ok:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.format = 'jpeg'
                msg.data = np.asarray(buf).tobytes()
                self._compressed_pub.publish(msg)

        if self._image_pub is not None:
            msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            self._image_pub.publish(msg)

        self._info_pub.publish(self._camera_info(stamp))

    def _camera_info(self, stamp):
        """
        Geometry only - this camera is uncalibrated.

        k/d/p are left zeroed rather than filled with a plausible-looking guess,
        so anything needing real intrinsics fails loudly instead of quietly
        projecting with invented numbers.  Run camera_calibration and load a
        real file before using this for anything metric.
        """
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = self.out_width
        info.height = self.out_height
        info.distortion_model = 'plumb_bob'
        return info

    def _log_status(self):
        frames, self._frames = self._frames, 0
        rate = frames / 5.0
        msg = f'[camera] {rate:.1f} fps published'
        if self._publish_failures:
            msg += f' | {self._publish_failures} publish failure(s)'
        self.get_logger().info(msg)

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'camera_node failed: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
