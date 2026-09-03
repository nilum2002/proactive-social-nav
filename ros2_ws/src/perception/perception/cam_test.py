"""
cam_test.py

Camera diagnostic with two display paths:

  mode:=gui        cv2.imshow window - needs a display, use at the desk
  mode:=headless   no window at all - stats to the log, optional snapshots to
                   disk.  This is the one to use over SSH or on the robot.
  mode:=auto       gui when DISPLAY/WAYLAND_DISPLAY is set, headless otherwise

and two sources:

  source:=topic    subscribe to what camera_node publishes (tests the ROS path)
  source:=device   open /dev/videoN directly (tests the hardware alone, and
                   tells you whether a fault is in the camera or in ROS)

Headless mode is not just gui-without-the-window: calling cv2.imshow with no
display raises, and on a build of OpenCV compiled without GUI support it raises
even when a display exists.  The two paths are kept separate so the headless one
never touches highgui.

Examples
--------
    ros2 run perception cam_test --ros-args -p mode:=headless
    ros2 run perception cam_test --ros-args -p mode:=gui -p source:=device
    ros2 run perception cam_test --ros-args -p mode:=headless -p save_dir:=/tmp/frames
"""

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image

# Below this mean pixel value the frames are effectively black: lens cap on,
# no light, or the sensor never started exposing.
DARK_FRAME_THRESHOLD = 5.0


def display_available() -> bool:
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def opencv_has_gui() -> bool:
    """True if this OpenCV build has highgui (the -headless wheels do not)."""
    if not hasattr(cv2, 'imshow'):
        return False
    try:
        return 'GUI:' in cv2.getBuildInformation() or True
    except Exception:
        return False


class CamTestNode(Node):

    def __init__(self):
        super().__init__('cam_test')

        self.declare_parameter('mode', 'auto')            # auto | gui | headless
        self.declare_parameter('source', 'topic')         # topic | device
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        self.declare_parameter('video_device', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('report_period_s', 3.0)
        self.declare_parameter('save_dir', '')            # headless: where to drop snapshots
        self.declare_parameter('save_every_n', 0)         # 0 disables saving
        self.declare_parameter('window_name', 'cam_test')
        # Orientation, applied for source:=device only.  In topic mode the
        # frames arrive already oriented by camera_node, so correcting again
        # here would double-apply it.  Keep these matching camera_params.yaml.
        self.declare_parameter('rotate_deg', 0)       # 0 | 90 | 180 | 270
        self.declare_parameter('flip_horizontal', False)
        self.declare_parameter('flip_vertical', False)

        gp = self.get_parameter
        self.source = gp('source').get_parameter_value().string_value
        self.image_topic = gp('image_topic').get_parameter_value().string_value
        self.device = gp('video_device').value
        self.width = int(gp('width').value)
        self.height = int(gp('height').value)
        self.report_period = float(gp('report_period_s').value)
        self.save_dir = gp('save_dir').get_parameter_value().string_value
        self.save_every_n = int(gp('save_every_n').value)
        self.window_name = gp('window_name').get_parameter_value().string_value

        self.flip_h = bool(gp('flip_horizontal').value)
        self.flip_v = bool(gp('flip_vertical').value)
        self.rotate_deg = int(gp('rotate_deg').value) % 360
        if self.rotate_deg not in (0, 90, 180, 270):
            self.get_logger().warn(
                f'rotate_deg={self.rotate_deg} is not 0/90/180/270 - ignoring')
            self.rotate_deg = 0
        self._rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(self.rotate_deg)

        self.mode = self._resolve_mode(gp('mode').get_parameter_value().string_value)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._frames = 0
        self._total = 0
        self._saved = 0
        self._dark_frames = 0
        self._last_shape = None
        self._last_report = time.time()
        self._window_open = False
        self._cap = None

        self.get_logger().info(
            f'cam_test  mode={self.mode}  source={self.source}  '
            + (f'topic={self.image_topic}  (orientation comes from camera_node)'
               if self.source == 'topic'
               else f'device=/dev/video{self.device}  '
                    f'rot={self.rotate_deg}deg flip_h={self.flip_h} flip_v={self.flip_v}')
        )

        if self.source == 'device':
            self._start_device()
        else:
            self._start_topic()

        self.create_timer(self.report_period, self._report)

    # ── mode selection ───────────────────────────────────────────────────────

    def _resolve_mode(self, requested: str) -> str:
        if requested == 'gui':
            if not display_available():
                self.get_logger().warn(
                    'mode:=gui but no DISPLAY/WAYLAND_DISPLAY - falling back to headless. '
                    'Over SSH use `ssh -X`, or just run headless.'
                )
                return 'headless'
            if not opencv_has_gui():
                self.get_logger().warn(
                    'this OpenCV build has no highgui (opencv-python-headless?) - '
                    'falling back to headless'
                )
                return 'headless'
            return 'gui'

        if requested == 'headless':
            return 'headless'

        # auto
        chosen = 'gui' if (display_available() and opencv_has_gui()) else 'headless'
        self.get_logger().info(f'mode=auto resolved to {chosen}')
        return chosen

    # ── sources ──────────────────────────────────────────────────────────────

    def _start_topic(self):
        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        if self.image_topic.endswith('/compressed'):
            self.create_subscription(CompressedImage, self.image_topic, self._on_compressed, qos)
        else:
            self.create_subscription(Image, self.image_topic, self._on_image, qos)

    def _start_device(self):
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f'could not open /dev/video{self.device}')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Poll a little faster than a 30 fps sensor so we never gate the device.
        self.create_timer(1.0 / 40.0, self._poll_device)

    def _poll_device(self):
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._handle(self._orient(frame))

    def _orient(self, frame):
        """Same correction camera_node applies, for the raw-device path."""
        if self._rotate_code is not None:
            frame = cv2.rotate(frame, self._rotate_code)
        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)
        return frame

    def _on_image(self, msg: Image):
        self._handle(self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))

    def _on_compressed(self, msg: CompressedImage):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._handle(frame)
        else:
            self.get_logger().warn('could not decode JPEG', throttle_duration_sec=5.0)

    # ── per-frame handling ───────────────────────────────────────────────────

    def _handle(self, frame):
        self._frames += 1
        self._total += 1
        self._last_shape = frame.shape

        mean = float(frame.mean())
        if mean < DARK_FRAME_THRESHOLD:
            self._dark_frames += 1

        if self.save_every_n > 0 and self.save_dir and self._total % self.save_every_n == 0:
            path = os.path.join(self.save_dir, f'frame_{self._total:06d}.jpg')
            if cv2.imwrite(path, frame):
                self._saved += 1

        if self.mode == 'gui':
            self._show(frame, mean)

    def _show(self, frame, mean):
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        for i, line in enumerate((
            f'{w}x{h}  frame {self._total}',
            f'mean {mean:.1f}',
        )):
            cv2.putText(annotated, line, (8, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        try:
            cv2.imshow(self.window_name, annotated)
            self._window_open = True
            # Required for the window to actually paint; also gives us the key.
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                self.get_logger().info('q/ESC pressed - shutting down')
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            raise
        except cv2.error as exc:
            self.get_logger().error(
                f'imshow failed ({exc}) - switching to headless for the rest of this run'
            )
            self.mode = 'headless'
            self._window_open = False

    # ── reporting ────────────────────────────────────────────────────────────

    def _report(self):
        now = time.time()
        elapsed = now - self._last_report
        self._last_report = now
        rate = self._frames / elapsed if elapsed > 0 else 0.0
        self._frames = 0

        if self._total == 0:
            where = (self.image_topic if self.source == 'topic'
                     else f'/dev/video{self.device}')
            self.get_logger().warn(
                f'no frames yet from {where}'
                + ('  - is camera_node running? check `ros2 topic info <topic>`'
                   if self.source == 'topic' else '')
            )
            return

        shape = f'{self._last_shape[1]}x{self._last_shape[0]}' if self._last_shape else '?'
        line = f'[cam_test] {rate:.1f} fps | {shape} | {self._total} frames'
        if self.save_every_n > 0:
            line += f' | {self._saved} saved'
        self.get_logger().info(line)

        if self._dark_frames == self._total:
            self.get_logger().warn(
                f'every frame so far is near-black (mean < {DARK_FRAME_THRESHOLD}) - '
                'lens cap on, or pointed at something dark?'
            )

    def destroy_node(self):
        if self._cap is not None:
            self._cap.release()
        if self._window_open:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CamTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'cam_test failed: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
