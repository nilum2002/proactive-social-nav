"""Pipelined inf_server_wifi: detection and tracking on separate threads.
"""
import collections
import queue
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException

from inf_server_wifi.wifi_server_node import WifiServerNode

WorkItem = collections.namedtuple(
    "WorkItem", "dets_xy frame_id dt recv_mono det_s")


class StageStats:
    """Per-interval stage timings for the status log. Written from both stages,
    so every access is under the lock; drained and reset on each report."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.det_s = []
        self.predict_s = []
        self.track_s = []
        self.lat_s = []
        self.q_depth = []
        self.q_dropped = 0

    def record_detect(self, det_s, depth, predict_s=None):
        with self._lock:
            self.det_s.append(det_s)
            # Not paired with det_s: an un-instrumented dr_spaam reports
            # nothing, and T_fwd must go to zero without disturbing T_det.
            if predict_s is not None:
                self.predict_s.append(predict_s)
            self.q_depth.append(depth)

    def record_track(self, track_s, lat_s):
        with self._lock:
            self.track_s.append(track_s)
            self.lat_s.append(lat_s)

    def record_drop(self):
        with self._lock:
            self.q_dropped += 1

    def drain(self):
        with self._lock:
            snap = (list(self.det_s), list(self.predict_s), list(self.track_s),
                    list(self.lat_s), list(self.q_depth), self.q_dropped)
            self.reset()
        return snap


class PipelinedWifiServerNode(WifiServerNode):

    def __init__(self):
        self.stats = StageStats()
        super().__init__(node_name="inf_server_wifi_pipelined_node")
        self.get_logger().info(
            f"  Pipelined variant: detection (stage 1) and tracking (stage 2) run\n"
            f"  on separate threads, bounded queue of {self.queue_size}, single\n"
            f"  consumer so scan order and the dt chain are preserved.")

    def _start_workers(self):
        self.declare_parameter("queue_size", 5)
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self._work_q = queue.Queue(maxsize=self.queue_size)
        self._workers.append(threading.Thread(target=self._detect_loop, daemon=True))
        self._workers.append(threading.Thread(target=self._track_loop, daemon=True))

    # ── stage 1: detection ──────────────────────────────────────────────────
    def _detect_loop(self):
        while not self._stop.is_set():
            item = self._await_pending()
            if item is None:
                break
            try:
                self._stage_one(*item)
            except Exception as e:
                self.get_logger().error(f"detection stage failed: {e}",
                                        throttle_duration_sec=5.0)
        self._work_q.put(None)                  # sentinel stops stage 2

    def _stage_one(self, msg, stamp, recv_mono):
        dt = self.scan_dt(stamp)
        odom = self.odom_for(stamp, self.odom_tolerance_s)
        self.republish_scan(msg)

        t0 = time.perf_counter()
        with self._process_lock:
            dets_xy = self._detect(msg)
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, odom)
            predict_s = getattr(self._detector, "last_predict_s", None)
        det_s = time.perf_counter() - t0

        item = WorkItem(dets_xy, frame_id, dt, recv_mono, det_s)
        try:
            self._work_q.put_nowait(item)
        except queue.Full:
            self.stats.record_drop()
            try:
                self._work_q.get_nowait()
                self._work_q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
            self.get_logger().warn(
                "tracker queue full: tracking is the bottleneck, dropping oldest",
                throttle_duration_sec=5.0)
        self.stats.record_detect(det_s, self._work_q.qsize(), predict_s)

    # ── stage 2: tracking + publishing ──────────────────────────────────────
    def _track_loop(self):
        while True:
            item = self._work_q.get()
            if item is None:
                return
            try:
                t0 = time.perf_counter()
                active = self.tracker.step(item.dt, item.dets_xy)
                self._publish_detections_marker(item.dets_xy, item.frame_id)
                self._publish_ros(item.frame_id, active)
                now = time.perf_counter()
                self.stats.record_track(now - t0, now - item.recv_mono)
                self.scans_processed += 1
                self._scan_count += 1
                self._scans_since_log += 1
            except Exception as e:
                # Must not kill the thread: nothing would ever drain the queue.
                self.get_logger().error(f"tracking stage failed on one frame: {e}",
                                        throttle_duration_sec=5.0)

    def _log_status(self):
        super()._log_status()
        det, pred, track, lat, depth, dropped = self.stats.drain()

        def ms(xs):
            return (sum(xs) / len(xs)) * 1e3 if xs else 0.0

        self.get_logger().info(
            f"    T_det {ms(det):6.2f} ms   T_track {ms(track):6.2f} ms   "
            f"T_lat {ms(lat):6.2f} ms   queue avg "
            f"{sum(depth) / len(depth) if depth else 0.0:.2f} "
            f"max {max(depth) if depth else 0}/{self.queue_size}"
            + (f"   DROPPED x{dropped}" if dropped else "") + "\n"
            f"    T_fwd {ms(pred):6.2f} ms   (forward pass; "
            f"{ms(det) - ms(pred):6.2f} ms pre/post)")

    def shutdown(self):
        self._stop.set()
        try:
            self._work_q.put_nowait(None)
        except queue.Full:
            pass
        super().shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PipelinedWifiServerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass          # Ctrl-C / SIGTERM are normal shutdowns, not errors
    except Exception as e:
        print(f"Error starting inf_server_wifi pipelined node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
