"""Pipelined inf_server_udp: detection and tracking on separate threads.

Stage 1 takes the newest scan waiting from the receive path and runs the
detector; stage 2 takes the resulting detections off a bounded queue and runs
the tracker and the publishers. The two overlap, so scan i+1 is already in the
detector while scan i is still being tracked, and the server keeps up with a
shorter scan period than the sequential arrangement can.

The queue is bounded with a single consumer, so scan order -- and the dt
sequence the Kalman filter depends on -- is preserved exactly as in the
sequential node. Unlike the gRPC pipelined server, a full queue does *not*
block: there is no backpressure to propagate to a UDP sender, so the oldest
queued item is discarded and counted. Blocking stage 1 would only push the loss
one hop upstream, into the socket receive buffer, where it would be invisible
instead of counted.
"""
import collections
import queue
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException

from inf_server_udp.udp_server_node import UdpServerNode

WorkItem = collections.namedtuple(
    "WorkItem", "session dets_xy frame_id dt recv_mono det_s")


class StageStats:
    """Per-interval stage timings for the status log."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.det_s = []
        self.track_s = []
        self.lat_s = []
        self.q_depth = []
        self.q_dropped = 0

    def record_detect(self, det_s, depth):
        with self._lock:
            self.det_s.append(det_s)
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
            snap = (list(self.det_s), list(self.track_s), list(self.lat_s),
                    list(self.q_depth), self.q_dropped)
            self.reset()
        return snap


class PipelinedUdpServerNode(UdpServerNode):

    def __init__(self):
        self.stats = StageStats()
        super().__init__(node_name="inf_server_udp_pipelined_node")
        self.get_logger().info(
            f"  Pipelined variant: detection (stage 1) and tracking (stage 2) run\n"
            f"  on separate threads, bounded queue of {self.queue_size}, single\n"
            f"  consumer so scan order and the dt chain are preserved.")

    def _start_workers(self):
        # Declared here rather than beside the other parameters because the base
        # __init__ calls this before the receive loop starts, so queue_size is
        # guaranteed to exist before any packet can arrive.
        self.declare_parameter("queue_size", 5)
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        self._work_q = queue.Queue(maxsize=self.queue_size)
        self._workers.append(threading.Thread(target=self._detect_loop, daemon=True))
        self._workers.append(threading.Thread(target=self._track_loop, daemon=True))

    # ── stage 1: detection ──────────────────────────────────────────────────
    def _detect_loop(self):
        while not self._stop.is_set():
            with self._work_cv:
                item = self._take_pending()
                while item is None:
                    if self._stop.is_set():
                        self._work_q.put(None)
                        return
                    self._work_cv.wait(0.2)
                    item = self._take_pending()
            session, view = item
            try:
                self._stage_one(session, view)
            except Exception as e:
                self.get_logger().error(f"detection stage failed: {e}",
                                        throttle_duration_sec=5.0)
        self._work_q.put(None)

    def _stage_one(self, session, view):
        dt = 0.1
        if session.last_scan_time is not None:
            candidate = view.timestamp - session.last_scan_time
            if 0.0 < candidate <= 2.0:
                dt = candidate
        session.last_scan_time = view.timestamp

        odom = session.odom_for(view.timestamp, self.odom_tolerance_s)
        self.republish_scan(view)

        t0 = time.perf_counter()
        with self._process_lock:
            dets_xy = self._detect(view)
            # Transformed here, not in stage 2, so detections are projected with
            # the odometry contemporaneous with THIS scan rather than whatever
            # has arrived by the time stage 2 dequeues it.
            dets_xy, frame_id = self._to_tracking_frame(dets_xy, odom)
        det_s = time.perf_counter() - t0

        item = WorkItem(session, dets_xy, frame_id, dt, view.recv_mono, det_s)
        try:
            self._work_q.put_nowait(item)
        except queue.Full:
            # Tracking has become the bottleneck. Drop the oldest rather than
            # block: blocking stage 1 would just move the loss into the socket
            # buffer, where nothing counts it.
            self.stats.record_drop()
            try:
                self._work_q.get_nowait()
                self._work_q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
            self.get_logger().warn(
                "tracker queue full: tracking is the bottleneck, dropping oldest",
                throttle_duration_sec=5.0)
        self.stats.record_detect(det_s, self._work_q.qsize())

    # ── stage 2: tracking + publishing ──────────────────────────────────────
    def _track_loop(self):
        while True:
            item = self._work_q.get()
            if item is None:
                return
            try:
                t0 = time.perf_counter()
                active = item.session.tracker.step(item.dt, item.dets_xy)
                self._publish_detections_marker(item.dets_xy, item.frame_id)
                self._publish_ros(item.frame_id, active)
                now = time.perf_counter()
                self.stats.record_track(now - t0, now - item.recv_mono)
                item.session.scans_processed += 1
                self._scan_count += 1
                self._scans_since_log += 1
            except Exception as e:
                # Must not kill the thread: nothing would ever drain the queue.
                self.get_logger().error(f"tracking stage failed on one frame: {e}",
                                        throttle_duration_sec=5.0)

    def _log_status(self):
        super()._log_status()
        det, track, lat, depth, dropped = self.stats.drain()

        def ms(xs):
            return (sum(xs) / len(xs)) * 1e3 if xs else 0.0

        self.get_logger().info(
            f"    T_det {ms(det):6.2f} ms   T_track {ms(track):6.2f} ms   "
            f"T_lat {ms(lat):6.2f} ms   queue avg "
            f"{sum(depth) / len(depth) if depth else 0.0:.2f} "
            f"max {max(depth) if depth else 0}/{self.queue_size}"
            + (f"   DROPPED x{dropped}" if dropped else ""))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PipelinedUdpServerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass          # Ctrl-C / SIGTERM are normal shutdowns, not errors
    except Exception as e:
        print(f"Error starting inf_server_udp pipelined node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
