import collections
import queue
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException

from benchmark.wifi_server_node import WifiServerNode

WorkItem = collections.namedtuple(
    "WorkItem", "dets_xy frame_id dt recv_mono det_s")


class QueueStats:
    """Queue occupancy for the status log.

    The T_* timings live in the node's shared PerfStats, which feeds the CSV
    as well; what is left here is the one thing only a pipelined run has --
    how much depth the stage-1 -> stage-2 queue carries, and how often it
    overflowed and dropped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.depth = []
        self.dropped = 0

    def record_depth(self, depth):
        with self._lock:
            self.depth.append(depth)

    def record_drop(self):
        with self._lock:
            self.dropped += 1

    def drain(self):
        with self._lock:
            snap = (list(self.depth), self.dropped)
            self.reset()
        return snap


class PipelinedWifiServerNode(WifiServerNode):

    def __init__(self):
        # Built before super().__init__ because _start_workers -- which the
        # base constructor calls -- records into it.
        self.queue_stats = QueueStats()
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
        self._work_q.put(None)

    def _stage_one(self, msg, stamp, recv_mono):
        dt, gap = self.scan_dt(stamp)
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
            self.queue_stats.record_drop()
            try:
                self._work_q.get_nowait()
                self._work_q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
            self.get_logger().warn(
                "tracker queue full: tracking is the bottleneck, dropping oldest",
                throttle_duration_sec=5.0)
        self._perf.record_detect(det_s, fwd_s=predict_s, scan_gap_s=gap)
        self.queue_stats.record_depth(self._work_q.qsize())

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
                # Closed out at publish time, so T_lat spans the queue wait.
                self._perf.record_track(now - t0, now - item.recv_mono)
                self.scans_processed += 1
                self._scan_count += 1
                self._scans_since_log += 1
            except Exception as e:
                self.get_logger().error(f"tracking stage failed on one frame: {e}",
                                        throttle_duration_sec=5.0)

    def _log_status(self):
        # The base class prints the link line plus the shared T_* block
        # (draining the console side of PerfStats); only the queue is left.
        super()._log_status()
        depth, dropped = self.queue_stats.drain()
        self.get_logger().info(
            f"    queue depth avg {sum(depth) / len(depth) if depth else 0.0:.2f} "
            f"max {max(depth) if depth else 0}/{self.queue_size}"
            + (f"   DROPPED x{dropped}" if dropped else ""))

    def shutdown(self):
        # Wake stage 2 even if stage 1 never got the chance to post its
        # sentinel, so join() in the base class cannot hang for its full
        # timeout on every shutdown.
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
        pass
    except Exception as e:
        print(f"Error starting benchmark wifi pipelined node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
