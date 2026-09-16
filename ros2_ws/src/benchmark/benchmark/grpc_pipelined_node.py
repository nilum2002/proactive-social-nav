import collections
import math
import threading
import time
import queue

import rclpy

from benchmark.grpc_server_node import (
    InfServerNode,
    PerceptionServicer,
    perception_stream_pb2,
)
from benchmark.kalman_tracker import MultiObjectTracker

WorkItem = collections.namedtuple(
    "WorkItem", "dets_xy frame_id dt scan_timestamp arrival_t arrival_wall det_s"
)


class QueueStats:
    """Queue occupancy for the status log.

    The T_* timings all live in the node's shared PerfStats, which is what
    feeds the CSV as well; what is left here is the one thing that only
    exists in a pipelined run -- how much depth the stage-1 -> stage-2 queue
    is actually carrying, and how often it filled.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.depth = []
        self.full_events = 0

    def record_depth(self, depth):
        with self._lock:
            self.depth.append(depth)

    def record_full(self):
        with self._lock:
            self.full_events += 1

    def drain(self):
        with self._lock:
            snapshot = (list(self.depth), self.full_events)
            self.reset()
        return snapshot


class PipelinedPerceptionServicer(PerceptionServicer):
    """Detection on the receive thread, tracking on a per-connection thread."""

    def StreamSensorData(self, request_iterator, context):
        node = self._node
        peer = context.peer()
        with self._count_lock:
            self._client_count += 1
        self._logger.info(f"robot connected: {peer} (now {self.client_count} client(s))")

        tracker = MultiObjectTracker(**node.tracker_kwargs)
        work_q = queue.Queue(maxsize=node.queue_size)
        tracker_thread = threading.Thread(
            target=self._tracker_loop, args=(tracker, work_q), daemon=True
        )
        tracker_thread.start()

        latest_odom = None   # (x, y, yaw)
        last_scan_time = None
        scans_processed = 0

        try:
            for frame in request_iterator:
                kind = frame.WhichOneof("payload")
                if kind == "odom":
                    o = frame.odom
                    yaw = 2.0 * math.atan2(o.qz, o.qw)
                    latest_odom = (o.x, o.y, yaw)
                    node.broadcast_odom_tf(o.x, o.y, yaw, o.timestamp)
                    continue
                if kind != "scan":
                    continue

                arrival_t = time.perf_counter()
                arrival_wall = time.time()
                scan_pb = frame.scan

                dt = 0.1
                gap = None
                if last_scan_time is not None:
                    candidate = scan_pb.timestamp - last_scan_time
                    gap = candidate
                    if 0.0 < candidate <= 2.0:
                        dt = candidate
                last_scan_time = scan_pb.timestamp

                node.republish_scan(scan_pb)

                # ── Stage 1: detect + transform ──────────────────────────────
                t0 = time.perf_counter()
                with node._process_lock:
                    dets_xy = node._detect(scan_pb)
                    dets_xy, frame_id = node._to_tracking_frame(dets_xy, latest_odom)
                    predict_s = getattr(node._detector, "last_predict_s", None)
                det_s = time.perf_counter() - t0

                # One sink for the numbers (PerfStats: CSV + status log,
                # each with its own drain), one for queue occupancy.
                node._perf.record_detect(det_s, fwd_s=predict_s, scan_gap_s=gap)
                node.queue_stats.record_depth(work_q.qsize())

                item = WorkItem(dets_xy, frame_id, dt, scan_pb.timestamp,
                                arrival_t, arrival_wall, det_s)
                if work_q.full():
                    node.queue_stats.record_full()
                    self._logger.warn(
                        "tracker queue full: tracking is now the bottleneck, "
                        "detection will block", throttle_duration_sec=5.0,
                    )
                work_q.put(item)

                node._scan_count += 1
                node._scans_since_log += 1
                scans_processed += 1
        finally:
            work_q.put(None)                    # sentinel stops the tracker loop
            tracker_thread.join(timeout=2.0)
            with self._count_lock:
                self._client_count -= 1
            self._logger.info(
                f"robot disconnected: {peer} (now {self.client_count} client(s))"
            )

        return perception_stream_pb2.SensorAck(scans_processed=scans_processed)

    def _tracker_loop(self, tracker, work_q):
        """Stage 2. Single consumer, so arrival order (and the KF state chain
        that depends on it) is preserved without any reorder buffer."""
        node = self._node
        while True:
            item = work_q.get()
            if item is None:
                return
            try:
                t0 = time.perf_counter()
                active_tracks = tracker.step(item.dt, item.dets_xy)
                node._publish_detections_marker(item.dets_xy, item.frame_id)
                node._publish_ros(item.frame_id, active_tracks)
                now = time.perf_counter()
                node._perf.record_track(now - t0, now - item.arrival_t)
                node.record_wire_latency(item.arrival_wall - item.scan_timestamp)
            except Exception as e:
                self._logger.error(f"tracker stage failed on one frame: {e}")


class PipelinedInfServerNode(InfServerNode):

    def __init__(self):
        self.queue_stats = QueueStats()
        super().__init__(node_name="inf_server_pipelined_node")
        self.get_logger().info(
            f"  Pipelined variant: detection (stage 1) and tracking (stage 2) run\n"
            f"  on separate threads, bounded queue of {self.queue_size}, single\n"
            f"  consumer so KF frame order is preserved."
        )

    def _make_servicer(self):
        self.declare_parameter("queue_size", 5)
        self.queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value
        return PipelinedPerceptionServicer(self)

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        depth, q_full = self.queue_stats.drain()

        queue_line = (
            f"    queue depth avg {sum(depth) / len(depth) if depth else 0.0:.2f} "
            f"max {max(depth) if depth else 0}/{self.queue_size}"
            + (f"   FULL x{q_full}" if q_full else "")
        )

        self.get_logger().info(
            f"[inf_server-pipelined] {self._servicer.client_count} client(s) | "
            f"{rate:.1f} scans/s | {self._scan_count} total\n"
            + self._perf.console_block(self._perf.drain_console(), tail=queue_line)
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PipelinedInfServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting pipelined inf_server node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
