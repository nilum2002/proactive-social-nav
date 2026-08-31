"""Pipelined inference server using the Norfair tracker.

Stage 1 (detector inference, T_det) runs on the receive thread; stage 2 (tracker
update, T_track) runs on a per-connection consumer thread behind a bounded
queue. Because the two stages overlap, scan i+1's inference proceeds while scan
i's tracker update is still running, so the server keeps up with a shorter
T_scan than the sequential arrangement can. End-to-end latency for a scan,
T_lat, still spans its own detection plus its own tracker update.

The queue is bounded with blocking puts and a single consumer: no frame is ever
dropped and arrival order is preserved, so the tracker's state chain -- and the
dt sequence its motion model depends on -- is identical to the sequential node's.
Pipelining buys throughput and latency, not different tracking.

Everything else -- odometry-frame tracking, counter-based initiation and
deletion -- is as described in norfair_server_node.
"""
import rclpy

from benchmark.grpc_pipelined_node import PipelinedInfServerNode
from benchmark.norfair_server_node import NorfairMixin


class NorfairPipelinedInfServerNode(NorfairMixin, PipelinedInfServerNode):

    def __init__(self):
        # Distinct from the KF pipelined node: the marker/pose topics are
        # private (~/markers), so sharing a node name would also mean sharing
        # topics, and an A/B run would interleave both trackers' markers.
        super().__init__(node_name="norfair_inf_server_pipelined_node")
        self.get_logger().info(self._norfair_banner())

    def _make_servicer(self):
        self._declare_norfair_params()
        return super()._make_servicer()

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        det, pred, track, lat, gap, depth, q_full = self.stats.drain()

        def ms(xs):
            return (sum(xs) / len(xs)) * 1e3 if xs else 0.0

        self.get_logger().info(
            f"[norfair-pipelined] {self._servicer.client_count} client(s) | "
            f"{rate:.1f} scans/s | {self._scan_count} total\n"
            f"    T_det   {ms(det):6.2f} ms   T_track {ms(track):6.2f} ms   "
            f"T_lat {ms(lat):6.2f} ms   T_scan {ms(gap):6.2f} ms\n"
            f"    T_fwd   {ms(pred):6.2f} ms   (forward pass; "
            f"{ms(det) - ms(pred):6.2f} ms pre/post + lock wait)\n"
            f"    queue depth avg {sum(depth) / len(depth) if depth else 0.0:.2f} "
            f"max {max(depth) if depth else 0}/{self.queue_size}"
            + (f"   FULL x{q_full}" if q_full else ""))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NorfairPipelinedInfServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting norfair pipelined node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
