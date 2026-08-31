"""Sequential inference server using the Norfair tracker.

Same gRPC service, detector, TF and publishing as `grpc_server_node`; only the
tracking backend differs. Detection and tracker update run inline, one after the
other, for each scan received -- the non-pipelined arrangement.

Tracking happens in the robot's odometry frame: `_to_tracking_frame` projects
each detection with the odometry contemporaneous with that scan before the
tracker sees it, which is what keeps velocity estimates meaningful while the
robot is driving. With no odometry on the stream it falls back to the laser
frame and ego-motion leaks into the velocity estimates.

Track initiation and deletion are counter-based (see norfair_tracker): a
candidate is created from every unmatched detection but only initiated once its
counter passes `c_init`, and an established track is dropped after `c_del`
updates without a match. Initiation latency is therefore ~c_init / scan_rate
seconds, which is what to minimise for obstacle avoidance; the false positives
that come with a low c_init are held down by keeping `conf_thresh` high (>= 0.8).
"""
import rclpy

from benchmark.grpc_server_node import InfServerNode
from benchmark.norfair_tracker import NorfairMultiObjectTracker


class NorfairMixin:
    """Parameter declaration + tracker factory, shared by both variants."""

    tracker_name = "Norfair"

    def _declare_norfair_params(self):
        # Declared from _make_servicer, which the base __init__ calls before the
        # gRPC server starts -- so these exist before any client can connect.
        self.declare_parameter("c_init", 3)
        self.declare_parameter("c_del", 10)
        self.declare_parameter("nominal_dt", 0.1)
        gp = self.get_parameter
        self.c_init = gp("c_init").get_parameter_value().integer_value
        self.c_del = gp("c_del").get_parameter_value().integer_value
        self.nominal_dt = gp("nominal_dt").get_parameter_value().double_value
        # Fail at startup, not on the first client connection.
        if not 0 <= self.c_init < self.c_del:
            raise ValueError(
                f"invalid parameters: c_init={self.c_init}, c_del={self.c_del}. "
                f"Require 0 <= c_init < c_del.")

    def make_tracker(self):
        return NorfairMultiObjectTracker(
            association_threshold=self.tracker_kwargs["association_threshold"],
            c_init=self.c_init,
            c_del=self.c_del,
            nominal_dt=self.nominal_dt,
        )

    def _norfair_banner(self):
        rate = 1.0 / self.nominal_dt if self.nominal_dt > 1e-6 else float("nan")
        return (f"  Tracker      : Norfair (counter-based init/delete)\n"
                f"  C_init={self.c_init}  C_del={self.c_del}  "
                f"gate={self.tracker_kwargs['association_threshold']:.2f} m\n"
                f"  initiation latency ~= {self.c_init / rate * 1e3:.0f} ms at "
                f"{rate:.0f} Hz | conf_thresh={self.conf_thresh}")


class NorfairInfServerNode(NorfairMixin, InfServerNode):

    def __init__(self):
        super().__init__(node_name="norfair_inf_server_node")
        self.get_logger().info(self._norfair_banner())

    def _make_servicer(self):
        self._declare_norfair_params()
        return super()._make_servicer()

    def _log_status(self):
        rate = self._scans_since_log / self.status_log_period_s
        self._scans_since_log = 0
        self.get_logger().info(
            f"[norfair-seq] {self._servicer.client_count} client(s) | "
            f"{rate:.1f} scans/s | {self.inference_fps:.1f} FPS (DR-SPAAM) | "
            f"{self._scan_count} total")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NorfairInfServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting norfair inf_server node: {e}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
