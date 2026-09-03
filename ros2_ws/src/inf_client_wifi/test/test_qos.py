"""Unit tests for the QoS construction.

QoS is the entire substance of this package -- it is what separates "DDS over
WiFi" from "DDS over WiFi done in the way that actually works". A silently
wrong policy here does not raise; it produces a run whose numbers look like a
transport result but are really a configuration mistake, which is the worst
possible failure for a benchmark. Hence tests rather than a successful run.
"""
import pytest
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from inf_client_wifi.dds_relay_node import build_qos


def test_best_effort_is_default_shape():
    qos = build_qos("best_effort", 1)
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    # VOLATILE matters as much as BEST_EFFORT: TRANSIENT_LOCAL would hand a
    # late-joining inf_server a stale scan as its first sample.
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_reliable_is_selectable():
    # Not a misconfiguration -- running reliable is how the cost of DDS
    # retransmission over a lossy link gets measured.
    assert build_qos("reliable", 10).reliability == ReliabilityPolicy.RELIABLE


def test_unknown_reliability_raises():
    # Must fail loudly at startup. Falling back to a default would quietly
    # change which transport the experiment is measuring.
    with pytest.raises(ValueError, match="unknown reliability"):
        build_qos("realiable", 1)


@pytest.mark.parametrize("depth", [0, -5])
def test_depth_is_floored_at_one(depth):
    # A KEEP_LAST queue of zero is not a valid DDS depth; clamp rather than
    # let the middleware reject the publisher at construction time.
    assert build_qos("best_effort", depth).depth == 1
