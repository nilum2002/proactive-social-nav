"""Unit tests for the stamp-gap loss estimator.

This is the one piece of inf_server_wifi with no counterpart in the UDP server:
DDS delivers a sample with no sequence number, so loss has to be inferred from
the header timestamps. An estimator that silently reports zero would make the
DDS transport look flawless next to a UDP server that counts honestly, which is
precisely the wrong conclusion to hand a reader.
"""
import pytest

from inf_server_wifi.wifi_server_node import ArrivalStats


PERIOD = 0.025          # 40 Hz, the FROG rate


def feed(stats, n, period=PERIOD, start=1000.0, transit=0.004):
    """Deliver n evenly spaced scans. Returns the stamp of the last one."""
    stamp = start
    for _ in range(n):
        stats.observe(stamp, stamp + transit)
        stamp += period
    return stamp - period


def test_steady_stream_charges_no_loss():
    s = ArrivalStats()
    feed(s, 200)
    assert s.received == 200
    assert s.missed_est == 0
    assert s.reordered == 0
    assert s.miss_ratio == 0.0


def test_period_is_learned_and_needs_a_warmup():
    s = ArrivalStats()
    # period_s reads the deltas, and n observations produce n-1 of them.
    feed(s, ArrivalStats.PERIOD_MIN_SAMPLES)
    assert s.period_s is None
    feed(s, 2, start=1000.0 + ArrivalStats.PERIOD_MIN_SAMPLES * PERIOD)
    assert s.period_s == pytest.approx(PERIOD, rel=1e-6)


def test_one_missing_scan_is_counted_once():
    s = ArrivalStats()
    last = feed(s, 40)
    s.observe(last + 2 * PERIOD, last + 2 * PERIOD + 0.004)   # one scan skipped
    assert s.missed_est == 1


def test_a_burst_is_counted_in_full():
    s = ArrivalStats()
    last = feed(s, 40)
    s.observe(last + 6 * PERIOD, last + 6 * PERIOD + 0.004)   # five skipped
    assert s.missed_est == 5


def test_jitter_below_the_threshold_is_not_charged():
    # A scan 40% late is late, not lost. Charging it would make every busy WiFi
    # channel look like a lossy one.
    s = ArrivalStats()
    last = feed(s, 40)
    s.observe(last + 1.4 * PERIOD, last + 1.4 * PERIOD + 0.004)
    assert s.missed_est == 0


def test_loss_before_the_warmup_is_not_charged():
    # Honest limitation, asserted so it stays known: with no period estimate
    # yet there is nothing to compare a gap against, so an early burst is
    # invisible. The UDP server's sequence numbers catch this; DDS cannot.
    s = ArrivalStats()
    s.observe(1000.0, 1000.004)
    s.observe(1000.0 + 10 * PERIOD, 1000.0 + 10 * PERIOD + 0.004)
    assert s.missed_est == 0


def test_out_of_order_is_rejected_not_tracked():
    s = ArrivalStats()
    last = feed(s, 40)
    assert s.observe(last - PERIOD, last + 0.004) is False
    assert s.reordered == 1
    # The rejected sample must not have moved last_stamp, or the next in-order
    # scan would look like a two-period gap and be charged as a miss.
    assert s.last_stamp == pytest.approx(last)
    assert s.observe(last + PERIOD, last + PERIOD + 0.004) is True
    assert s.missed_est == 0


def test_duplicate_stamp_is_rejected():
    s = ArrivalStats()
    last = feed(s, 40)
    assert s.observe(last, last + 0.004) is False
    assert s.reordered == 1


def test_constant_transit_gives_zero_jitter():
    # The two clocks are unsynchronised by construction; a fixed offset must
    # cancel, or every run would report the clock skew as network jitter.
    s = ArrivalStats()
    feed(s, 100, transit=12.5)          # 12.5 s of clock offset, no variation
    assert s.jitter_ms == pytest.approx(0.0, abs=1e-6)


def test_varying_transit_shows_up_as_jitter():
    s = ArrivalStats()
    stamp = 1000.0
    for i in range(200):
        wobble = 0.01 if i % 2 else 0.0
        s.observe(stamp, stamp + 0.004 + wobble)
        stamp += PERIOD
    # The RFC 3550 EMA converges towards the mean |transit difference|, 10 ms
    # here, from below; assert it got clearly off the floor rather than pinning
    # an exact value to the gain.
    assert 5.0 < s.jitter_ms < 10.0


def test_miss_ratio_is_over_expected_not_received():
    s = ArrivalStats()
    last = feed(s, 40)
    s.observe(last + 2 * PERIOD, last + 2 * PERIOD + 0.004)
    assert s.miss_ratio == pytest.approx(1.0 / 42.0)
