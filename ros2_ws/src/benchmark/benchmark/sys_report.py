"""Timing, GPU and CPU instrumentation shared by every benchmark variant.

All six nodes (gRPC / UDP / WiFi, each sequential and pipelined) report the
same five figures, in the status log and in the CSV, so runs can be read the
same way regardless of transport or threading arrangement:

  T_det    Stage-1 detection: the detector call plus the odom transform, and
           whatever wait _process_lock imposed getting in. Includes the CPU
           work either side of the network, so it sits above T_fwd.
  T_track  Stage-2 tracking: tracker.step plus the marker/pose publishes.
  T_lat    Arrival -> published, end to end. In a pipelined run this spans the
           stage-1 -> stage-2 queue wait as well, which is the point: it is
           what the consumer of the tracks actually waited.
  T_scan   Gap between consecutive scans as the ROBOT stamped them -- the
           source rate, independent of anything this server does. Read it
           against T_lat: once T_lat exceeds T_scan the server is behind the
           sensor and the queue (or the drop counter) is absorbing it.
  T_fwd    The network forward pass alone (detector.last_predict_s). Blank
           against a dr_spaam checkout that does not expose it, without
           blanking anything else. T_det - T_fwd is the cutout/NMS and
           CPU-side remainder that a faster GPU would not fix.

`inference_ms_*` is kept alongside these: it is the whole Detector.__call__
(cutout preprocessing + forward + NMS) and so is narrower than T_det, which
also carries the odom transform and the lock wait.
"""
import os
import threading

import numpy as np


def stat_ms(samples, fn):
    """Reduce a list of second-valued samples to a rounded millisecond stat."""
    if not samples:
        return ""
    return round(float(fn(samples)) * 1e3, 3)


try:
    import pynvml
except ImportError:
    pynvml = None


class GpuSampler:
    """NVML view of the GPU the detector is inferencing on.

    Two different things are reported and they must not be confused when
    reading the CSV: `gpu_util_percent` is device-wide -- the fraction of the
    last NVML sample window in which *any* kernel was resident, so the desktop
    compositor counts too -- while `gpu_proc_mem_mb` / `torch_*_mb` and the
    duty cycle the node derives from inference time are attributable to this
    process alone.
    """

    def __init__(self, logger, enabled=True):
        self._logger = logger
        self._handle = None
        self._torch = None
        self._pid = os.getpid()
        self.name = "n/a"
        self.index = -1

        if not enabled:
            return
        if pynvml is None:
            logger.warn("nvidia-ml-py (pynvml) not installed -- GPU columns will be blank")
            return

        try:
            pynvml.nvmlInit()
            self.index = self._detector_device_index()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else name
        except Exception as e:
            self._handle = None
            logger.warn(f"NVML unavailable ({e}) -- GPU columns will be blank")

    def _detector_device_index(self):
        """Sample the device torch actually put the model on, not blindly GPU 0."""
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                return torch.cuda.current_device()
        except Exception:
            pass
        return 0

    @property
    def available(self):
        return self._handle is not None

    @staticmethod
    def _optional(fn, scale=1.0):
        """Power/temperature/clocks are unsupported on some (especially laptop)
        GPUs; one unsupported metric must not blank out the whole row."""
        try:
            return round(fn() * scale, 2)
        except Exception:
            return ""

    def _process_gpu_mem_mb(self):
        """GPU memory NVML attributes to this PID. Reported separately from the
        device total because other processes share the card."""
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if p.pid == self._pid and p.usedGpuMemory is not None:
                    return round(p.usedGpuMemory / 1024**2, 1)
        except Exception:
            return ""
        return 0.0

    def sample(self):
        row = {
            "gpu_util_percent": "",
            "gpu_mem_util_percent": "",
            "gpu_mem_used_mb": "",
            "gpu_mem_total_mb": "",
            "gpu_proc_mem_mb": "",
            "gpu_power_w": "",
            "gpu_temp_c": "",
            "gpu_sm_clock_mhz": "",
            "torch_alloc_mb": "",
            "torch_reserved_mb": "",
            "torch_peak_mb": "",
        }

        if self.available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                row["gpu_util_percent"] = float(util.gpu)
                row["gpu_mem_util_percent"] = float(util.memory)
                row["gpu_mem_used_mb"] = round(mem.used / 1024**2, 1)
                row["gpu_mem_total_mb"] = round(mem.total / 1024**2, 1)
                row["gpu_proc_mem_mb"] = self._process_gpu_mem_mb()
                row["gpu_power_w"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle), 1e-3)
                row["gpu_temp_c"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU))
                row["gpu_sm_clock_mhz"] = self._optional(
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM))
            except Exception as e:
                self._logger.warn(f"NVML sample failed: {e}", throttle_duration_sec=30.0)

        # NVML sees the whole reserved pool, `memory_allocated` sees live tensors.
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                row["torch_alloc_mb"] = round(self._torch.cuda.memory_allocated() / 1024**2, 1)
                row["torch_reserved_mb"] = round(self._torch.cuda.memory_reserved() / 1024**2, 1)
                row["torch_peak_mb"] = round(self._torch.cuda.max_memory_allocated() / 1024**2, 1)
            except Exception:
                pass

        return row

    def shutdown(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None


# CSV column family -> the accumulator it is drained from. Order matters only
# for readability of the header.
_FAMILIES = {
    "t_det": "det",
    "t_track": "track",
    "t_lat": "lat",
    "t_scan": "scan",
    "t_fwd": "fwd",
    "inference": "inference",
}

TIMING_FIELDNAMES = [f"{fam}_ms_{stat}"
                     for fam in _FAMILIES
                     for stat in ("mean", "p95", "max")]


WIRE_FIELDNAMES = ["wire_ms_mean", "wire_ms_p95", "wire_ms_max"]


class PerfStats:
    """Per-scan timings, recorded off the receive path."""

    MAX_SAMPLES = 20000

    def __init__(self):
        self._lock = threading.Lock()
        self._csv = self._empty()
        self._console = self._empty()

    @staticmethod
    def _empty():
        return {"det": [], "track": [], "lat": [], "scan": [],
                "fwd": [], "inference": [], "wire": []}

    def _append(self, key, value):
        """Caller holds the lock. The cap applies to the CSV side only: the
        console side is drained every few seconds and cannot run away."""
        if value is None:
            return
        if len(self._csv[key]) < self.MAX_SAMPLES:
            self._csv[key].append(value)
        self._console[key].append(value)

    def record_detect(self, det_s, fwd_s=None, inference_s=None, scan_gap_s=None):
        """Stage 1. fwd_s / inference_s / scan_gap_s are each optional and are
        kept in their own lists rather than paired with det_s: an
        un-instrumented dr_spaam reports no forward pass, and the first scan of
        a run has no predecessor to measure a gap against. Either must go blank
        on its own without blanking T_det."""
        with self._lock:
            self._append("det", det_s)
            self._append("fwd", fwd_s)
            self._append("inference", inference_s)
            self._append("scan", scan_gap_s)

    def record_track(self, track_s, lat_s):
        """Stage 2, closed out at publish time -- so lat_s spans the queue."""
        with self._lock:
            self._append("track", track_s)
            self._append("lat", lat_s)

    def record_inference(self, inference_s, predict_s=None):
        """Detector-only path, for the sequential nodes that time the detector
        inside _detect() before they know the surrounding stage-1 total."""
        with self._lock:
            self._append("inference", inference_s)
            self._append("fwd", predict_s)

    def record_wire(self, wire_s):
        """Robot -> server transit for one scan, however this transport can
        estimate it: gRPC and WiFi difference the server's wall clock against
        the robot's scan timestamp, UDP differences the two wall-clock
        microsecond stamps the datagram header carries. All three are
        estimates -- see WIRE_FIELDNAMES for why."""
        with self._lock:
            self._append("wire", wire_s)

    def drain(self):
        """For the CSV row. Empties the CSV accumulators only."""
        with self._lock:
            snapshot = self._csv
            self._csv = self._empty()
        return snapshot

    def drain_console(self):
        """For the status log. Empties the console accumulators only."""
        with self._lock:
            snapshot = self._console
            self._console = self._empty()
        return snapshot

    @staticmethod
    def timing_row(drained):
        """The 18 T_* / inference cells for one CSV row."""
        row = {}
        for fam, key in _FAMILIES.items():
            samples = drained[key]
            row[f"{fam}_ms_mean"] = stat_ms(samples, np.mean)
            row[f"{fam}_ms_p95"] = stat_ms(samples, lambda a: np.percentile(a, 95))
            row[f"{fam}_ms_max"] = stat_ms(samples, np.max)
        return row

    @staticmethod
    def wire_row(drained):
        """The 3 wire_ms cells, for the transports that can estimate them."""
        w = drained["wire"]
        return {
            "wire_ms_mean": stat_ms(w, np.mean),
            "wire_ms_p95": stat_ms(w, lambda a: np.percentile(a, 95)),
            "wire_ms_max": stat_ms(w, np.max),
        }

    @staticmethod
    def console_block(drained, tail=""):
        """The two-line T_* block for the status log, identical in every node."""
        def ms(xs):
            return (sum(xs) / len(xs)) * 1e3 if xs else 0.0

        det, fwd = ms(drained["det"]), ms(drained["fwd"])
        return (
            f"    T_det {det:6.2f} ms   T_track {ms(drained['track']):6.2f} ms   "
            f"T_lat {ms(drained['lat']):6.2f} ms   T_scan {ms(drained['scan']):7.2f} ms\n"
            f"    T_fwd {fwd:6.2f} ms   (forward pass; "
            f"{det - fwd:6.2f} ms pre/post + lock wait)"
            + (f"\n{tail}" if tail else "")
        )
