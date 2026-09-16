from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # .../sys_reports_server
ANALYSIS = HERE                       # .../sys_reports_server/Analysis

SCAN_TARGET = 500                     # analyse the first 500 scans of each run

# --------------------------------------------------------------------------
# Palette (validated categorical order, light surface)
# --------------------------------------------------------------------------

SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "font.size": 9,
    "figure.dpi": 130,
})


# --------------------------------------------------------------------------
# Run discovery / loading
# --------------------------------------------------------------------------

@dataclass
class Run:
    method: str        # grpc | udp | wifi
    motion: str        # static_robot | moving_robot
    mode: str          # seq | pipelined
    path: str          # absolute path to the raw service log

    @property
    def name(self) -> str:
        return f"{self.motion}_{self.mode}"

    @property
    def label(self) -> str:
        motion = "static" if self.motion.startswith("static") else "moving"
        return f"{motion} / {self.mode}"


def discover_runs(method: str) -> list[Run]:
    """Find every raw service log for a transport, in a stable order."""
    runs: list[Run] = []
    for motion in ("static_robot", "moving_robot"):
        for mode in ("seq", "pipelined"):
            d = os.path.join(ROOT, method, motion, mode)
            if not os.path.isdir(d):
                continue
            logs = sorted(
                f for f in os.listdir(d)
                if f.startswith("inf_server_service_log")
                and f.endswith(".csv")
                and not f.endswith("_summary.csv")
            )
            for f in logs:
                runs.append(Run(method, motion, mode, os.path.join(d, f)))
    return runs


def load_first_scans(run: Run, target: int = SCAN_TARGET):
    """Load a service log and keep the windows covering the first ``target`` scans.

    ``scans_total`` is cumulative across the run, so we keep every window up to
    and including the first one whose cumulative count reaches ``target``. If the
    run never gets that far, the whole run is kept and flagged as truncated.
    """
    df = pd.read_csv(run.path)
    df["wall_time"] = pd.to_datetime(df["wall_time"], errors="coerce")

    reached = df.index[df["scans_total"] >= target]
    truncated = len(reached) == 0
    cut = df if truncated else df.loc[: reached[0]]
    cut = cut.copy().reset_index(drop=True)

    scans_covered = float(cut["scans_total"].max())
    return cut, truncated, scans_covered


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric metric column (wall_time is the only non-metric column)."""
    return [c for c in df.columns
            if c != "wall_time" and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def run_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (and spread) of every metric over the trimmed window rows.

    NaNs are skipped per metric - warm-up windows report no latency numbers, and
    counting them as zero would bias the means down.
    """
    cols = metric_columns(df)
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "metric": c,
            "n_windows": int(s.size),
            "mean": s.mean() if s.size else np.nan,
            "std": s.std() if s.size > 1 else np.nan,
            "min": s.min() if s.size else np.nan,
            "median": s.median() if s.size else np.nan,
            "max": s.max() if s.size else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def _finish(fig, out_png: str) -> str:
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _lines(ax, df, cols, labels=None, offset=0):
    labels = labels or cols
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.plot(df["elapsed_s"], df[c], color=SERIES[(i + offset) % len(SERIES)],
                label=lab, solid_capstyle="round")
    ax.set_xlabel("elapsed (s)")
    if len(cols) >= 2:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=INK_2)


def plot_run_timeseries(df: pd.DataFrame, run: Run, out_dir: str,
                        truncated: bool, scans: float) -> list[str]:
    """Per-run time-series panels over the first-900-scan window."""
    made = []
    suffix = f" - first {int(scans)} scans" + (" (run ended early)" if truncated else "")
    stem = os.path.join(out_dir, run.name)

    # 1. Server-side pipeline latency (all in ms - one axis).
    cols = _present(df, ["t_lat_ms_mean", "t_det_ms_mean", "inference_ms_mean",
                         "t_track_ms_mean", "t_fwd_ms_mean"])
    if cols:
        fig, ax = plt.subplots(figsize=(9.2, 3.4))
        _lines(ax, df, cols,
               [c.replace("_ms_mean", "").replace("t_", "") for c in cols])
        ax.set_ylabel("ms")
        ax.set_title(f"{run.method.upper()} {run.label} - pipeline latency{suffix}")
        made.append(_finish(fig, f"{stem}_latency_timeseries.png"))

    # 2. Wire time - separate figure, its scale dwarfs the compute stages.
    cols = _present(df, ["wire_ms_mean", "wire_ms_p95", "wire_ms_max"])
    if cols:
        fig, ax = plt.subplots(figsize=(9.2, 3.0))
        _lines(ax, df, cols, [c.replace("wire_ms_", "") for c in cols], offset=1)
        ax.set_ylabel("wire time (ms)")
        ax.set_title(f"{run.method.upper()} {run.label} - transport wire time{suffix}")
        made.append(_finish(fig, f"{stem}_wire_timeseries.png"))

    # 3. Throughput (both Hz/fps - same unit family, one axis).
    cols = _present(df, ["scan_rate_hz", "drspaam_fps"])
    if cols:
        fig, ax = plt.subplots(figsize=(9.2, 3.0))
        _lines(ax, df, cols, ["scan rate (Hz)", "DR-SPAAM (fps)"], offset=2)
        ax.set_ylabel("per second")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{run.method.upper()} {run.label} - throughput{suffix}")
        made.append(_finish(fig, f"{stem}_throughput_timeseries.png"))

    # 4. GPU - percentages and physical units split into stacked subplots
    #    rather than a second y-axis.
    pct = _present(df, ["gpu_util_percent", "gpu_mem_util_percent",
                        "drspaam_gpu_duty_percent"])
    phys = _present(df, ["gpu_power_w", "gpu_temp_c", "gpu_proc_mem_mb"])
    if pct or phys:
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.2), sharex=True)
        if pct:
            _lines(axes[0], df, pct, [c.replace("_percent", "") for c in pct])
            axes[0].set_ylabel("%")
        if phys:
            for i, c in enumerate(phys):
                axes[1].plot(df["elapsed_s"], df[c] / df[c].abs().max(),
                             color=SERIES[(i + 3) % len(SERIES)],
                             label=f"{c} (max {df[c].max():.0f})")
            axes[1].set_ylabel("normalised to run max")
            axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                           labelcolor=INK_2)
        axes[1].set_xlabel("elapsed (s)")
        axes[0].set_title(f"{run.method.upper()} {run.label} - GPU{suffix}")
        made.append(_finish(fig, f"{stem}_gpu_timeseries.png"))

    # 5. Host CPU / memory.
    cols = _present(df, ["cpu_percent_system", "cpu_percent_process", "ram_percent"])
    if cols:
        fig, ax = plt.subplots(figsize=(9.2, 3.0))
        _lines(ax, df, cols, [c.replace("_percent", "").replace("percent_", "")
                              for c in cols], offset=4)
        ax.set_ylabel("%")
        ax.set_title(f"{run.method.upper()} {run.label} - host CPU / RAM{suffix}")
        made.append(_finish(fig, f"{stem}_host_timeseries.png"))

    # 6. Transport reliability - only udp/wifi logs carry these columns.
    cols = _present(df, ["loss_percent", "missed_percent", "jitter_ms",
                         "busy_drop_in_window", "bad_packets_in_window"])
    if cols:
        fig, axes = plt.subplots(len(cols), 1, figsize=(9.2, 1.5 * len(cols) + 1.2),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for i, c in enumerate(cols):
            axes[i].plot(df["elapsed_s"], df[c], color=SERIES[i % len(SERIES)],
                         solid_capstyle="round")
            axes[i].set_ylabel(c, fontsize=8)
        axes[-1].set_xlabel("elapsed (s)")
        axes[0].set_title(f"{run.method.upper()} {run.label} - transport reliability{suffix}")
        made.append(_finish(fig, f"{stem}_reliability_timeseries.png"))

    return made


def _bar_panel(means: pd.DataFrame, metrics: list[str], title: str,
               unit: str, out_png: str) -> str | None:
    """Grouped bars: one group per metric, one bar per run."""
    metrics = [m for m in metrics if m in means.index and means.loc[m].notna().any()]
    if not metrics:
        return None
    runs = list(means.columns)
    x = np.arange(len(metrics), dtype=float)
    width = min(0.8 / len(runs), 0.28)

    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3.0, 3.8))
    for i, r in enumerate(runs):
        # 2px surface gap between adjacent bars.
        pos = x + (i - (len(runs) - 1) / 2) * width
        vals = means.loc[metrics, r].to_numpy(dtype=float)
        ax.bar(pos, vals, width * 0.9, color=SERIES[i % len(SERIES)], label=r,
               edgecolor=SURFACE, linewidth=1.6)
        for px, v in zip(pos, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:,.4g}", (px, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel(unit)
    ax.set_title(title)
    if len(runs) >= 2:
        ax.legend(loc="upper right", labelcolor=INK_2, ncols=min(len(runs), 4))
    ax.margins(y=0.18)
    return _finish(fig, out_png)


def plot_method_comparison(means: pd.DataFrame, method: str, out_dir: str) -> list[str]:
    """Cross-run comparison figures for one transport."""
    made = []
    tag = method.upper()
    groups = [
        (["t_lat_ms_mean", "t_det_ms_mean", "inference_ms_mean",
          "t_track_ms_mean", "t_fwd_ms_mean"],
         f"{tag} - mean pipeline latency over first {SCAN_TARGET} scans", "ms",
         "latency_means.png"),
        (["wire_ms_mean", "wire_ms_p95", "wire_ms_max", "t_scan_ms_mean"],
         f"{tag} - mean wire / scan interval over first {SCAN_TARGET} scans", "ms",
         "wire_means.png"),
        (["scan_rate_hz", "drspaam_fps"],
         f"{tag} - mean throughput over first {SCAN_TARGET} scans", "per second",
         "throughput_means.png"),
        (["gpu_util_percent", "gpu_mem_util_percent", "drspaam_gpu_duty_percent",
          "cpu_percent_system", "cpu_percent_process", "ram_percent"],
         f"{tag} - mean resource utilisation over first {SCAN_TARGET} scans", "%",
         "resources_means.png"),
        (["gpu_power_w", "gpu_temp_c", "gpu_proc_mem_mb", "proc_rss_mb"],
         f"{tag} - mean GPU power / thermals / memory over first {SCAN_TARGET} scans",
         "W, °C, MB", "gpu_physical_means.png"),
        (["loss_percent", "missed_percent", "jitter_ms",
          "busy_drop_in_window", "bad_packets_in_window"],
         f"{tag} - mean transport reliability over first {SCAN_TARGET} scans",
         "% / ms / count", "reliability_means.png"),
    ]
    for metrics, title, unit, fname in groups:
        p = _bar_panel(means, metrics, title, unit, os.path.join(out_dir, f"{method}_{fname}"))
        if p:
            made.append(p)
    return made


def plot_latency_distribution(frames: dict[str, pd.DataFrame], method: str,
                              out_dir: str) -> str | None:
    """ECDF of per-window end-to-end latency, one line per run.

    A boxplot is the wrong form here: the bulk of the windows sit inside a ~1 ms
    band while rare warm-up windows land in the hundreds, so every box collapses
    to a line. The ECDF shows both the tight body and the tail.
    """
    series = {name: df["t_lat_ms_mean"].dropna()
              for name, df in frames.items()
              if "t_lat_ms_mean" in df.columns and df["t_lat_ms_mean"].notna().any()}
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for i, (name, s) in enumerate(series.items()):
        x = np.sort(s.to_numpy())
        y = np.arange(1, x.size + 1) / x.size * 100.0
        ax.step(x, y, where="post", color=SERIES[i % len(SERIES)],
                label=f"{name} (median {np.median(x):.1f} ms)",
                solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlabel("per-window mean end-to-end latency (ms, log)")
    ax.set_ylabel("windows at or below (%)")
    ax.set_ylim(0, 101)
    ax.set_title(f"{method.upper()} - latency distribution over first {SCAN_TARGET} scans")
    if len(series) >= 2:
        ax.legend(loc="lower right", labelcolor=INK_2)
    return _finish(fig, os.path.join(out_dir, f"{method}_latency_ecdf.png"))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def analyze_method(method: str, target: int = SCAN_TARGET) -> pd.DataFrame:
    """Run the full first-N-scan analysis for one transport.

    Writes, under ``Analysis/<method>/``:
      * ``<run>/<run>_first<N>_windows.csv``  - the trimmed raw window rows
      * ``<run>/<run>_first<N>_stats.csv``    - per-metric mean/std/min/median/max
      * ``<method>_first<N>_means.csv``       - metric x run matrix of means
      * ``<method>_first<N>_stats_long.csv``  - all stats, one row per run+metric
      * ``<method>_coverage.csv``             - what each run actually covered
      * ``figures/`` and per-run PNGs
    """
    out_root = os.path.join(ANALYSIS, method)
    fig_root = os.path.join(out_root, "figures")
    os.makedirs(fig_root, exist_ok=True)

    runs = discover_runs(method)
    if not runs:
        raise SystemExit(f"no service logs found for method '{method}' under {ROOT}")

    means: dict[str, pd.Series] = {}
    long_rows, coverage, frames = [], [], {}

    for run in runs:
        df, truncated, scans = load_first_scans(run, target)
        frames[run.label] = df

        run_dir = os.path.join(out_root, run.name)
        os.makedirs(run_dir, exist_ok=True)
        df.to_csv(os.path.join(run_dir, f"{run.name}_first{target}_windows.csv"),
                  index=False)

        stats = run_stats(df)
        stats.to_csv(os.path.join(run_dir, f"{run.name}_first{target}_stats.csv"),
                     index=False)

        means[run.label] = stats.set_index("metric")["mean"]
        s = stats.copy()
        s.insert(0, "run", run.label)
        s.insert(0, "method", method)
        long_rows.append(s)

        coverage.append({
            "method": method,
            "run": run.label,
            "source_csv": os.path.relpath(run.path, ROOT),
            "windows_used": len(df),
            "scans_covered": scans,
            "target_scans": target,
            "reached_target": not truncated,
            "elapsed_s_covered": float(df["elapsed_s"].max()),
        })

        plot_run_timeseries(df, run, run_dir, truncated, scans)

    means_df = pd.DataFrame(means)
    means_df.index.name = "metric"
    means_df.to_csv(os.path.join(out_root, f"{method}_first{target}_means.csv"))

    pd.concat(long_rows, ignore_index=True).to_csv(
        os.path.join(out_root, f"{method}_first{target}_stats_long.csv"), index=False)

    cov = pd.DataFrame(coverage)
    cov.to_csv(os.path.join(out_root, f"{method}_coverage.csv"), index=False)

    plot_method_comparison(means_df, method, fig_root)
    plot_latency_distribution(frames, method, fig_root)

    print(f"\n=== {method.upper()} - mean over first {target} scans ===")
    print(cov.to_string(index=False))
    short = [m for m in ["scan_rate_hz", "drspaam_fps", "t_lat_ms_mean",
                         "t_det_ms_mean", "inference_ms_mean", "wire_ms_mean",
                         "gpu_util_percent", "cpu_percent_process"]
             if m in means_df.index]
    print()
    print(means_df.loc[short].round(3).to_string())
    print(f"\nwritten to {out_root}")
    return means_df
