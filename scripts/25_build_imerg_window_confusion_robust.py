from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WINDOWS_MIN = (30, 60, 180, 360)
THRESHOLD_MODES = ("fixed_threshold", "relative_threshold")
STRIDE_MODES = ("overlap", "scale_stride")
FILE_TIME_RE = re.compile(r"(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})")
BOOTSTRAP_METRICS = (
    "point_pod",
    "grid3x3_pod",
    "point_far",
    "grid3x3_far",
    "point_ets",
    "grid3x3_ets",
    "point_bias",
    "grid3x3_bias",
    "point_hss",
    "grid3x3_hss",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build robust IMERG-vs-AVAMET window confusion metrics with fixed and relative thresholds, "
            "including overlap and non-overlapping stride evaluations."
        )
    )
    parser.add_argument(
        "--qc-input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="Input AVAMET QC parquet.",
    )
    parser.add_argument(
        "--station-scale-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_events_station_scale_summary.csv"),
        help="Station-scale summary CSV with AVAMET POT thresholds.",
    )
    parser.add_argument(
        "--station-alignment-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_imerg.csv"),
        help="Station-to-IMERG alignment CSV.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("data/imerg_cv"),
        help="Root directory with IMERG CV subset files.",
    )
    parser.add_argument(
        "--imerg-quantile",
        type=float,
        default=0.995,
        help="IMERG wet-window quantile used for the relative-threshold evaluation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_by_scale.csv"),
        help="Scale-level robust confusion summary CSV.",
    )
    parser.add_argument(
        "--output-station-csv",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_station_scale.csv"),
        help="Per-station, per-scale robust confusion summary CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_summary.json"),
        help="JSON summary.",
    )
    parser.add_argument(
        "--bootstrap-output-csv",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_bootstrap.csv"),
        help="Bootstrap confidence intervals for the scale-level robust confusion table.",
    )
    parser.add_argument(
        "--bootstrap-output-json",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_bootstrap.json"),
        help="JSON bootstrap summary for the scale-level robust confusion table.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of UTC-day block bootstrap replicates for the scale-level robust confusion table.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the UTC-day block bootstrap.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200000,
        help="Batch size for streaming the AVAMET QC parquet.",
    )
    parser.add_argument(
        "--station-limit",
        type=int,
        default=0,
        help="Optional limit on number of stations for testing.",
    )
    return parser


def to_minute_index(values: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    ts = pd.to_datetime(values, errors="coerce", utc=False)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def parse_subset_timestamp(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = FILE_TIME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse IMERG timestamps from {path.name}")
    start_ts = pd.to_datetime(
        match.group("date") + match.group("start"),
        format="%Y%m%d%H%M%S",
        errors="raise",
    )
    end_ts = start_ts + pd.Timedelta(minutes=30)
    return start_ts, end_ts


def build_subset_index(subset_root: Path) -> pd.DataFrame:
    files = sorted(subset_root.rglob("*.cv.npz"))
    if not files:
        raise FileNotFoundError(f"No .cv.npz files found under {subset_root}")

    records: list[dict[str, object]] = []
    for path in files:
        start_ts, end_ts = parse_subset_timestamp(path)
        records.append(
            {
                "path": str(path),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "path_depth": len(path.parts),
            }
        )

    index_df = pd.DataFrame(records).sort_values(
        ["start_ts", "path_depth", "path"],
        ascending=[True, False, True],
    )
    index_df = index_df.drop_duplicates(subset=["start_ts"], keep="first").reset_index(drop=True)
    index_df["end_min"] = to_minute_index(index_df["end_ts"])
    return index_df


def load_avamet_thresholds(summary_csv: Path) -> dict[tuple[str, int], float]:
    summary = pd.read_csv(summary_csv)
    summary["station_id"] = summary["station_id"].astype(str)
    if "completeness_mode_used" in summary.columns:
        summary = summary.loc[summary["completeness_mode_used"].astype(str) == "strict"].copy()

    thresholds: dict[tuple[str, int], float] = {}
    for row in summary.itertuples(index=False):
        threshold = pd.to_numeric(pd.Series([row.threshold_mm]), errors="coerce").iloc[0]
        if pd.isna(threshold) or float(threshold) <= 0.0:
            continue
        thresholds[(str(row.station_id), int(row.window_min))] = float(threshold)
    return thresholds


def load_station_cells(alignment_csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    alignment = pd.read_csv(alignment_csv)
    alignment["station_id"] = alignment["station_id"].astype(str)
    alignment = alignment.sort_values("station_id").reset_index(drop=True)

    required = ["station_id", "imerg_cv_lon_idx", "imerg_cv_lat_idx"]
    missing = [column for column in required if column not in alignment.columns]
    if missing:
        raise KeyError(f"Missing required columns in {alignment_csv}: {missing}")

    lon_min = int(alignment["imerg_cv_lon_idx"].min())
    lat_min = int(alignment["imerg_cv_lat_idx"].min())
    station_ids = alignment["station_id"].tolist()
    lon_local = alignment["imerg_cv_lon_idx"].astype(int).to_numpy(dtype=np.int32) - lon_min
    lat_local = alignment["imerg_cv_lat_idx"].astype(int).to_numpy(dtype=np.int32) - lat_min
    return station_ids, lon_local, lat_local


class NPZCache:
    def __init__(self, max_items: int = 32) -> None:
        self.max_items = max(1, int(max_items))
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        cached = self._store.get(path)
        if cached is not None:
            self._store.move_to_end(path)
            return cached

        with np.load(path) as handle:
            data = handle["precipitation"].astype(np.float32, copy=False)

        self._store[path] = data
        self._store.move_to_end(path)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return data


def build_imerg_station_masses(
    subset_index: pd.DataFrame,
    lon_local: np.ndarray,
    lat_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_times = int(subset_index.shape[0])
    n_stations = int(lon_local.size)
    direct_mass = np.full((n_times, n_stations), np.nan, dtype=np.float32)
    max3x3_mass = np.full((n_times, n_stations), np.nan, dtype=np.float32)
    end_min = subset_index["end_min"].to_numpy(dtype=np.int64, copy=False)

    cache = NPZCache(max_items=8)
    for idx, path in enumerate(subset_index["path"].tolist()):
        precip = cache.get(str(path))
        direct_rate = precip[lon_local, lat_local]
        direct_mass[idx, :] = direct_rate * np.float32(0.5)

        padded = np.pad(precip, pad_width=1, mode="constant", constant_values=np.nan)
        neighborhood_stack = np.stack(
            [
                padded[1 + dx : 1 + dx + precip.shape[0], 1 + dy : 1 + dy + precip.shape[1]]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            ],
            axis=0,
        )
        finite_any = np.isfinite(neighborhood_stack).any(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_grid = np.nanmax(neighborhood_stack, axis=0)
        max_grid = max_grid.astype(np.float32, copy=False)
        max_grid[~finite_any] = np.nan
        max3x3_mass[idx, :] = max_grid[lon_local, lat_local] * np.float32(0.5)

        if (idx + 1) % 2000 == 0:
            print(f"Loaded IMERG subset windows: {idx + 1} / {n_times}")

    return end_min, direct_mass, max3x3_mass


def compute_reference_window_metrics(
    source_end_min: np.ndarray,
    source_duration_min: np.ndarray,
    source_mass: np.ndarray,
    usable_mask: np.ndarray,
    fail_mask: np.ndarray,
    ref_end_min: np.ndarray,
    window_min: int,
) -> dict[str, np.ndarray]:
    n_src = int(source_end_min.size)
    n_ref = int(ref_end_min.size)

    if n_src == 0 or n_ref == 0:
        return {
            "accum_mm": np.zeros(n_ref, dtype=np.float64),
            "coverage_ratio": np.zeros(n_ref, dtype=np.float64),
            "fail_count": np.zeros(n_ref, dtype=np.int32),
            "complete_strict": np.zeros(n_ref, dtype=bool),
        }

    source_end_min = source_end_min.astype(np.int64, copy=False)
    source_duration_min = source_duration_min.astype(np.float64, copy=False)
    source_mass = source_mass.astype(np.float64, copy=False)
    usable_mask = usable_mask.astype(bool, copy=False)
    fail_mask = fail_mask.astype(bool, copy=False)
    ref_end_min = ref_end_min.astype(np.int64, copy=False)

    source_start_min = source_end_min.astype(np.float64) - source_duration_min
    usable_duration = np.where(usable_mask, source_duration_min, 0.0)
    usable_mass = np.where(usable_mask, source_mass, 0.0)
    fail_countable = (fail_mask & (source_duration_min > 0)).astype(np.int32)

    prefix_duration = np.concatenate(([0.0], np.cumsum(usable_duration, dtype=np.float64)))
    prefix_mass = np.concatenate(([0.0], np.cumsum(usable_mass, dtype=np.float64)))
    prefix_fail = np.concatenate(([0], np.cumsum(fail_countable, dtype=np.int64)))

    window_start = ref_end_min.astype(np.float64) - float(window_min)
    left = np.searchsorted(source_end_min, window_start, side="right")
    right = np.searchsorted(source_end_min, ref_end_min, side="right")

    accum = prefix_mass[right] - prefix_mass[left]
    covered = prefix_duration[right] - prefix_duration[left]
    fail_count = prefix_fail[right] - prefix_fail[left]

    left_overlap_mask = left < right
    if np.any(left_overlap_mask):
        ref_idx = np.flatnonzero(left_overlap_mask)
        src_idx = left[ref_idx]
        excess_before = window_start[ref_idx] - (source_end_min[src_idx].astype(np.float64) - source_duration_min[src_idx])
        trim_mask = excess_before > 0.0
        if np.any(trim_mask):
            ref_trim = ref_idx[trim_mask]
            src_trim = src_idx[trim_mask]
            excess = excess_before[trim_mask]
            frac = np.zeros(excess.size, dtype=np.float64)
            valid = source_duration_min[src_trim] > 0
            frac[valid] = np.clip(excess[valid] / source_duration_min[src_trim][valid], 0.0, 1.0)
            covered[ref_trim] -= frac * usable_duration[src_trim]
            accum[ref_trim] -= frac * usable_mass[src_trim]

    right_overlap_mask = right < n_src
    if np.any(right_overlap_mask):
        ref_idx = np.flatnonzero(right_overlap_mask)
        src_idx = right[ref_idx]
        source_start = source_end_min[src_idx].astype(np.float64) - source_duration_min[src_idx]
        overlap_start = np.maximum(window_start[ref_idx], source_start)
        overlap = ref_end_min[ref_idx].astype(np.float64) - overlap_start
        add_mask = overlap > 0.0
        if np.any(add_mask):
            ref_add = ref_idx[add_mask]
            src_add = src_idx[add_mask]
            overlap_add = overlap[add_mask]
            frac = np.zeros(overlap_add.size, dtype=np.float64)
            valid = source_duration_min[src_add] > 0
            frac[valid] = np.clip(overlap_add[valid] / source_duration_min[src_add][valid], 0.0, 1.0)
            covered[ref_add] += frac * usable_duration[src_add]
            accum[ref_add] += frac * usable_mass[src_add]
            fail_count[ref_add] += ((fail_mask[src_add]) & (source_duration_min[src_add] > 0)).astype(np.int64)

    coverage_ratio = np.clip(covered / float(window_min), 0.0, 1.0)
    complete_strict = (coverage_ratio >= 0.95) & (fail_count == 0)

    return {
        "accum_mm": accum,
        "coverage_ratio": coverage_ratio,
        "fail_count": fail_count.astype(np.int32),
        "complete_strict": complete_strict.astype(bool),
    }


def compute_stride_mask(ref_end_min: np.ndarray, window_min: int, stride_mode: str) -> np.ndarray:
    if stride_mode == "overlap":
        return np.ones(ref_end_min.size, dtype=bool)
    return (ref_end_min % int(window_min)) == 0


def compute_wet_quantile(accum_mm: np.ndarray, complete_mask: np.ndarray, quantile: float) -> float:
    wet = complete_mask & np.isfinite(accum_mm) & (accum_mm > 0.0)
    if not np.any(wet):
        return float("nan")
    return float(np.nanquantile(accum_mm[wet], float(quantile)))


def summarize_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int | None]:
    total = tp + fp + fn + tn
    pod = tp / (tp + fn) if (tp + fn) > 0 else None
    far = fp / (tp + fp) if (tp + fp) > 0 else None
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else None
    if total > 0:
        random_hits = ((tp + fp) * (tp + fn)) / total
        ets_denom = tp + fp + fn - random_hits
        ets = (tp - random_hits) / ets_denom if ets_denom > 0 else None
        hss_denom = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
        hss = (2.0 * (tp * tn - fn * fp) / hss_denom) if hss_denom > 0 else None
    else:
        random_hits = None
        ets = None
        hss = None
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pod": pod,
        "far": far,
        "csi": csi,
        "precision": precision,
        "specificity": specificity,
        "bias": bias,
        "ets": ets,
        "hss": hss,
        "total_windows": int(total),
        "random_hits": random_hits,
    }


def accumulate_block_counts(
    accumulator: dict[tuple[str, str, int], np.ndarray],
    key: tuple[str, str, int],
    block_codes: np.ndarray,
    n_blocks: int,
    point_tp_mask: np.ndarray,
    point_fp_mask: np.ndarray,
    point_fn_mask: np.ndarray,
    point_tn_mask: np.ndarray,
    grid_tp_mask: np.ndarray,
    grid_fp_mask: np.ndarray,
    grid_fn_mask: np.ndarray,
    grid_tn_mask: np.ndarray,
) -> None:
    counts = accumulator.setdefault(key, np.zeros((n_blocks, 8), dtype=np.int64))
    masks = (
        point_tp_mask,
        point_fp_mask,
        point_fn_mask,
        point_tn_mask,
        grid_tp_mask,
        grid_fp_mask,
        grid_fn_mask,
        grid_tn_mask,
    )
    for col_idx, mask in enumerate(masks):
        if np.any(mask):
            counts[:, col_idx] += np.bincount(block_codes[mask], minlength=n_blocks).astype(np.int64, copy=False)


def summarize_block_confusion(counts: np.ndarray) -> dict[str, object]:
    point = summarize_confusion(
        int(counts[:, 0].sum()),
        int(counts[:, 1].sum()),
        int(counts[:, 2].sum()),
        int(counts[:, 3].sum()),
    )
    grid = summarize_confusion(
        int(counts[:, 4].sum()),
        int(counts[:, 5].sum()),
        int(counts[:, 6].sum()),
        int(counts[:, 7].sum()),
    )
    return {
        "block_count": int(counts.shape[0]),
        "point_metrics": point,
        "grid3x3_metrics": grid,
    }


def bootstrap_block_confusion(counts: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, object]:
    point_estimate = summarize_block_confusion(counts)
    n_blocks = int(counts.shape[0])
    if n_blocks == 0:
        return {
            "point_estimate": point_estimate,
            "confidence_intervals": {
                metric: {"low": None, "high": None} for metric in BOOTSTRAP_METRICS
            },
            "block_count": 0,
        }

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(int(n_bootstrap)):
        choice = rng.integers(0, n_blocks, size=n_blocks)
        sample_counts = counts[choice].sum(axis=0, dtype=np.int64).reshape(1, 8)
        metrics = summarize_block_confusion(sample_counts)
        sample_row = {
            "point_pod": metrics["point_metrics"]["pod"],
            "grid3x3_pod": metrics["grid3x3_metrics"]["pod"],
            "point_far": metrics["point_metrics"]["far"],
            "grid3x3_far": metrics["grid3x3_metrics"]["far"],
            "point_ets": metrics["point_metrics"]["ets"],
            "grid3x3_ets": metrics["grid3x3_metrics"]["ets"],
            "point_bias": metrics["point_metrics"]["bias"],
            "grid3x3_bias": metrics["grid3x3_metrics"]["bias"],
            "point_hss": metrics["point_metrics"]["hss"],
            "grid3x3_hss": metrics["grid3x3_metrics"]["hss"],
        }
        for metric in BOOTSTRAP_METRICS:
            value = sample_row[metric]
            if value is not None and not pd.isna(value):
                samples[metric].append(float(value))

    intervals: dict[str, dict[str, float | None]] = {}
    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(samples[metric], dtype=np.float64)
        if values.size == 0:
            intervals[metric] = {"low": None, "high": None}
        else:
            intervals[metric] = {
                "low": float(np.nanquantile(values, 0.025)),
                "high": float(np.nanquantile(values, 0.975)),
            }

    return {
        "point_estimate": point_estimate,
        "confidence_intervals": intervals,
        "block_count": n_blocks,
    }


def bootstrap_rows_from_counts(
    daily_block_counts: dict[tuple[str, str, int], np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    json_rows: list[dict[str, object]] = []

    for key in sorted(daily_block_counts):
        threshold_mode, stride_mode, window_min = key
        result = bootstrap_block_confusion(
            daily_block_counts[key],
            n_bootstrap=int(n_bootstrap),
            seed=int(seed) + int(window_min),
        )
        point_metrics = result["point_estimate"]["point_metrics"]
        grid_metrics = result["point_estimate"]["grid3x3_metrics"]
        intervals = result["confidence_intervals"]

        row: dict[str, object] = {
            "threshold_mode": threshold_mode,
            "stride_mode": stride_mode,
            "window_min": int(window_min),
            "block_count": int(result["block_count"]),
            "point_pod": point_metrics["pod"],
            "grid3x3_pod": grid_metrics["pod"],
            "point_far": point_metrics["far"],
            "grid3x3_far": grid_metrics["far"],
            "point_ets": point_metrics["ets"],
            "grid3x3_ets": grid_metrics["ets"],
            "point_bias": point_metrics["bias"],
            "grid3x3_bias": grid_metrics["bias"],
            "point_hss": point_metrics["hss"],
            "grid3x3_hss": grid_metrics["hss"],
        }
        for metric in BOOTSTRAP_METRICS:
            row[f"{metric}_ci_low"] = intervals[metric]["low"]
            row[f"{metric}_ci_high"] = intervals[metric]["high"]
        rows.append(row)

        json_rows.append(
            {
                "threshold_mode": threshold_mode,
                "stride_mode": stride_mode,
                "window_min": int(window_min),
                **result,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["threshold_mode", "stride_mode", "window_min"]).reset_index(drop=True),
        json_rows,
    )


def process_station(
    station_df: pd.DataFrame,
    station_id: str,
    station_col: int,
    avamet_thresholds: dict[tuple[str, int], float],
    imerg_end_min: np.ndarray,
    imerg_direct_mass: np.ndarray,
    imerg_3x3_mass: np.ndarray,
    imerg_quantile: float,
    block_codes: np.ndarray,
    n_blocks: int,
    daily_block_counts: dict[tuple[str, str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    station_df = station_df.sort_values("timestamp_utc").reset_index(drop=True)
    source_end_min = to_minute_index(station_df["timestamp_utc"])
    source_duration_min = pd.to_numeric(station_df["delta_prev_min"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    source_duration_min = np.where(np.isfinite(source_duration_min) & (source_duration_min > 0), source_duration_min, 0.0)

    source_mass = pd.to_numeric(station_df["precip_mm_qc"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    source_mass = np.where(np.isfinite(source_mass), source_mass, 0.0)
    usable_mask = station_df["usable_interval_extremes"].fillna(False).to_numpy(dtype=bool, copy=False)
    fail_mask = (
        station_df["qc_precip"].astype(str).to_numpy() == "FAIL"
    ) | (
        station_df["qc_precsum"].astype(str).to_numpy() == "FAIL"
    )

    imerg_duration = np.full(imerg_end_min.size, 30.0, dtype=np.float64)
    imerg_fail = np.zeros(imerg_end_min.size, dtype=bool)
    imerg_direct_series = imerg_direct_mass[:, station_col]
    imerg_3x3_series = imerg_3x3_mass[:, station_col]

    rows: list[dict[str, object]] = []
    for window_min in WINDOWS_MIN:
        avamet_threshold = avamet_thresholds.get((station_id, int(window_min)))
        if avamet_threshold is None or not np.isfinite(avamet_threshold) or float(avamet_threshold) <= 0.0:
            continue

        avamet_metrics = compute_reference_window_metrics(
            source_end_min=source_end_min,
            source_duration_min=source_duration_min,
            source_mass=source_mass,
            usable_mask=usable_mask,
            fail_mask=fail_mask,
            ref_end_min=imerg_end_min,
            window_min=int(window_min),
        )
        imerg_direct_metrics = compute_reference_window_metrics(
            source_end_min=imerg_end_min,
            source_duration_min=imerg_duration,
            source_mass=np.where(np.isfinite(imerg_direct_series), imerg_direct_series, 0.0),
            usable_mask=np.isfinite(imerg_direct_series),
            fail_mask=imerg_fail,
            ref_end_min=imerg_end_min,
            window_min=int(window_min),
        )
        imerg_3x3_metrics = compute_reference_window_metrics(
            source_end_min=imerg_end_min,
            source_duration_min=imerg_duration,
            source_mass=np.where(np.isfinite(imerg_3x3_series), imerg_3x3_series, 0.0),
            usable_mask=np.isfinite(imerg_3x3_series),
            fail_mask=imerg_fail,
            ref_end_min=imerg_end_min,
            window_min=int(window_min),
        )

        point_valid_raw = avamet_metrics["complete_strict"] & imerg_direct_metrics["complete_strict"]
        grid_valid_raw = avamet_metrics["complete_strict"] & imerg_3x3_metrics["complete_strict"]

        imerg_threshold_point = compute_wet_quantile(
            imerg_direct_metrics["accum_mm"],
            imerg_direct_metrics["complete_strict"],
            imerg_quantile,
        )
        imerg_threshold_grid = compute_wet_quantile(
            imerg_3x3_metrics["accum_mm"],
            imerg_3x3_metrics["complete_strict"],
            imerg_quantile,
        )

        for stride_mode in STRIDE_MODES:
            stride_mask = compute_stride_mask(imerg_end_min, int(window_min), stride_mode)
            point_valid = point_valid_raw & stride_mask
            grid_valid = grid_valid_raw & stride_mask

            point_avamet_positive = point_valid & (avamet_metrics["accum_mm"] >= float(avamet_threshold))
            grid_avamet_positive = grid_valid & (avamet_metrics["accum_mm"] >= float(avamet_threshold))

            positive_masks: dict[str, tuple[np.ndarray, np.ndarray, float | None, float | None]] = {
                "fixed_threshold": (
                    point_valid & (imerg_direct_metrics["accum_mm"] >= float(avamet_threshold)),
                    grid_valid & (imerg_3x3_metrics["accum_mm"] >= float(avamet_threshold)),
                    float(avamet_threshold),
                    float(avamet_threshold),
                ),
                "relative_threshold": (
                    np.zeros(point_valid.size, dtype=bool),
                    np.zeros(grid_valid.size, dtype=bool),
                    None,
                    None,
                ),
            }

            if np.isfinite(imerg_threshold_point):
                positive_masks["relative_threshold"][0][:] = point_valid & (
                    imerg_direct_metrics["accum_mm"] >= float(imerg_threshold_point)
                )
            if np.isfinite(imerg_threshold_grid):
                positive_masks["relative_threshold"][1][:] = grid_valid & (
                    imerg_3x3_metrics["accum_mm"] >= float(imerg_threshold_grid)
                )
            positive_masks["relative_threshold"] = (
                positive_masks["relative_threshold"][0],
                positive_masks["relative_threshold"][1],
                float(imerg_threshold_point) if np.isfinite(imerg_threshold_point) else None,
                float(imerg_threshold_grid) if np.isfinite(imerg_threshold_grid) else None,
            )

            for threshold_mode in THRESHOLD_MODES:
                point_imerg_positive, grid_imerg_positive, point_threshold_used, grid_threshold_used = positive_masks[threshold_mode]

                point_tp_mask = point_avamet_positive & point_imerg_positive
                point_fn_mask = point_avamet_positive & (~point_imerg_positive)
                point_fp_mask = (~point_avamet_positive) & point_valid & point_imerg_positive
                point_tn_mask = (~point_avamet_positive) & point_valid & (~point_imerg_positive)

                grid_tp_mask = grid_avamet_positive & grid_imerg_positive
                grid_fn_mask = grid_avamet_positive & (~grid_imerg_positive)
                grid_fp_mask = (~grid_avamet_positive) & grid_valid & grid_imerg_positive
                grid_tn_mask = (~grid_avamet_positive) & grid_valid & (~grid_imerg_positive)

                point_tp = int(np.count_nonzero(point_tp_mask))
                point_fn = int(np.count_nonzero(point_fn_mask))
                point_fp = int(np.count_nonzero(point_fp_mask))
                point_tn = int(np.count_nonzero(point_tn_mask))

                grid_tp = int(np.count_nonzero(grid_tp_mask))
                grid_fn = int(np.count_nonzero(grid_fn_mask))
                grid_fp = int(np.count_nonzero(grid_fp_mask))
                grid_tn = int(np.count_nonzero(grid_tn_mask))

                if daily_block_counts is not None:
                    accumulate_block_counts(
                        accumulator=daily_block_counts,
                        key=(threshold_mode, stride_mode, int(window_min)),
                        block_codes=block_codes,
                        n_blocks=int(n_blocks),
                        point_tp_mask=point_tp_mask,
                        point_fp_mask=point_fp_mask,
                        point_fn_mask=point_fn_mask,
                        point_tn_mask=point_tn_mask,
                        grid_tp_mask=grid_tp_mask,
                        grid_fp_mask=grid_fp_mask,
                        grid_fn_mask=grid_fn_mask,
                        grid_tn_mask=grid_tn_mask,
                    )

                row = {
                    "station_id": station_id,
                    "window_min": int(window_min),
                    "stride_mode": stride_mode,
                    "threshold_mode": threshold_mode,
                    "avamet_threshold_mm": float(avamet_threshold),
                    "imerg_threshold_point_mm": point_threshold_used,
                    "imerg_threshold_3x3_mm": grid_threshold_used,
                    "imerg_quantile": float(imerg_quantile),
                    "valid_window_count_point_raw": int(np.count_nonzero(point_valid_raw)),
                    "valid_window_count_3x3_raw": int(np.count_nonzero(grid_valid_raw)),
                    "valid_window_count_point_used": int(np.count_nonzero(point_valid)),
                    "valid_window_count_3x3_used": int(np.count_nonzero(grid_valid)),
                    "avamet_positive_count_point": int(np.count_nonzero(point_avamet_positive)),
                    "avamet_positive_count_3x3": int(np.count_nonzero(grid_avamet_positive)),
                    "imerg_positive_count_point": int(np.count_nonzero(point_imerg_positive)),
                    "imerg_positive_count_3x3": int(np.count_nonzero(grid_imerg_positive)),
                }
                for key, value in summarize_confusion(point_tp, point_fp, point_fn, point_tn).items():
                    row[f"point_{key}"] = value
                for key, value in summarize_confusion(grid_tp, grid_fp, grid_fn, grid_tn).items():
                    row[f"grid3x3_{key}"] = value
                rows.append(row)

    return rows


def aggregate_by_scale(station_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["threshold_mode", "stride_mode", "window_min"]
    for (threshold_mode, stride_mode, window_min), group in station_rows.groupby(group_cols, sort=True):
        point_tp = int(group["point_tp"].sum())
        point_fp = int(group["point_fp"].sum())
        point_fn = int(group["point_fn"].sum())
        point_tn = int(group["point_tn"].sum())
        grid_tp = int(group["grid3x3_tp"].sum())
        grid_fp = int(group["grid3x3_fp"].sum())
        grid_fn = int(group["grid3x3_fn"].sum())
        grid_tn = int(group["grid3x3_tn"].sum())

        row = {
            "threshold_mode": threshold_mode,
            "stride_mode": stride_mode,
            "window_min": int(window_min),
            "station_count": int(group["station_id"].nunique()),
            "valid_window_count_point_raw": int(group["valid_window_count_point_raw"].sum()),
            "valid_window_count_3x3_raw": int(group["valid_window_count_3x3_raw"].sum()),
            "valid_window_count_point_used": int(group["valid_window_count_point_used"].sum()),
            "valid_window_count_3x3_used": int(group["valid_window_count_3x3_used"].sum()),
            "avamet_positive_count_point": int(group["avamet_positive_count_point"].sum()),
            "avamet_positive_count_3x3": int(group["avamet_positive_count_3x3"].sum()),
            "imerg_positive_count_point": int(group["imerg_positive_count_point"].sum()),
            "imerg_positive_count_3x3": int(group["imerg_positive_count_3x3"].sum()),
            "median_avamet_threshold_mm": float(group["avamet_threshold_mm"].median()),
            "median_imerg_threshold_point_mm": float(group["imerg_threshold_point_mm"].dropna().median())
            if group["imerg_threshold_point_mm"].notna().any()
            else None,
            "median_imerg_threshold_3x3_mm": float(group["imerg_threshold_3x3_mm"].dropna().median())
            if group["imerg_threshold_3x3_mm"].notna().any()
            else None,
        }
        for key, value in summarize_confusion(point_tp, point_fp, point_fn, point_tn).items():
            row[f"point_{key}"] = value
        for key, value in summarize_confusion(grid_tp, grid_fp, grid_fn, grid_tn).items():
            row[f"grid3x3_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_station_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_output_json.parent.mkdir(parents=True, exist_ok=True)

    avamet_thresholds = load_avamet_thresholds(args.station_scale_summary_csv)
    subset_index = build_subset_index(args.subset_root)
    block_dates = pd.to_datetime(subset_index["end_ts"], errors="coerce").dt.normalize()
    unique_block_dates, block_codes = np.unique(
        block_dates.to_numpy(dtype="datetime64[D]"),
        return_inverse=True,
    )
    block_codes = block_codes.astype(np.int32, copy=False)
    n_blocks = int(unique_block_dates.size)
    station_ids, lon_local, lat_local = load_station_cells(args.station_alignment_csv)
    if args.station_limit > 0:
        station_ids = station_ids[: int(args.station_limit)]
        lon_local = lon_local[: int(args.station_limit)]
        lat_local = lat_local[: int(args.station_limit)]
    station_to_col = {station_id: idx for idx, station_id in enumerate(station_ids)}

    imerg_end_min, imerg_direct_mass, imerg_3x3_mass = build_imerg_station_masses(
        subset_index=subset_index,
        lon_local=lon_local,
        lat_local=lat_local,
    )

    parquet = pq.ParquetFile(args.qc_input)
    columns = [
        "station_id",
        "timestamp_utc",
        "delta_prev_min",
        "precip_mm_qc",
        "usable_interval_extremes",
        "qc_precip",
        "qc_precsum",
    ]

    current_parts: list[pd.DataFrame] = []
    current_station: str | None = None
    processed_stations = 0
    processed_target_stations = 0
    target_station_count = len(station_to_col)
    station_rows_accum: list[dict[str, object]] = []
    daily_block_counts: dict[tuple[str, str, int], np.ndarray] = {}

    def flush_station(station_id: str, parts: list[pd.DataFrame]) -> bool:
        nonlocal processed_stations, processed_target_stations
        if station_id not in station_to_col:
            processed_stations += 1
            return False

        station_df = pd.concat(parts, ignore_index=True)
        rows = process_station(
            station_df=station_df,
            station_id=station_id,
            station_col=station_to_col[station_id],
            avamet_thresholds=avamet_thresholds,
            imerg_end_min=imerg_end_min,
            imerg_direct_mass=imerg_direct_mass,
            imerg_3x3_mass=imerg_3x3_mass,
            imerg_quantile=float(args.imerg_quantile),
            block_codes=block_codes,
            n_blocks=n_blocks,
            daily_block_counts=daily_block_counts,
        )
        station_rows_accum.extend(rows)
        processed_stations += 1
        processed_target_stations += 1
        if processed_stations % 25 == 0:
            print(f"Processed AVAMET stations: {processed_stations}")
        return args.station_limit > 0 and processed_target_stations >= target_station_count

    stop_early = False
    for batch in parquet.iter_batches(batch_size=int(args.batch_size), columns=columns):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        for station_id, station_chunk in batch_df.groupby("station_id", sort=False):
            station_id = str(station_id)
            if current_station is None:
                current_station = station_id

            if station_id != current_station:
                stop_early = flush_station(current_station, current_parts)
                current_parts = []
                current_station = station_id
                if stop_early:
                    break

            current_parts.append(station_chunk.copy())

        if stop_early:
            break

    if not stop_early and current_station is not None and current_parts:
        flush_station(current_station, current_parts)

    station_rows = pd.DataFrame(station_rows_accum)
    if station_rows.empty:
        raise RuntimeError("No robust confusion rows were generated.")

    station_rows = station_rows.sort_values(["threshold_mode", "stride_mode", "window_min", "station_id"]).reset_index(drop=True)
    station_rows.to_csv(args.output_station_csv, index=False)

    by_scale = aggregate_by_scale(station_rows)
    by_scale.to_csv(args.output_csv, index=False)

    bootstrap_df, bootstrap_json_rows = bootstrap_rows_from_counts(
        daily_block_counts=daily_block_counts,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    bootstrap_df.to_csv(args.bootstrap_output_csv, index=False)

    bootstrap_summary = {
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "block_definition": "UTC day blocks from IMERG reference-window end times",
        "block_count": int(n_blocks),
        "block_dates_utc": [str(pd.Timestamp(day).date()) for day in unique_block_dates],
        "rows": bootstrap_json_rows,
    }
    args.bootstrap_output_json.write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")

    summary = {
        "imerg_subset_window_count": int(subset_index.shape[0]),
        "station_count_processed": int(station_rows["station_id"].nunique()),
        "threshold_modes": list(THRESHOLD_MODES),
        "stride_modes": list(STRIDE_MODES),
        "imerg_quantile": float(args.imerg_quantile),
        "bootstrap_output_csv": str(args.bootstrap_output_csv),
        "bootstrap_output_json": str(args.bootstrap_output_json),
        "scale_rows": by_scale.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_station_csv}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.bootstrap_output_csv}")
    print(f"Wrote: {args.bootstrap_output_json}")
    print(f"Stations processed: {station_rows['station_id'].nunique()}")


if __name__ == "__main__":
    main()

