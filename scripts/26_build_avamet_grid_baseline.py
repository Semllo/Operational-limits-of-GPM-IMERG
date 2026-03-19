from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AVAMET-to-IMERG-grid representativeness baseline by comparing station windows "
            "against the AVAMET median within the corresponding IMERG cell."
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
        help="Root directory with IMERG subset files, used only to define the 30 min reference timeline.",
    )
    parser.add_argument(
        "--windows-min",
        type=str,
        default="30,60,180,360",
        help="Comma-separated list of window sizes (minutes) to evaluate. Default runs the full multi-scale baseline.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_grid_baseline_by_scale.csv"),
        help="Scale-level representativeness baseline CSV.",
    )
    parser.add_argument(
        "--output-station-csv",
        type=Path,
        default=Path("results/avamet_grid_baseline_station_scale.csv"),
        help="Per-station representativeness baseline CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_grid_baseline_summary.json"),
        help="Representativeness baseline summary JSON.",
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
        help="Optional station limit for testing.",
    )
    return parser


def parse_windows_min(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one window size must be provided.")
    values = tuple(int(part) for part in parts)
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"Window sizes must be positive integers, got: {invalid}")
    return values


def to_minute_index(values: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    ts = pd.to_datetime(values, errors="coerce", utc=False)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def build_reference_end_times(subset_root: Path) -> np.ndarray:
    paths = sorted(subset_root.rglob("*.cv.npz"))
    if not paths:
        raise FileNotFoundError(f"No IMERG subset files found under {subset_root}")

    records: list[tuple[pd.Timestamp, int, str]] = []
    for path in paths:
        name = path.name
        date_part = name.split(".")[4] if len(name.split(".")) > 4 else ""
        if "-S" not in date_part:
            date_part = name
        try:
            date_token = name.split(".")[4]
        except IndexError:
            raise ValueError(f"Unexpected IMERG filename format: {name}")
        # Example token: 20230101-S000000-E002959
        day_str, rest = date_token.split("-S", 1)
        start_str = rest.split("-E", 1)[0]
        start_ts = pd.to_datetime(day_str + start_str, format="%Y%m%d%H%M%S", errors="raise")
        end_ts = start_ts + pd.Timedelta(minutes=30)
        records.append((end_ts, len(path.parts), str(path)))

    ref = pd.DataFrame(records, columns=["end_ts", "path_depth", "path"]).sort_values(
        ["end_ts", "path_depth", "path"], ascending=[True, False, True]
    )
    ref = ref.drop_duplicates(subset=["end_ts"], keep="first").reset_index(drop=True)
    return to_minute_index(ref["end_ts"])


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


def load_station_cells(alignment_csv: Path) -> tuple[list[str], list[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    alignment = pd.read_csv(alignment_csv)
    alignment["station_id"] = alignment["station_id"].astype(str)
    alignment = alignment.sort_values("station_id").reset_index(drop=True)

    required = ["station_id", "imerg_cv_lon_idx", "imerg_cv_lat_idx"]
    missing = [column for column in required if column not in alignment.columns]
    if missing:
        raise KeyError(f"Missing required columns in {alignment_csv}: {missing}")

    station_ids = alignment["station_id"].tolist()
    cell_keys = list(
        zip(
            alignment["imerg_cv_lon_idx"].astype(int).tolist(),
            alignment["imerg_cv_lat_idx"].astype(int).tolist(),
        )
    )

    groups: dict[tuple[int, int], list[int]] = {}
    for idx, cell_key in enumerate(cell_keys):
        groups.setdefault(cell_key, []).append(idx)
    return station_ids, cell_keys, groups


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
        excess_before = window_start[ref_idx] - source_start_min[src_idx]
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
        overlap_start = np.maximum(window_start[ref_idx], source_start_min[src_idx])
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
        "accum_mm": accum.astype(np.float32, copy=False),
        "coverage_ratio": coverage_ratio.astype(np.float32, copy=False),
        "fail_count": fail_count.astype(np.int32, copy=False),
        "complete_strict": complete_strict.astype(bool, copy=False),
    }


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
    else:
        random_hits = None
        ets = None
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
        "total_windows": int(total),
        "random_hits": random_hits,
    }


def build_station_window_arrays(
    qc_input: Path,
    station_ids: list[str],
    thresholds: dict[tuple[str, int], float],
    ref_end_min: np.ndarray,
    windows_min: tuple[int, ...],
    batch_size: int,
    station_limit: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    selected_station_ids = station_ids[: station_limit] if station_limit > 0 else station_ids
    selected_set = set(selected_station_ids)
    station_to_col = {station_id: idx for idx, station_id in enumerate(selected_station_ids)}
    n_ref = int(ref_end_min.size)
    n_stations = int(len(selected_station_ids))

    accum_by_scale = {window_min: np.full((n_ref, n_stations), np.nan, dtype=np.float32) for window_min in windows_min}
    complete_by_scale = {window_min: np.zeros((n_ref, n_stations), dtype=bool) for window_min in windows_min}

    parquet = pq.ParquetFile(qc_input)
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
    processed_target = 0

    def flush_station(station_id: str, parts: list[pd.DataFrame]) -> bool:
        nonlocal processed_target
        if station_id not in station_to_col:
            return False

        station_df = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
        source_end_min = to_minute_index(station_df["timestamp_utc"])
        source_duration_min = pd.to_numeric(station_df["delta_prev_min"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        source_duration_min = np.where(
            np.isfinite(source_duration_min) & (source_duration_min > 0), source_duration_min, 0.0
        )
        source_mass = pd.to_numeric(station_df["precip_mm_qc"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        source_mass = np.where(np.isfinite(source_mass), source_mass, 0.0)
        usable_mask = station_df["usable_interval_extremes"].fillna(False).to_numpy(dtype=bool, copy=False)
        fail_mask = (
            station_df["qc_precip"].astype(str).to_numpy() == "FAIL"
        ) | (
            station_df["qc_precsum"].astype(str).to_numpy() == "FAIL"
        )

        col = station_to_col[station_id]
        for window_min in windows_min:
            if (station_id, int(window_min)) not in thresholds:
                continue
            metrics = compute_reference_window_metrics(
                source_end_min=source_end_min,
                source_duration_min=source_duration_min,
                source_mass=source_mass,
                usable_mask=usable_mask,
                fail_mask=fail_mask,
                ref_end_min=ref_end_min,
                window_min=int(window_min),
            )
            accum_by_scale[window_min][:, col] = metrics["accum_mm"]
            complete_by_scale[window_min][:, col] = metrics["complete_strict"]

        processed_target += 1
        if processed_target % 25 == 0:
            print(f"Processed AVAMET stations for baseline: {processed_target}")
        return station_limit > 0 and processed_target >= len(selected_station_ids)

    stop_early = False
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue
        batch_df["station_id"] = batch_df["station_id"].astype(str)

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
            if station_id in selected_set:
                current_parts.append(station_chunk.copy())

        if stop_early:
            break

    if not stop_early and current_station is not None and current_parts:
        flush_station(current_station, current_parts)

    return accum_by_scale, complete_by_scale, selected_station_ids


def build_cell_medians(
    accum_by_scale: dict[int, np.ndarray],
    complete_by_scale: dict[int, np.ndarray],
    cell_groups: dict[tuple[int, int], list[int]],
    n_stations: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    n_ref = next(iter(accum_by_scale.values())).shape[0]
    cell_accum_by_scale = {window_min: np.full((n_ref, n_stations), np.nan, dtype=np.float32) for window_min in accum_by_scale}
    cell_complete_by_scale = {window_min: np.zeros((n_ref, n_stations), dtype=bool) for window_min in accum_by_scale}
    station_cell_count = np.ones(n_stations, dtype=np.int32)

    for station_indices in cell_groups.values():
        station_indices_arr = np.asarray(station_indices, dtype=np.int64)
        group_size = int(station_indices_arr.size)
        station_cell_count[station_indices_arr] = group_size

        for window_min in accum_by_scale:
            accum = accum_by_scale[window_min][:, station_indices_arr]
            complete = complete_by_scale[window_min][:, station_indices_arr]

            if group_size == 1:
                cell_series = accum[:, 0]
                cell_complete = complete[:, 0]
            else:
                masked = np.where(complete, accum, np.nan).astype(np.float32, copy=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cell_series = np.nanmedian(masked, axis=1).astype(np.float32, copy=False)
                cell_complete = np.isfinite(cell_series)

            cell_accum_by_scale[window_min][:, station_indices_arr] = cell_series[:, None]
            cell_complete_by_scale[window_min][:, station_indices_arr] = cell_complete[:, None]

    return cell_accum_by_scale, cell_complete_by_scale, station_cell_count


def build_baseline_rows(
    station_ids: list[str],
    windows_min: tuple[int, ...],
    thresholds: dict[tuple[str, int], float],
    accum_by_scale: dict[int, np.ndarray],
    complete_by_scale: dict[int, np.ndarray],
    cell_accum_by_scale: dict[int, np.ndarray],
    cell_complete_by_scale: dict[int, np.ndarray],
    station_cell_count: np.ndarray,
    cell_groups: dict[tuple[int, int], list[int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for station_idx, station_id in enumerate(station_ids):
        cell_station_count = int(station_cell_count[station_idx])
        for window_min in windows_min:
            threshold = thresholds.get((station_id, int(window_min)))
            if threshold is None:
                continue

            point_accum = accum_by_scale[window_min][:, station_idx].astype(np.float64, copy=False)
            point_complete = complete_by_scale[window_min][:, station_idx]
            cell_accum = cell_accum_by_scale[window_min][:, station_idx].astype(np.float64, copy=False)
            cell_complete = cell_complete_by_scale[window_min][:, station_idx]

            valid = point_complete & cell_complete
            point_positive = valid & (point_accum >= float(threshold))
            cell_positive = valid & (cell_accum >= float(threshold))

            tp = int(np.count_nonzero(point_positive & cell_positive))
            fn = int(np.count_nonzero(point_positive & (~cell_positive)))
            fp = int(np.count_nonzero((~point_positive) & valid & cell_positive))
            tn = int(np.count_nonzero((~point_positive) & valid & (~cell_positive)))

            diff = cell_accum - point_accum
            diff_valid = diff[valid]
            diff_positive = diff[point_positive]

            row = {
                "station_id": station_id,
                "window_min": int(window_min),
                "subset_type": "all_cells",
                "cell_station_count": cell_station_count,
                "threshold_mm": float(threshold),
                "valid_window_count": int(np.count_nonzero(valid)),
                "point_positive_count": int(np.count_nonzero(point_positive)),
                "cell_positive_count": int(np.count_nonzero(cell_positive)),
                "median_bias_mm_all_valid": float(np.nanmedian(diff_valid)) if diff_valid.size else None,
                "mae_mm_all_valid": float(np.nanmean(np.abs(diff_valid))) if diff_valid.size else None,
                "median_bias_mm_point_positive": float(np.nanmedian(diff_positive)) if diff_positive.size else None,
                "mae_mm_point_positive": float(np.nanmean(np.abs(diff_positive))) if diff_positive.size else None,
            }
            for key, value in summarize_confusion(tp, fp, fn, tn).items():
                row[key] = value
            rows.append(row)

            if cell_station_count >= 2:
                multi_row = row.copy()
                multi_row["subset_type"] = "multi_station_cells"
                rows.append(multi_row)

    # Leave-one-out areal baseline for multi-station cells only.
    for station_indices in cell_groups.values():
        station_indices_arr = np.asarray(station_indices, dtype=np.int64)
        group_size = int(station_indices_arr.size)
        if group_size < 2:
            continue

        for station_idx in station_indices_arr.tolist():
            station_id = station_ids[int(station_idx)]
            others = station_indices_arr[station_indices_arr != int(station_idx)]
            if others.size == 0:
                continue

            for window_min in windows_min:
                threshold = thresholds.get((station_id, int(window_min)))
                if threshold is None:
                    continue

                point_accum = accum_by_scale[window_min][:, station_idx].astype(np.float64, copy=False)
                point_complete = complete_by_scale[window_min][:, station_idx]

                other_accum = accum_by_scale[window_min][:, others].astype(np.float32, copy=False)
                other_complete = complete_by_scale[window_min][:, others]
                masked_other = np.where(other_complete, other_accum, np.nan).astype(np.float32, copy=False)

                if masked_other.ndim == 1 or masked_other.shape[1] == 1:
                    loo_cell_accum = np.ravel(masked_other).astype(np.float64, copy=False)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        loo_cell_accum = np.nanmedian(masked_other, axis=1).astype(np.float64, copy=False)
                loo_cell_complete = np.isfinite(loo_cell_accum)

                valid = point_complete & loo_cell_complete
                point_positive = valid & (point_accum >= float(threshold))
                cell_positive = valid & (loo_cell_accum >= float(threshold))

                tp = int(np.count_nonzero(point_positive & cell_positive))
                fn = int(np.count_nonzero(point_positive & (~cell_positive)))
                fp = int(np.count_nonzero((~point_positive) & valid & cell_positive))
                tn = int(np.count_nonzero((~point_positive) & valid & (~cell_positive)))

                diff = loo_cell_accum - point_accum
                diff_valid = diff[valid]
                diff_positive = diff[point_positive]

                row = {
                    "station_id": station_id,
                    "window_min": int(window_min),
                    "subset_type": "leave_one_out_multi_station_cells",
                    "cell_station_count": group_size,
                    "threshold_mm": float(threshold),
                    "valid_window_count": int(np.count_nonzero(valid)),
                    "point_positive_count": int(np.count_nonzero(point_positive)),
                    "cell_positive_count": int(np.count_nonzero(cell_positive)),
                    "median_bias_mm_all_valid": float(np.nanmedian(diff_valid)) if diff_valid.size else None,
                    "mae_mm_all_valid": float(np.nanmean(np.abs(diff_valid))) if diff_valid.size else None,
                    "median_bias_mm_point_positive": float(np.nanmedian(diff_positive)) if diff_positive.size else None,
                    "mae_mm_point_positive": float(np.nanmean(np.abs(diff_positive))) if diff_positive.size else None,
                }
                for key, value in summarize_confusion(tp, fp, fn, tn).items():
                    row[key] = value
                rows.append(row)

    baseline_df = pd.DataFrame(rows)
    if baseline_df.empty:
        raise RuntimeError("No AVAMET grid baseline rows were generated.")
    return baseline_df


def aggregate_baseline(station_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subset_type, window_min), group in station_rows.groupby(["subset_type", "window_min"], sort=True):
        tp = int(group["tp"].sum())
        fp = int(group["fp"].sum())
        fn = int(group["fn"].sum())
        tn = int(group["tn"].sum())

        row = {
            "subset_type": subset_type,
            "window_min": int(window_min),
            "station_count": int(group["station_id"].nunique()),
            "median_cell_station_count": float(group["cell_station_count"].median()),
            "valid_window_count": int(group["valid_window_count"].sum()),
            "point_positive_count": int(group["point_positive_count"].sum()),
            "cell_positive_count": int(group["cell_positive_count"].sum()),
            "median_threshold_mm": float(group["threshold_mm"].median()),
            "median_bias_mm_all_valid": float(group["median_bias_mm_all_valid"].dropna().median())
            if group["median_bias_mm_all_valid"].notna().any()
            else None,
            "median_bias_mm_point_positive": float(group["median_bias_mm_point_positive"].dropna().median())
            if group["median_bias_mm_point_positive"].notna().any()
            else None,
            "median_mae_mm_all_valid": float(group["mae_mm_all_valid"].dropna().median())
            if group["mae_mm_all_valid"].notna().any()
            else None,
            "median_mae_mm_point_positive": float(group["mae_mm_point_positive"].dropna().median())
            if group["mae_mm_point_positive"].notna().any()
            else None,
        }
        for key, value in summarize_confusion(tp, fp, fn, tn).items():
            row[key] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["subset_type", "window_min"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_station_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    windows_min = parse_windows_min(args.windows_min)
    thresholds = load_avamet_thresholds(args.station_scale_summary_csv)
    ref_end_min = build_reference_end_times(args.subset_root)
    station_ids_all, cell_keys_all, cell_groups_all = load_station_cells(args.station_alignment_csv)

    selected_station_ids = station_ids_all[: int(args.station_limit)] if args.station_limit > 0 else station_ids_all
    selected_set = set(selected_station_ids)
    selected_station_ids, selected_cell_keys, selected_groups = (
        selected_station_ids,
        [cell_keys_all[idx] for idx, station_id in enumerate(station_ids_all) if station_id in selected_set],
        {},
    )
    selected_index_map = {station_id: idx for idx, station_id in enumerate(selected_station_ids)}
    for original_idx, station_id in enumerate(station_ids_all):
        if station_id not in selected_index_map:
            continue
        cell_key = cell_keys_all[original_idx]
        selected_groups.setdefault(cell_key, []).append(selected_index_map[station_id])

    accum_by_scale, complete_by_scale, selected_station_ids = build_station_window_arrays(
        qc_input=args.qc_input,
        station_ids=station_ids_all,
        thresholds=thresholds,
        ref_end_min=ref_end_min,
        windows_min=windows_min,
        batch_size=int(args.batch_size),
        station_limit=int(args.station_limit),
    )

    cell_accum_by_scale, cell_complete_by_scale, station_cell_count = build_cell_medians(
        accum_by_scale=accum_by_scale,
        complete_by_scale=complete_by_scale,
        cell_groups=selected_groups,
        n_stations=len(selected_station_ids),
    )

    station_rows = build_baseline_rows(
        station_ids=selected_station_ids,
        windows_min=windows_min,
        thresholds=thresholds,
        accum_by_scale=accum_by_scale,
        complete_by_scale=complete_by_scale,
        cell_accum_by_scale=cell_accum_by_scale,
        cell_complete_by_scale=cell_complete_by_scale,
        station_cell_count=station_cell_count,
        cell_groups=selected_groups,
    )
    station_rows = station_rows.sort_values(["subset_type", "window_min", "station_id"]).reset_index(drop=True)
    station_rows.to_csv(args.output_station_csv, index=False)

    by_scale = aggregate_baseline(station_rows)
    by_scale.to_csv(args.output_csv, index=False)

    summary = {
        "windows_min": list(windows_min),
        "reference_window_count": int(ref_end_min.size),
        "station_count_processed": int(len(selected_station_ids)),
        "subset_rows": by_scale.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_station_csv}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Stations processed: {len(selected_station_ids)}")


if __name__ == "__main__":
    main()

