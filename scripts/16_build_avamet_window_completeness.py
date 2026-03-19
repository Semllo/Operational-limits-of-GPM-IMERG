from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WINDOWS_MIN = (30, 60, 180, 360)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-station window completeness summaries for AVAMET precipitation extremes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="Input QC parquet.",
    )
    parser.add_argument(
        "--station-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_window_completeness_station_summary.csv"),
        help="Per-station window completeness summary CSV.",
    )
    parser.add_argument(
        "--global-summary-json",
        type=Path,
        default=Path("results/avamet_cv_window_completeness_summary.json"),
        help="Global window completeness summary JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200000,
        help="Row batch size for parquet streaming.",
    )
    parser.add_argument(
        "--station-limit",
        type=int,
        default=0,
        help="Optional limit on number of stations to process (for testing).",
    )
    return parser


def compute_window_metrics(
    end_min: np.ndarray,
    duration_min: np.ndarray,
    precip_mm_qc: np.ndarray,
    usable: np.ndarray,
    fail_interval: np.ndarray,
    suspect_interval: np.ndarray,
    window_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = end_min.size
    start_min = end_min - duration_min

    usable_duration = np.where(usable, duration_min.astype(np.float64), 0.0)
    usable_mass = np.where(usable, precip_mm_qc, 0.0)
    fail_countable = (fail_interval & (duration_min > 0)).astype(np.int32)
    suspect_countable = (suspect_interval & (duration_min > 0)).astype(np.int32)

    prefix_duration = np.concatenate(([0.0], np.cumsum(usable_duration, dtype=np.float64)))
    prefix_mass = np.concatenate(([0.0], np.cumsum(usable_mass, dtype=np.float64)))
    prefix_fail = np.concatenate(([0], np.cumsum(fail_countable, dtype=np.int64)))
    prefix_suspect = np.concatenate(([0], np.cumsum(suspect_countable, dtype=np.int64)))

    window_start = end_min - int(window_min)
    left_idx = np.searchsorted(end_min, window_start, side="right")
    idx = np.arange(n, dtype=np.int64)

    full_duration = prefix_duration[idx + 1] - prefix_duration[left_idx + 1]
    full_mass = prefix_mass[idx + 1] - prefix_mass[left_idx + 1]
    full_fail = prefix_fail[idx + 1] - prefix_fail[left_idx + 1]
    full_suspect = prefix_suspect[idx + 1] - prefix_suspect[left_idx + 1]

    left_duration = duration_min[left_idx].astype(np.float64)
    left_end = end_min[left_idx]
    left_start = start_min[left_idx]
    left_overlap = (left_end - np.maximum(left_start, window_start)).astype(np.float64)
    left_overlap = np.clip(left_overlap, 0.0, None)

    left_fraction = np.zeros(n, dtype=np.float64)
    valid_left = left_duration > 0
    left_fraction[valid_left] = np.clip(left_overlap[valid_left] / left_duration[valid_left], 0.0, 1.0)

    partial_duration = left_fraction * usable_duration[left_idx]
    partial_mass = left_fraction * usable_mass[left_idx]
    partial_fail = np.where((left_overlap > 0) & (fail_countable[left_idx] > 0), 1, 0)
    partial_suspect = np.where((left_overlap > 0) & (suspect_countable[left_idx] > 0), 1, 0)

    coverage_ratio = np.clip((full_duration + partial_duration) / float(window_min), 0.0, 1.0)
    accum_mm = full_mass + partial_mass
    fail_count = full_fail + partial_fail
    suspect_count = full_suspect + partial_suspect
    complete_strict = (coverage_ratio >= 0.95) & (fail_count == 0)
    complete_loose = coverage_ratio >= 0.80

    return coverage_ratio, accum_mm, fail_count, suspect_count, complete_strict, complete_loose


def to_minute_index(ts: pd.Series) -> np.ndarray:
    ts_parsed = pd.to_datetime(ts, errors="coerce", utc=False)
    ts_ns = ts_parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def build_station_summary(station_df: pd.DataFrame) -> dict[str, object]:
    station_df = station_df.sort_values("timestamp_utc").reset_index(drop=True)
    n_rows = int(station_df.shape[0])

    end_min = to_minute_index(station_df["timestamp_utc"])

    duration_min = pd.to_numeric(station_df["delta_prev_min"], errors="coerce").to_numpy(dtype=np.float64)
    duration_min = np.where(np.isfinite(duration_min) & (duration_min > 0), duration_min, 0.0)

    precip_mm_qc = pd.to_numeric(station_df["precip_mm_qc"], errors="coerce").to_numpy(dtype=np.float64)
    precip_mm_qc = np.where(np.isfinite(precip_mm_qc), precip_mm_qc, 0.0)

    usable = station_df["usable_interval_extremes"].fillna(False).to_numpy(dtype=bool)
    fail_interval = (
        station_df["qc_precip"].astype(str).to_numpy() == "FAIL"
    ) | (
        station_df["qc_precsum"].astype(str).to_numpy() == "FAIL"
    )
    suspect_interval = (
        station_df["qc_precip"].astype(str).to_numpy() == "SUSPECT"
    ) | (
        station_df["qc_precsum"].astype(str).to_numpy() == "SUSPECT"
    )

    summary: dict[str, object] = {
        "station_id": str(station_df["station_id"].iloc[0]),
        "n_rows": n_rows,
        "n_usable_interval_extremes": int(usable.sum()),
    }

    for window_min in WINDOWS_MIN:
        coverage_ratio, accum_mm, fail_count, suspect_count, complete_strict, complete_loose = compute_window_metrics(
            end_min=end_min,
            duration_min=duration_min,
            precip_mm_qc=precip_mm_qc,
            usable=usable,
            fail_interval=fail_interval,
            suspect_interval=suspect_interval,
            window_min=window_min,
        )

        prefix = f"w{window_min}"
        summary[f"n_windows_{prefix}"] = n_rows
        summary[f"n_complete_{prefix}_strict"] = int(complete_strict.sum())
        summary[f"n_complete_{prefix}_loose"] = int(complete_loose.sum())
        summary[f"n_with_fail_{prefix}"] = int((fail_count > 0).sum())
        summary[f"n_with_suspect_{prefix}"] = int((suspect_count > 0).sum())
        summary[f"median_coverage_{prefix}"] = float(np.nanmedian(coverage_ratio)) if coverage_ratio.size else float("nan")
        summary[f"p95_coverage_{prefix}"] = float(np.nanquantile(coverage_ratio, 0.95)) if coverage_ratio.size else float("nan")

        strict_accum = accum_mm[complete_strict]
        summary[f"p99_accum_mm_{prefix}_strict"] = (
            float(np.nanquantile(strict_accum, 0.99)) if strict_accum.size else float("nan")
        )
        summary[f"max_accum_mm_{prefix}_strict"] = (
            float(np.nanmax(strict_accum)) if strict_accum.size else float("nan")
        )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.station_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.global_summary_json.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(args.input)
    columns = [
        "station_id",
        "timestamp_utc",
        "delta_prev_min",
        "precip_mm_qc",
        "usable_interval_extremes",
        "qc_precip",
        "qc_precsum",
    ]

    summaries: list[dict[str, object]] = []
    current_parts: list[pd.DataFrame] = []
    current_station: str | None = None
    processed_stations = 0

    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        for station_id, station_chunk in batch_df.groupby("station_id", sort=False):
            station_id = str(station_id)
            if current_station is None:
                current_station = station_id

            if station_id != current_station:
                station_df = pd.concat(current_parts, ignore_index=True)
                summaries.append(build_station_summary(station_df))
                processed_stations += 1
                if args.station_limit > 0 and processed_stations >= args.station_limit:
                    break
                current_parts = []
                current_station = station_id

            current_parts.append(station_chunk.copy())

        if args.station_limit > 0 and processed_stations >= args.station_limit:
            break

    if (args.station_limit == 0 or processed_stations < args.station_limit) and current_parts:
        station_df = pd.concat(current_parts, ignore_index=True)
        summaries.append(build_station_summary(station_df))
        processed_stations += 1

    if not summaries:
        raise RuntimeError("No station summaries were generated.")

    summary_df = pd.DataFrame(summaries).sort_values("station_id").reset_index(drop=True)
    summary_df.to_csv(args.station_summary_csv, index=False)

    global_summary: dict[str, object] = {
        "input": str(args.input),
        "station_count": int(summary_df.shape[0]),
        "windows_min": list(WINDOWS_MIN),
        "counts": {},
    }

    total_rows = int(summary_df["n_rows"].sum())
    total_usable = int(summary_df["n_usable_interval_extremes"].sum())
    global_summary["counts"]["n_rows"] = total_rows
    global_summary["counts"]["n_usable_interval_extremes"] = total_usable

    for window_min in WINDOWS_MIN:
        prefix = f"w{window_min}"
        global_summary["counts"][prefix] = {
            "n_windows": int(summary_df[f"n_windows_{prefix}"].sum()),
            "n_complete_strict": int(summary_df[f"n_complete_{prefix}_strict"].sum()),
            "n_complete_loose": int(summary_df[f"n_complete_{prefix}_loose"].sum()),
            "n_with_fail": int(summary_df[f"n_with_fail_{prefix}"].sum()),
            "n_with_suspect": int(summary_df[f"n_with_suspect_{prefix}"].sum()),
            "median_station_median_coverage": float(summary_df[f"median_coverage_{prefix}"].median()),
            "median_station_p95_coverage": float(summary_df[f"p95_coverage_{prefix}"].median()),
        }

    args.global_summary_json.write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.station_summary_csv}")
    print(f"Wrote: {args.global_summary_json}")
    print(f"Stations processed: {processed_stations}")


if __name__ == "__main__":
    main()

