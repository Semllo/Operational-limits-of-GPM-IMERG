from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WINDOWS_MIN = (30, 60, 180, 360)
SUPPORTED_DELTAS = (4, 5, 6, 10, 15, 20, 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a POT/declustered AVAMET event catalog for sub-hourly precipitation extremes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="Input QC parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/avamet_cv_events.parquet"),
        help="Output event catalog parquet.",
    )
    parser.add_argument(
        "--station-scale-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_events_station_scale_summary.csv"),
        help="Per-station, per-scale event summary CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/avamet_cv_events_summary.json"),
        help="Global event catalog summary JSON.",
    )
    parser.add_argument(
        "--pot-quantile",
        type=float,
        default=0.995,
        help="Wet-window POT threshold quantile, computed per station and scale.",
    )
    parser.add_argument(
        "--windows-min",
        type=str,
        default="30,60,180,360",
        help="Comma-separated window sizes (minutes) to process.",
    )
    parser.add_argument(
        "--completeness-mode",
        choices=("strict", "loose"),
        default="strict",
        help="Whether the event catalog should use strict or loose completeness gating.",
    )
    parser.add_argument(
        "--decluster-gap-short-h",
        type=float,
        default=6.0,
        help="Declustering separation (hours) for 30/60 min windows.",
    )
    parser.add_argument(
        "--decluster-gap-long-h",
        type=float,
        default=12.0,
        help="Declustering separation (hours) for 180/360 min windows.",
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


def _decluster_gap_min(window_min: int, short_h: float, long_h: float) -> int:
    if window_min <= 60:
        return int(round(short_h * 60.0))
    return int(round(long_h * 60.0))


def parse_windows_min(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one window size must be provided in --windows-min.")
    values = tuple(int(part) for part in parts)
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"Window sizes must be positive integers, got: {invalid}")
    return values


def _to_minute_index(ts: pd.Series) -> np.ndarray:
    ts_parsed = pd.to_datetime(ts, errors="coerce", utc=False)
    ts_ns = ts_parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def compute_window_metrics(
    end_min: np.ndarray,
    duration_min: np.ndarray,
    precip_mm_qc: np.ndarray,
    usable: np.ndarray,
    fail_interval: np.ndarray,
    suspect_interval: np.ndarray,
    delta_int: np.ndarray,
    window_min: int,
) -> dict[str, np.ndarray]:
    n = end_min.size
    start_min = end_min - duration_min

    usable_duration = np.where(usable, duration_min.astype(np.float64), 0.0)
    usable_mass = np.where(usable, precip_mm_qc, 0.0)
    fail_countable = (fail_interval & (duration_min > 0)).astype(np.int32)
    suspect_countable = (suspect_interval & (duration_min > 0)).astype(np.int32)
    interval_countable = (duration_min > 0).astype(np.int32)
    usable_interval_countable = (usable & (duration_min > 0)).astype(np.int32)

    prefix_duration = np.concatenate(([0.0], np.cumsum(usable_duration, dtype=np.float64)))
    prefix_mass = np.concatenate(([0.0], np.cumsum(usable_mass, dtype=np.float64)))
    prefix_fail = np.concatenate(([0], np.cumsum(fail_countable, dtype=np.int64)))
    prefix_suspect = np.concatenate(([0], np.cumsum(suspect_countable, dtype=np.int64)))
    prefix_interval_count = np.concatenate(([0], np.cumsum(interval_countable, dtype=np.int64)))
    prefix_usable_interval_count = np.concatenate(([0], np.cumsum(usable_interval_countable, dtype=np.int64)))

    window_start = end_min - int(window_min)
    left_idx = np.searchsorted(end_min, window_start, side="right")
    idx = np.arange(n, dtype=np.int64)

    full_duration = prefix_duration[idx + 1] - prefix_duration[left_idx + 1]
    full_mass = prefix_mass[idx + 1] - prefix_mass[left_idx + 1]
    full_fail = prefix_fail[idx + 1] - prefix_fail[left_idx + 1]
    full_suspect = prefix_suspect[idx + 1] - prefix_suspect[left_idx + 1]
    full_interval_count = prefix_interval_count[idx + 1] - prefix_interval_count[left_idx + 1]
    full_usable_interval_count = prefix_usable_interval_count[idx + 1] - prefix_usable_interval_count[left_idx + 1]

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
    partial_interval_count = np.where((left_overlap > 0) & (interval_countable[left_idx] > 0), 1, 0)
    partial_usable_interval_count = np.where((left_overlap > 0) & (usable_interval_countable[left_idx] > 0), 1, 0)

    coverage_ratio = np.clip((full_duration + partial_duration) / float(window_min), 0.0, 1.0)
    accum_mm = full_mass + partial_mass
    fail_count = full_fail + partial_fail
    suspect_count = full_suspect + partial_suspect
    interval_count = full_interval_count + partial_interval_count
    usable_interval_count = full_usable_interval_count + partial_usable_interval_count
    usable_minutes = full_duration + partial_duration

    complete_strict = (coverage_ratio >= 0.95) & (fail_count == 0)
    complete_loose = coverage_ratio >= 0.80

    dominant_counts = np.zeros((len(SUPPORTED_DELTAS), n), dtype=np.int32)
    for k, delta_value in enumerate(SUPPORTED_DELTAS):
        delta_mask = (delta_int == delta_value).astype(np.int32)
        prefix_delta = np.concatenate(([0], np.cumsum(delta_mask, dtype=np.int64)))
        full_delta = prefix_delta[idx + 1] - prefix_delta[left_idx + 1]
        partial_delta = np.where((left_overlap > 0) & (delta_mask[left_idx] > 0), 1, 0)
        dominant_counts[k] = (full_delta + partial_delta).astype(np.int32)

    dominant_idx = dominant_counts.argmax(axis=0)
    dominant_max = dominant_counts.max(axis=0)
    dominant_cadence = np.full(n, np.nan, dtype=np.float64)
    has_cadence = dominant_max > 0
    dominant_cadence[has_cadence] = np.asarray(SUPPORTED_DELTAS, dtype=np.float64)[dominant_idx[has_cadence]]

    return {
        "coverage_ratio": coverage_ratio,
        "accum_mm": accum_mm,
        "fail_count": fail_count.astype(np.int32),
        "suspect_count": suspect_count.astype(np.int32),
        "interval_count": interval_count.astype(np.int32),
        "usable_interval_count": usable_interval_count.astype(np.int32),
        "usable_minutes": usable_minutes,
        "complete_strict": complete_strict,
        "complete_loose": complete_loose,
        "dominant_cadence_min": dominant_cadence,
    }


def extract_station_events(
    station_df: pd.DataFrame,
    pot_quantile: float,
    decluster_gap_short_h: float,
    decluster_gap_long_h: float,
    windows_min: tuple[int, ...],
    completeness_mode: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    station_df = station_df.sort_values("timestamp_utc").reset_index(drop=True)
    station_id = str(station_df["station_id"].iloc[0])
    n_rows = int(station_df.shape[0])

    timestamp_series = pd.to_datetime(station_df["timestamp_utc"], errors="coerce", utc=False)
    end_min = _to_minute_index(station_df["timestamp_utc"])

    duration_min = pd.to_numeric(station_df["delta_prev_min"], errors="coerce").to_numpy(dtype=np.float64)
    duration_min = np.where(np.isfinite(duration_min) & (duration_min > 0), duration_min, 0.0)
    delta_int = duration_min.astype(np.int32, copy=False)

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

    all_events: list[dict[str, object]] = []
    station_scale_summary: list[dict[str, object]] = []

    for window_min in windows_min:
        metrics = compute_window_metrics(
            end_min=end_min,
            duration_min=duration_min,
            precip_mm_qc=precip_mm_qc,
            usable=usable,
            fail_interval=fail_interval,
            suspect_interval=suspect_interval,
            delta_int=delta_int,
            window_min=window_min,
        )

        complete_strict = metrics["complete_strict"]
        complete_loose = metrics["complete_loose"]
        complete_mask = complete_strict if completeness_mode == "strict" else complete_loose
        accum_mm = metrics["accum_mm"]
        wet_mask = complete_mask & (accum_mm > 0)
        n_wet_windows = int(wet_mask.sum())

        threshold_mm = float(np.nan)
        candidate_idx = np.array([], dtype=np.int64)
        decluster_gap_min = _decluster_gap_min(window_min, decluster_gap_short_h, decluster_gap_long_h)

        if n_wet_windows > 0:
            threshold_mm = float(np.nanquantile(accum_mm[wet_mask], pot_quantile))
            candidate_idx = np.flatnonzero(wet_mask & (accum_mm >= threshold_mm))

        station_scale_summary.append(
            {
                "station_id": station_id,
                "window_min": int(window_min),
                "n_rows": n_rows,
                "n_complete_strict": int(complete_strict.sum()),
                "n_complete_loose": int(complete_loose.sum()),
                "completeness_mode_used": completeness_mode,
                "n_complete_used": int(complete_mask.sum()),
                "n_wet_windows_used": n_wet_windows,
                "threshold_mm": threshold_mm,
                "n_candidate_windows": int(candidate_idx.size),
                "decluster_gap_min": int(decluster_gap_min),
                "median_coverage_ratio": float(np.nanmedian(metrics["coverage_ratio"])),
                "p95_coverage_ratio": float(np.nanquantile(metrics["coverage_ratio"], 0.95)),
                "p99_accum_mm_strict": float(np.nanquantile(accum_mm[complete_strict], 0.99))
                if np.any(complete_strict)
                else float(np.nan),
                "max_accum_mm_strict": float(np.nanmax(accum_mm[complete_strict])) if np.any(complete_strict) else float(np.nan),
                "n_events_declustered": 0,
            }
        )

        if candidate_idx.size == 0:
            continue

        candidate_times = end_min[candidate_idx]
        cluster_break = np.ones(candidate_idx.size, dtype=bool)
        if candidate_idx.size > 1:
            cluster_break[1:] = np.diff(candidate_times) > decluster_gap_min
        cluster_ids = np.cumsum(cluster_break, dtype=np.int64) - 1

        cluster_count = int(cluster_ids[-1] + 1)
        station_scale_summary[-1]["n_events_declustered"] = cluster_count

        for cluster_id in range(cluster_count):
            cluster_mask = cluster_ids == cluster_id
            cluster_candidate_idx = candidate_idx[cluster_mask]
            cluster_values = accum_mm[cluster_candidate_idx]
            peak_local = int(np.argmax(cluster_values))
            peak_idx = int(cluster_candidate_idx[peak_local])

            cluster_start_idx = int(cluster_candidate_idx[0])
            cluster_end_idx = int(cluster_candidate_idx[-1])

            peak_ts = timestamp_series.iloc[peak_idx]
            cluster_start_ts = timestamp_series.iloc[cluster_start_idx]
            cluster_end_ts = timestamp_series.iloc[cluster_end_idx]

            all_events.append(
                {
                    "station_id": station_id,
                    "window_min": int(window_min),
                    "timestamp_utc": peak_ts,
                    "calendar_year": int(peak_ts.year),
                    "event_accum_mm": float(accum_mm[peak_idx]),
                    "event_rate_mmh": float(accum_mm[peak_idx] * 60.0 / window_min),
                    "threshold_mm": threshold_mm,
                    "threshold_quantile": float(pot_quantile),
                    "completeness_mode": completeness_mode,
                    "coverage_ratio": float(metrics["coverage_ratio"][peak_idx]),
                    "complete_strict": bool(metrics["complete_strict"][peak_idx]),
                    "complete_loose": bool(metrics["complete_loose"][peak_idx]),
                    "fail_count_window": int(metrics["fail_count"][peak_idx]),
                    "suspect_count_window": int(metrics["suspect_count"][peak_idx]),
                    "window_interval_count": int(metrics["interval_count"][peak_idx]),
                    "window_usable_interval_count": int(metrics["usable_interval_count"][peak_idx]),
                    "window_usable_minutes": float(metrics["usable_minutes"][peak_idx]),
                    "dominant_cadence_min": None
                    if np.isnan(metrics["dominant_cadence_min"][peak_idx])
                    else int(metrics["dominant_cadence_min"][peak_idx]),
                    "cluster_id_station_scale": int(cluster_id),
                    "cluster_size_windows": int(cluster_candidate_idx.size),
                    "cluster_start_utc": cluster_start_ts,
                    "cluster_end_utc": cluster_end_ts,
                    "cluster_span_min": int(end_min[cluster_end_idx] - end_min[cluster_start_idx]),
                    "window_rank_ratio": float(accum_mm[peak_idx] / threshold_mm) if threshold_mm > 0 else float("nan"),
                }
            )

    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df = events_df.sort_values(["window_min", "station_id", "timestamp_utc"]).reset_index(drop=True)
    return events_df, station_scale_summary


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.station_scale_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    windows_min = parse_windows_min(args.windows_min)

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

    writer: pq.ParquetWriter | None = None
    current_parts: list[pd.DataFrame] = []
    current_station: str | None = None
    processed_stations = 0
    station_scale_summaries: list[dict[str, object]] = []
    total_events = 0

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
                events_df, station_summary = extract_station_events(
                    station_df=station_df,
                    pot_quantile=float(args.pot_quantile),
                    decluster_gap_short_h=float(args.decluster_gap_short_h),
                    decluster_gap_long_h=float(args.decluster_gap_long_h),
                    windows_min=windows_min,
                    completeness_mode=str(args.completeness_mode),
                )
                station_scale_summaries.extend(station_summary)
                if not events_df.empty:
                    table = pa.Table.from_pandas(events_df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
                    writer.write_table(table)
                    total_events += int(events_df.shape[0])
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
        events_df, station_summary = extract_station_events(
            station_df=station_df,
            pot_quantile=float(args.pot_quantile),
            decluster_gap_short_h=float(args.decluster_gap_short_h),
            decluster_gap_long_h=float(args.decluster_gap_long_h),
            windows_min=windows_min,
            completeness_mode=str(args.completeness_mode),
        )
        station_scale_summaries.extend(station_summary)
        if not events_df.empty:
            table = pa.Table.from_pandas(events_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
            total_events += int(events_df.shape[0])
        processed_stations += 1

    if writer is not None:
        writer.close()

    if not station_scale_summaries:
        raise RuntimeError("No station summaries were generated.")

    if writer is None:
        empty_df = pd.DataFrame(
            columns=[
                "station_id",
                "window_min",
                "timestamp_utc",
                "calendar_year",
                "event_accum_mm",
                "event_rate_mmh",
                "threshold_mm",
                "threshold_quantile",
                "coverage_ratio",
                "complete_strict",
                "complete_loose",
                "fail_count_window",
                "suspect_count_window",
                "window_interval_count",
                "window_usable_interval_count",
                "window_usable_minutes",
                "dominant_cadence_min",
                "cluster_id_station_scale",
                "cluster_size_windows",
                "cluster_start_utc",
                "cluster_end_utc",
                "cluster_span_min",
                "window_rank_ratio",
            ]
        )
        empty_table = pa.Table.from_pandas(empty_df, preserve_index=False)
        pq.write_table(empty_table, args.output, compression="zstd")

    station_summary_df = pd.DataFrame(station_scale_summaries).sort_values(["window_min", "station_id"]).reset_index(drop=True)
    station_summary_df.to_csv(args.station_scale_summary_csv, index=False)

    summary: dict[str, object] = {
        "input": str(args.input),
        "output": str(args.output),
        "config": {
            "pot_quantile": float(args.pot_quantile),
            "windows_min": list(windows_min),
            "completeness_mode": str(args.completeness_mode),
            "decluster_gap_short_h": float(args.decluster_gap_short_h),
            "decluster_gap_long_h": float(args.decluster_gap_long_h),
        },
        "station_count": int(station_summary_df["station_id"].nunique()),
        "total_events": int(total_events),
        "scales": {},
    }

    for window_min in windows_min:
        sub = station_summary_df[station_summary_df["window_min"] == window_min].copy()
        if sub.empty:
            continue
        active_station_mask = sub["n_wet_windows_used"] > 0
        eventful_station_mask = sub["n_events_declustered"] > 0
        years_active = np.maximum(
            np.ceil(sub["n_complete_strict"].to_numpy(dtype=np.float64) / (365.25 * 24.0 * 60.0 / window_min)),
            1.0,
        )
        events_per_station_year = np.divide(
            sub["n_events_declustered"].to_numpy(dtype=np.float64),
            years_active,
            out=np.zeros(sub.shape[0], dtype=np.float64),
            where=years_active > 0,
        )
        summary["scales"][str(window_min)] = {
            "n_station_scale_rows": int(sub.shape[0]),
            "n_active_stations": int(active_station_mask.sum()),
            "n_eventful_stations": int(eventful_station_mask.sum()),
            "n_events_declustered": int(sub["n_events_declustered"].sum()),
            "median_threshold_mm_active": float(sub.loc[active_station_mask, "threshold_mm"].median())
            if active_station_mask.any()
            else None,
            "median_events_per_station_active": float(sub.loc[active_station_mask, "n_events_declustered"].median())
            if active_station_mask.any()
            else None,
            "median_events_per_station_year_active": float(np.median(events_per_station_year[active_station_mask.to_numpy()]))
            if active_station_mask.any()
            else None,
            "median_n_candidate_windows_active": float(sub.loc[active_station_mask, "n_candidate_windows"].median())
            if active_station_mask.any()
            else None,
        }

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.station_scale_summary_csv}")
    print(f"Wrote: {args.summary_json}")
    print(f"Stations processed: {processed_stations}")
    print(f"Total events: {total_events}")


if __name__ == "__main__":
    main()

