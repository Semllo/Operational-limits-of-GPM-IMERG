from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EARTH_RADIUS_KM = 6371.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an audit sheet for the top AVAMET extreme events per scale."
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_supported.parquet"),
        help="Input event catalog parquet with neighbor-support fields.",
    )
    parser.add_argument(
        "--qc-input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="QC parquet used to recompute neighbor-window accumulations.",
    )
    parser.add_argument(
        "--station-inventory-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv.csv"),
        help="Station inventory CSV with lat/lon.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_cv_top10_event_audit.csv"),
        help="Output CSV with one row per top event and neighbor.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/avamet_cv_top10_event_audit.md"),
        help="Output Markdown audit sheet.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of events to retain per scale.",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=3,
        help="Number of nearest neighbors to include per event.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200000,
        help="Row batch size for parquet streaming.",
    )
    return parser


def to_minute_index(values: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    ts = pd.to_datetime(values, errors="coerce", utc=False)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def load_top_events(events_path: Path, top_n: int) -> pd.DataFrame:
    events = pd.read_parquet(events_path).copy()
    events["station_id"] = events["station_id"].astype(str)
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], errors="coerce", utc=False)
    events["cluster_start_utc"] = pd.to_datetime(events["cluster_start_utc"], errors="coerce", utc=False)
    events["cluster_end_utc"] = pd.to_datetime(events["cluster_end_utc"], errors="coerce", utc=False)

    events = events.sort_values(
        ["window_min", "event_accum_mm", "timestamp_utc", "station_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for window_min, group in events.groupby("window_min", sort=True):
        top = group.head(top_n).copy()
        top["rank_in_scale"] = np.arange(1, top.shape[0] + 1, dtype=np.int32)
        parts.append(top)

    if not parts:
        return pd.DataFrame()

    top_events = pd.concat(parts, ignore_index=True)
    top_events["event_end_min"] = to_minute_index(top_events["timestamp_utc"])
    top_events["event_start_utc"] = top_events["timestamp_utc"] - pd.to_timedelta(
        top_events["window_min"].astype(int), unit="m"
    )
    return top_events


def build_neighbor_lookup(
    stations_df: pd.DataFrame, source_station_ids: list[str], neighbor_count: int
) -> dict[str, list[tuple[str, float]]]:
    stations = stations_df.copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.sort_values("station_id").reset_index(drop=True)

    ids = stations["station_id"].to_numpy(dtype=str)
    lat_rad = np.deg2rad(stations["lat"].to_numpy(dtype=np.float64))
    lon_rad = np.deg2rad(stations["lon"].to_numpy(dtype=np.float64))
    id_to_idx = {station_id: idx for idx, station_id in enumerate(ids)}

    neighbor_lookup: dict[str, list[tuple[str, float]]] = {}
    for station_id in source_station_ids:
        idx = id_to_idx.get(station_id)
        if idx is None:
            neighbor_lookup[station_id] = []
            continue

        dlat = lat_rad - lat_rad[idx]
        dlon = lon_rad - lon_rad[idx]
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_rad[idx]) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_km = EARTH_RADIUS_KM * c
        dist_km[idx] = np.inf
        order = np.argsort(dist_km)

        neighbors: list[tuple[str, float]] = []
        for neighbor_idx in order:
            if not np.isfinite(dist_km[neighbor_idx]):
                continue
            neighbors.append((ids[neighbor_idx], float(dist_km[neighbor_idx])))
            if len(neighbors) >= neighbor_count:
                break
        neighbor_lookup[station_id] = neighbors

    return neighbor_lookup


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    ordered = sorted(windows)
    merged: list[list[int]] = [[int(ordered[0][0]), int(ordered[0][1])]]
    for start_min, end_min in ordered[1:]:
        if int(start_min) <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], int(end_min))
        else:
            merged.append([int(start_min), int(end_min)])
    return [(start_min, end_min) for start_min, end_min in merged]


def build_station_requests(
    top_events: pd.DataFrame, neighbor_lookup: dict[str, list[tuple[str, float]]]
) -> dict[str, list[tuple[int, int]]]:
    requests: dict[str, list[tuple[int, int]]] = {}

    for row in top_events.itertuples(index=False):
        window_start = int(row.event_end_min - row.window_min)
        window_end = int(row.event_end_min)
        for neighbor_id, _distance_km in neighbor_lookup.get(str(row.station_id), []):
            requests.setdefault(neighbor_id, []).append((window_start, window_end))

    return {station_id: merge_windows(windows) for station_id, windows in requests.items()}


def load_station_series(
    qc_path: Path, station_requests: dict[str, list[tuple[int, int]]], batch_size: int
) -> dict[str, dict[str, np.ndarray]]:
    parquet = pq.ParquetFile(qc_path)
    columns = ["station_id", "timestamp_utc", "delta_prev_min", "precip_mm_qc", "usable_interval_extremes"]
    parts: dict[str, list[pd.DataFrame]] = {}
    target_station_ids = set(station_requests)

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        batch_df["station_id"] = batch_df["station_id"].astype(str)
        batch_df = batch_df.loc[batch_df["station_id"].isin(target_station_ids)].copy()
        if batch_df.empty:
            continue

        batch_df["end_min"] = to_minute_index(batch_df["timestamp_utc"])

        for station_id, group in batch_df.groupby("station_id", sort=False):
            windows = station_requests.get(station_id)
            if not windows:
                continue

            end_min = group["end_min"].to_numpy(dtype=np.int64, copy=False)
            keep_mask = np.zeros(group.shape[0], dtype=bool)
            for start_min, end_min_limit in windows:
                keep_mask |= (end_min > int(start_min)) & (end_min <= int(end_min_limit))

            if not keep_mask.any():
                continue

            trimmed = group.loc[
                keep_mask,
                ["end_min", "delta_prev_min", "precip_mm_qc", "usable_interval_extremes"],
            ].copy()
            parts.setdefault(station_id, []).append(trimmed)

    station_series: dict[str, dict[str, np.ndarray]] = {}
    for station_id, station_parts in parts.items():
        station_df = pd.concat(station_parts, ignore_index=True)
        station_df = station_df.sort_values("end_min").drop_duplicates(
            subset=["end_min", "delta_prev_min", "precip_mm_qc", "usable_interval_extremes"]
        )
        duration_min = pd.to_numeric(station_df["delta_prev_min"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        duration_min = np.where(np.isfinite(duration_min) & (duration_min > 0), duration_min, 0.0)
        precip_mm_qc = pd.to_numeric(station_df["precip_mm_qc"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        usable = station_df["usable_interval_extremes"].fillna(False).to_numpy(dtype=bool, copy=False)

        station_series[station_id] = {
            "end_min": station_df["end_min"].to_numpy(dtype=np.int64, copy=False),
            "duration_min": duration_min,
            "precip_mm_qc": precip_mm_qc,
            "usable": usable,
        }

    return station_series


def compute_window_stats(
    station_data: dict[str, np.ndarray] | None, window_start: int, window_end: int
) -> tuple[float, float, bool, int]:
    if station_data is None:
        return float("nan"), float("nan"), False, 0

    end_min = station_data["end_min"]
    duration_min = station_data["duration_min"]
    precip_mm_qc = station_data["precip_mm_qc"]
    usable = station_data["usable"]

    if end_min.size == 0:
        return 0.0, 0.0, False, 0

    start_min = end_min - duration_min
    overlap_min = np.minimum(end_min, float(window_end)) - np.maximum(start_min, float(window_start))
    overlap_min = np.clip(overlap_min, 0.0, None)
    contributes = overlap_min > 0
    if not contributes.any():
        return 0.0, 0.0, False, 0

    overlap_usable = usable & contributes
    fraction = np.zeros_like(duration_min, dtype=np.float64)
    valid_duration = overlap_usable & (duration_min > 0)
    fraction[valid_duration] = overlap_min[valid_duration] / duration_min[valid_duration]

    precip_safe = np.where(np.isfinite(precip_mm_qc), precip_mm_qc, 0.0)
    accum_mm = float(np.sum(precip_safe * fraction, dtype=np.float64))
    coverage_ratio = float(
        np.clip(np.sum(np.where(overlap_usable, overlap_min, 0.0), dtype=np.float64) / (window_end - window_start), 0.0, 1.0)
    )
    positive_signal = bool(np.any(overlap_usable & (precip_safe > 0)))
    contributing_intervals = int(np.count_nonzero(overlap_usable))

    return accum_mm, coverage_ratio, positive_signal, contributing_intervals


def build_audit_rows(
    top_events: pd.DataFrame,
    neighbor_lookup: dict[str, list[tuple[str, float]]],
    station_series: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in top_events.itertuples(index=False):
        station_id = str(row.station_id)
        window_start = int(row.event_end_min - row.window_min)
        window_end = int(row.event_end_min)
        neighbors = neighbor_lookup.get(station_id, [])

        for neighbor_rank, (neighbor_id, distance_km) in enumerate(neighbors, start=1):
            neighbor_accum_mm, neighbor_coverage_ratio, neighbor_positive_signal, neighbor_intervals = (
                compute_window_stats(station_series.get(neighbor_id), window_start, window_end)
            )
            rows.append(
                {
                    "window_min": int(row.window_min),
                    "rank_in_scale": int(row.rank_in_scale),
                    "station_id": station_id,
                    "timestamp_utc": row.timestamp_utc,
                    "event_start_utc": row.event_start_utc,
                    "event_accum_mm": float(row.event_accum_mm),
                    "event_rate_mmh": float(row.event_rate_mmh),
                    "threshold_mm": float(row.threshold_mm),
                    "threshold_quantile": float(row.threshold_quantile),
                    "coverage_ratio": float(row.coverage_ratio),
                    "complete_strict": bool(row.complete_strict),
                    "complete_loose": bool(row.complete_loose),
                    "fail_count_window": int(row.fail_count_window),
                    "suspect_count_window": int(row.suspect_count_window),
                    "dominant_cadence_min": (
                        float("nan")
                        if pd.isna(row.dominant_cadence_min)
                        else float(row.dominant_cadence_min)
                    ),
                    "cluster_size_windows": int(row.cluster_size_windows),
                    "cluster_start_utc": row.cluster_start_utc,
                    "cluster_end_utc": row.cluster_end_utc,
                    "cluster_span_min": int(row.cluster_span_min),
                    "window_rank_ratio": float(row.window_rank_ratio),
                    "neighbor_support": bool(row.neighbor_support),
                    "neighbor_supporting_count": int(row.neighbor_supporting_count),
                    "neighbor_candidate_count": int(row.neighbor_candidate_count),
                    "nearest_support_distance_km": (
                        float("nan")
                        if pd.isna(row.nearest_support_distance_km)
                        else float(row.nearest_support_distance_km)
                    ),
                    "neighbor_rank": int(neighbor_rank),
                    "neighbor_station_id": str(neighbor_id),
                    "neighbor_distance_km": float(distance_km),
                    "neighbor_accum_mm": float(neighbor_accum_mm),
                    "neighbor_rate_mmh": float(neighbor_accum_mm * 60.0 / row.window_min),
                    "neighbor_coverage_ratio": float(neighbor_coverage_ratio),
                    "neighbor_positive_signal": bool(neighbor_positive_signal),
                    "neighbor_contributing_intervals": int(neighbor_intervals),
                }
            )

    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["window_min", "rank_in_scale", "neighbor_rank"]).reset_index(drop=True)
    return audit_df


def write_markdown(audit_df: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = ["# AVAMET Top Event Audit", ""]

    for window_min, scale_df in audit_df.groupby("window_min", sort=True):
        lines.append(f"## {int(window_min)} min")
        lines.append("")

        for rank_in_scale, event_df in scale_df.groupby("rank_in_scale", sort=True):
            head = event_df.iloc[0]
            lines.append(
                (
                    f"### Rank {int(rank_in_scale)} | station `{head['station_id']}` | "
                    f"{pd.Timestamp(head['timestamp_utc']).isoformat()} | "
                    f"{head['event_accum_mm']:.1f} mm ({head['event_rate_mmh']:.1f} mm/h)"
                )
            )
            lines.append(
                (
                    f"- Window: {pd.Timestamp(head['event_start_utc']).isoformat()} -> "
                    f"{pd.Timestamp(head['timestamp_utc']).isoformat()} | "
                    f"coverage={head['coverage_ratio']:.3f} | "
                    f"fail={int(head['fail_count_window'])} | suspect={int(head['suspect_count_window'])}"
                )
            )
            lines.append(
                (
                    f"- Cluster: size={int(head['cluster_size_windows'])} windows | "
                    f"span={int(head['cluster_span_min'])} min | "
                    f"support={bool(head['neighbor_support'])} "
                    f"({int(head['neighbor_supporting_count'])}/{int(head['neighbor_candidate_count'])})"
                )
            )
            lines.append("- Neighbors:")
            for neighbor_row in event_df.itertuples(index=False):
                lines.append(
                    (
                        f"  - {int(neighbor_row.neighbor_rank)}. `{neighbor_row.neighbor_station_id}` | "
                        f"{neighbor_row.neighbor_distance_km:.2f} km | "
                        f"{neighbor_row.neighbor_accum_mm:.1f} mm ({neighbor_row.neighbor_rate_mmh:.1f} mm/h) | "
                        f"coverage={neighbor_row.neighbor_coverage_ratio:.3f} | "
                        f"positive={bool(neighbor_row.neighbor_positive_signal)}"
                    )
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    top_events = load_top_events(args.events_input, max(1, int(args.top_n)))
    if top_events.empty:
        raise RuntimeError(f"No events found in {args.events_input}")

    stations = pd.read_csv(args.station_inventory_csv)
    source_station_ids = sorted(top_events["station_id"].astype(str).unique().tolist())
    neighbor_lookup = build_neighbor_lookup(stations, source_station_ids, max(1, int(args.neighbor_count)))
    station_requests = build_station_requests(top_events, neighbor_lookup)
    station_series = load_station_series(args.qc_input, station_requests, int(args.batch_size))
    audit_df = build_audit_rows(top_events, neighbor_lookup, station_series)
    if audit_df.empty:
        raise RuntimeError("The audit table is empty.")

    audit_df.to_csv(args.output_csv, index=False)
    write_markdown(audit_df, args.output_md)

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    print(f"Audited events: {top_events.shape[0]}")
    print(f"Audit rows: {audit_df.shape[0]}")


if __name__ == "__main__":
    main()

