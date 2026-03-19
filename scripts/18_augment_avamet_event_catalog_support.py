from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EARTH_RADIUS_KM = 6371.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment the AVAMET event catalog with neighbor support and robustness summaries."
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events.parquet"),
        help="Input event catalog parquet.",
    )
    parser.add_argument(
        "--qc-input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="QC parquet used to derive neighbor support from positive precipitation intervals.",
    )
    parser.add_argument(
        "--station-inventory-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv.csv"),
        help="Station inventory CSV with lat/lon.",
    )
    parser.add_argument(
        "--station-scale-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_events_station_scale_summary.csv"),
        help="Per-station, per-scale event summary CSV produced by 17_build_avamet_event_catalog.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/avamet_cv_events_supported.parquet"),
        help="Output augmented event catalog parquet.",
    )
    parser.add_argument(
        "--scale-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_events_support_scale_summary.csv"),
        help="Output scale-level robustness summary CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/avamet_cv_events_support_summary.json"),
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--neighbor-radius-km",
        type=float,
        default=15.0,
        help="Neighbor search radius in kilometers.",
    )
    parser.add_argument(
        "--short-tolerance-min",
        type=int,
        default=30,
        help="Temporal tolerance (minutes) for 30/60 min windows.",
    )
    parser.add_argument(
        "--long-tolerance-min",
        type=int,
        default=60,
        help="Temporal tolerance (minutes) for 180/360 min windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200000,
        help="Row batch size for streaming the QC parquet.",
    )
    return parser


def to_minute_index(ts: pd.Series) -> np.ndarray:
    ts_parsed = pd.to_datetime(ts, errors="coerce", utc=False)
    ts_ns = ts_parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def build_neighbor_map(stations_df: pd.DataFrame, radius_km: float) -> dict[str, list[tuple[str, float]]]:
    stations = stations_df.sort_values("station_id").reset_index(drop=True)
    lat_rad = np.deg2rad(stations["lat"].to_numpy(dtype=np.float64))
    lon_rad = np.deg2rad(stations["lon"].to_numpy(dtype=np.float64))

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist_km = EARTH_RADIUS_KM * c

    ids = stations["station_id"].astype(str).tolist()
    neighbor_map: dict[str, list[tuple[str, float]]] = {}
    for i, station_id in enumerate(ids):
        mask = (dist_km[i] <= radius_km) & (dist_km[i] > 0.0)
        neighbor_idx = np.where(mask)[0]
        ordered = neighbor_idx[np.argsort(dist_km[i, neighbor_idx])]
        neighbor_map[station_id] = [(ids[j], float(dist_km[i, j])) for j in ordered]
    return neighbor_map


def load_positive_interval_times(qc_path: Path, batch_size: int) -> dict[str, np.ndarray]:
    parquet = pq.ParquetFile(qc_path)
    positive_times: dict[str, list[np.ndarray]] = {}

    for batch in parquet.iter_batches(batch_size=batch_size, columns=["station_id", "timestamp_utc", "precip_mm_qc"]):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        batch_df = batch_df.loc[batch_df["precip_mm_qc"].notna() & (batch_df["precip_mm_qc"] > 0)].copy()
        if batch_df.empty:
            continue

        batch_df["time_min"] = to_minute_index(batch_df["timestamp_utc"])

        for station_id, group in batch_df.groupby("station_id", sort=False):
            station_id = str(station_id)
            arr = group["time_min"].to_numpy(dtype=np.int64, copy=True)
            positive_times.setdefault(station_id, []).append(arr)

    merged: dict[str, np.ndarray] = {}
    for station_id, parts in positive_times.items():
        if len(parts) == 1:
            arr = parts[0]
        else:
            arr = np.concatenate(parts)
        arr.sort()
        merged[station_id] = arr
    return merged


def tolerance_for_window(window_min: int, short_tolerance_min: int, long_tolerance_min: int) -> int:
    if int(window_min) <= 60:
        return int(short_tolerance_min)
    return int(long_tolerance_min)


def augment_events(
    events_df: pd.DataFrame,
    neighbor_map: dict[str, list[tuple[str, float]]],
    positive_times: dict[str, np.ndarray],
    short_tolerance_min: int,
    long_tolerance_min: int,
    neighbor_radius_km: float,
) -> pd.DataFrame:
    out = events_df.copy()
    if "event_rate_mmh" not in out.columns:
        out["event_rate_mmh"] = out["event_accum_mm"].astype(float) * 60.0 / out["window_min"].astype(float)

    event_time_min = to_minute_index(out["timestamp_utc"])

    neighbor_support = np.zeros(out.shape[0], dtype=bool)
    neighbor_supporting_count = np.zeros(out.shape[0], dtype=np.int32)
    neighbor_candidate_count = np.zeros(out.shape[0], dtype=np.int32)
    neighbor_support_tolerance_min = np.zeros(out.shape[0], dtype=np.int32)
    nearest_support_distance_km = np.full(out.shape[0], np.nan, dtype=np.float64)

    for idx, row in enumerate(out.itertuples(index=False)):
        station_id = str(row.station_id)
        window_min = int(row.window_min)
        tol_min = tolerance_for_window(window_min, short_tolerance_min, long_tolerance_min)
        neighbor_support_tolerance_min[idx] = tol_min

        neighbors = neighbor_map.get(station_id, [])
        neighbor_candidate_count[idx] = len(neighbors)
        if not neighbors:
            continue

        search_start = int(event_time_min[idx] - window_min - tol_min)
        search_end = int(event_time_min[idx] + tol_min)

        support_count = 0
        support_distances: list[float] = []
        for neighbor_id, distance_km in neighbors:
            arr = positive_times.get(neighbor_id)
            if arr is None or arr.size == 0:
                continue
            pos = np.searchsorted(arr, search_start, side="left")
            if pos < arr.size and arr[pos] <= search_end:
                support_count += 1
                support_distances.append(distance_km)

        if support_count > 0:
            neighbor_support[idx] = True
            neighbor_supporting_count[idx] = support_count
            nearest_support_distance_km[idx] = min(support_distances)

    out["neighbor_radius_km"] = float(neighbor_radius_km)
    out["neighbor_support_tolerance_min"] = neighbor_support_tolerance_min
    out["neighbor_candidate_count"] = neighbor_candidate_count
    out["neighbor_supporting_count"] = neighbor_supporting_count
    out["neighbor_support"] = neighbor_support
    out["nearest_support_distance_km"] = nearest_support_distance_km
    return out


def build_scale_summary(
    augmented_events_df: pd.DataFrame,
    station_scale_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window_min in sorted(station_scale_summary_df["window_min"].unique()):
        station_sub = station_scale_summary_df.loc[station_scale_summary_df["window_min"] == window_min].copy()
        events_sub = augmented_events_df.loc[augmented_events_df["window_min"] == window_min].copy()

        event_counts = station_sub["n_events_declustered"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "window_min": int(window_min),
                "n_station_scale_rows": int(station_sub.shape[0]),
                "n_events": int(events_sub.shape[0]),
                "p10_events_per_station": float(np.quantile(event_counts, 0.10)),
                "p50_events_per_station": float(np.quantile(event_counts, 0.50)),
                "p90_events_per_station": float(np.quantile(event_counts, 0.90)),
                "pct_events_with_suspect_gt0": float(100.0 * (events_sub["suspect_count_window"] > 0).mean())
                if not events_sub.empty
                else float("nan"),
                "pct_events_with_neighbor_support": float(100.0 * events_sub["neighbor_support"].mean())
                if not events_sub.empty
                else float("nan"),
                "pct_events_with_neighbor_support_and_no_suspect": float(
                    100.0 * (events_sub["neighbor_support"] & (events_sub["suspect_count_window"] == 0)).mean()
                )
                if not events_sub.empty
                else float("nan"),
                "median_neighbor_supporting_count": float(events_sub["neighbor_supporting_count"].median())
                if not events_sub.empty
                else float("nan"),
                "median_event_rate_mmh": float(events_sub["event_rate_mmh"].median()) if not events_sub.empty else float("nan"),
                "p90_event_rate_mmh": float(events_sub["event_rate_mmh"].quantile(0.90)) if not events_sub.empty else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("window_min").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scale_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    events_df = pd.read_parquet(args.events_input)
    station_inventory_df = pd.read_csv(args.station_inventory_csv)
    station_scale_summary_df = pd.read_csv(args.station_scale_summary_csv)

    neighbor_map = build_neighbor_map(station_inventory_df, radius_km=float(args.neighbor_radius_km))
    positive_times = load_positive_interval_times(args.qc_input, batch_size=int(args.batch_size))

    augmented_events_df = augment_events(
        events_df=events_df,
        neighbor_map=neighbor_map,
        positive_times=positive_times,
        short_tolerance_min=int(args.short_tolerance_min),
        long_tolerance_min=int(args.long_tolerance_min),
        neighbor_radius_km=float(args.neighbor_radius_km),
    )
    augmented_events_df.to_parquet(args.output, index=False)

    scale_summary_df = build_scale_summary(augmented_events_df, station_scale_summary_df)
    scale_summary_df.to_csv(args.scale_summary_csv, index=False)

    summary = {
        "events_input": str(args.events_input),
        "output": str(args.output),
        "neighbor_radius_km": float(args.neighbor_radius_km),
        "short_tolerance_min": int(args.short_tolerance_min),
        "long_tolerance_min": int(args.long_tolerance_min),
        "n_events": int(augmented_events_df.shape[0]),
        "pct_events_with_neighbor_support": float(100.0 * augmented_events_df["neighbor_support"].mean())
        if not augmented_events_df.empty
        else None,
        "pct_events_with_suspect_gt0": float(100.0 * (augmented_events_df["suspect_count_window"] > 0).mean())
        if not augmented_events_df.empty
        else None,
        "scales": scale_summary_df.to_dict(orient="records"),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.scale_summary_csv}")
    print(f"Wrote: {args.summary_json}")
    print(f"Events processed: {augmented_events_df.shape[0]}")


if __name__ == "__main__":
    main()

