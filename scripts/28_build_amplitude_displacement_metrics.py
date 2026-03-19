from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0
FILE_TIME_RE = re.compile(r"(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an amplitude-vs-displacement decomposition for AVAMET extreme events using "
            "the IMERG 3x3 neighborhood around each station-mapped cell."
        )
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common.parquet"),
        help="Input event parquet restricted to the common AVAMET-IMERG period.",
    )
    parser.add_argument(
        "--station-alignment-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_imerg.csv"),
        help="Station-to-IMERG alignment CSV.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("results/imerg_cv_subset_metadata.json"),
        help="IMERG subset metadata JSON.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("data/imerg_cv"),
        help="Root directory with IMERG CV subset files.",
    )
    parser.add_argument(
        "--candidate-days-csv",
        type=Path,
        default=Path("results/dana_candidate_days.csv"),
        help="Optional CSV with candidate DANA days for case-study summaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement.parquet"),
        help="Output enriched event parquet.",
    )
    parser.add_argument(
        "--output-by-scale-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement_by_scale.csv"),
        help="Output summary CSV by scale.",
    )
    parser.add_argument(
        "--output-case-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement_case_day_window.csv"),
        help="Output summary CSV by selected case-study day and scale.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement_summary.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=256,
        help="Maximum number of IMERG subset files kept in the LRU cache.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional event limit for testing.",
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
    index_df["start_min"] = to_minute_index(index_df["start_ts"])
    index_df["end_min"] = to_minute_index(index_df["end_ts"])
    return index_df


def candidate_subset_slice(start_min_values: np.ndarray, event_start_min: int, event_end_min: int) -> slice:
    left = int(np.searchsorted(start_min_values, event_start_min - 30, side="right"))
    right = int(np.searchsorted(start_min_values, event_end_min, side="left"))
    return slice(left, right)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    denominator_f = float(denominator)
    if denominator_f == 0.0:
        return None
    return float(numerator) / denominator_f


class NPZCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(1, int(max_items))
        self._store: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        cached = self._store.get(path)
        if cached is not None:
            self._store.move_to_end(path)
            return cached

        with np.load(path) as handle:
            data = {
                "precipitation": handle["precipitation"].astype(np.float32, copy=False),
                "lon": handle["lon"].astype(np.float32, copy=False),
                "lat": handle["lat"].astype(np.float32, copy=False),
            }

        self._store[path] = data
        self._store.move_to_end(path)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return data


def load_station_meta(alignment_csv: Path, metadata_json: Path) -> dict[str, dict[str, float | int | bool]]:
    alignment = pd.read_csv(alignment_csv)
    alignment["station_id"] = alignment["station_id"].astype(str)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    lon_start = int(metadata["lon_index_range"]["start"])
    lat_start = int(metadata["lat_index_range"]["start"])

    station_meta: dict[str, dict[str, float | int | bool]] = {}
    for row in alignment.itertuples(index=False):
        station_id = str(row.station_id)
        station_meta[station_id] = {
            "lon_idx_local": int(row.imerg_cv_lon_idx) - lon_start,
            "lat_idx_local": int(row.imerg_cv_lat_idx) - lat_start,
            "lon_idx_global": int(row.imerg_cv_lon_idx),
            "lat_idx_global": int(row.imerg_cv_lat_idx),
            "lon": float(row.imerg_cv_lon),
            "lat": float(row.imerg_cv_lat),
            "used_fallback": not bool(row.imerg_cell_in_cv_polygon),
        }
    return station_meta


def enrich_events(
    events_df: pd.DataFrame,
    subset_index: pd.DataFrame,
    station_meta: dict[str, dict[str, float | int | bool]],
    cache_size: int,
) -> pd.DataFrame:
    out = events_df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    out = out.sort_values(["timestamp_utc", "window_min", "station_id"]).reset_index(drop=True)
    event_end_min = to_minute_index(out["timestamp_utc"])
    event_start_min = event_end_min - out["window_min"].astype(int).to_numpy(dtype=np.int64, copy=False)

    start_min_values = subset_index["start_min"].to_numpy(dtype=np.int64, copy=False)
    end_min_values = subset_index["end_min"].to_numpy(dtype=np.int64, copy=False)
    path_values = subset_index["path"].tolist()
    cache = NPZCache(cache_size)

    argmax_lon_local = np.full(out.shape[0], np.nan, dtype=np.float64)
    argmax_lat_local = np.full(out.shape[0], np.nan, dtype=np.float64)
    argmax_lon = np.full(out.shape[0], np.nan, dtype=np.float64)
    argmax_lat = np.full(out.shape[0], np.nan, dtype=np.float64)
    argmax_distance_km = np.full(out.shape[0], np.nan, dtype=np.float64)
    delta_lon_cells = np.full(out.shape[0], np.nan, dtype=np.float64)
    delta_lat_cells = np.full(out.shape[0], np.nan, dtype=np.float64)
    best_same_cell = np.zeros(out.shape[0], dtype=bool)
    valid_neighbor_cell_count = np.zeros(out.shape[0], dtype=np.int32)

    for idx, row in enumerate(out.itertuples(index=False)):
        station_id = str(row.station_id)
        meta = station_meta.get(station_id)
        if meta is None:
            continue

        lon_idx_local = int(meta["lon_idx_local"])
        lat_idx_local = int(meta["lat_idx_local"])
        event_start = int(event_start_min[idx])
        event_end = int(event_end_min[idx])

        candidate_slice = candidate_subset_slice(start_min_values, event_start, event_end)
        if candidate_slice.start == candidate_slice.stop:
            continue

        accum_grid: np.ndarray | None = None
        valid_grid: np.ndarray | None = None
        lon_coords: np.ndarray | None = None
        lat_coords: np.ndarray | None = None
        lon_offset = max(0, lon_idx_local - 1)
        lat_offset = max(0, lat_idx_local - 1)

        for pos in range(candidate_slice.start, candidate_slice.stop):
            file_start = int(start_min_values[pos])
            file_end = int(end_min_values[pos])
            overlap_min = min(file_end, event_end) - max(file_start, event_start)
            if overlap_min <= 0:
                continue

            subset = cache.get(path_values[pos])
            precip = subset["precipitation"]
            lon_lo = max(0, lon_idx_local - 1)
            lon_hi = min(precip.shape[0], lon_idx_local + 2)
            lat_lo = max(0, lat_idx_local - 1)
            lat_hi = min(precip.shape[1], lat_idx_local + 2)
            neighborhood = precip[lon_lo:lon_hi, lat_lo:lat_hi]

            if accum_grid is None:
                accum_grid = np.zeros_like(neighborhood, dtype=np.float64)
                valid_grid = np.zeros_like(neighborhood, dtype=bool)
                lon_coords = subset["lon"][lon_lo:lon_hi].astype(np.float64, copy=False)
                lat_coords = subset["lat"][lat_lo:lat_hi].astype(np.float64, copy=False)
                lon_offset = lon_lo
                lat_offset = lat_lo

            factor = float(overlap_min) / 60.0
            finite = np.isfinite(neighborhood)
            if finite.any():
                accum_grid[finite] += neighborhood[finite].astype(np.float64) * factor
                valid_grid[finite] = True

        if accum_grid is None or valid_grid is None or not valid_grid.any():
            continue

        masked = np.where(valid_grid, accum_grid, np.nan)
        max_value = float(np.nanmax(masked))
        center_i = lon_idx_local - lon_offset
        center_j = lat_idx_local - lat_offset

        if not np.isfinite(max_value) or max_value <= 0.0:
            best_lon_idx_local = lon_idx_local
            best_lat_idx_local = lat_idx_local
            local_i = center_i
            local_j = center_j
        else:
            candidate_local = np.argwhere(np.isclose(masked, max_value, rtol=0.0, atol=1e-7))
            if candidate_local.shape[0] == 0:
                candidate_local = np.argwhere(np.isfinite(masked))
            cell_dist2 = (candidate_local[:, 0] - center_i) ** 2 + (candidate_local[:, 1] - center_j) ** 2
            best_choice = int(np.argmin(cell_dist2))
            local_i = int(candidate_local[best_choice, 0])
            local_j = int(candidate_local[best_choice, 1])
            best_lon_idx_local = lon_offset + local_i
            best_lat_idx_local = lat_offset + local_j

        argmax_lon_local[idx] = float(best_lon_idx_local)
        argmax_lat_local[idx] = float(best_lat_idx_local)
        if lon_coords is not None and lat_coords is not None:
            best_lon = float(lon_coords[local_i])
            best_lat = float(lat_coords[local_j])
            argmax_lon[idx] = best_lon
            argmax_lat[idx] = best_lat
            argmax_distance_km[idx] = haversine_km(float(meta["lon"]), float(meta["lat"]), best_lon, best_lat)

        delta_lon_cells[idx] = float(best_lon_idx_local - lon_idx_local)
        delta_lat_cells[idx] = float(best_lat_idx_local - lat_idx_local)
        best_same_cell[idx] = (best_lon_idx_local == lon_idx_local) and (best_lat_idx_local == lat_idx_local)
        valid_neighbor_cell_count[idx] = int(np.count_nonzero(valid_grid))

    out["imerg_best_3x3_lon_idx_local"] = argmax_lon_local
    out["imerg_best_3x3_lat_idx_local"] = argmax_lat_local
    out["imerg_best_3x3_lon"] = argmax_lon
    out["imerg_best_3x3_lat"] = argmax_lat
    out["imerg_best_3x3_distance_km"] = argmax_distance_km
    out["imerg_best_3x3_delta_lon_cells"] = delta_lon_cells
    out["imerg_best_3x3_delta_lat_cells"] = delta_lat_cells
    out["imerg_best_3x3_same_cell"] = best_same_cell
    out["imerg_best_3x3_valid_neighbor_cell_count"] = valid_neighbor_cell_count

    point = pd.to_numeric(out["imerg_event_accum_mm"], errors="coerce").astype(float)
    best = pd.to_numeric(out["imerg_3x3_event_accum_max_mm"], errors="coerce").astype(float)
    avamet = pd.to_numeric(out["event_accum_mm"], errors="coerce").astype(float)

    out["imerg_3x3_gain_mm"] = best - point
    point_nonzero = point.replace(0.0, np.nan)
    out["imerg_3x3_gain_ratio_vs_point"] = best / point_nonzero
    out["imerg_point_recovery_ratio"] = point / avamet.replace(0.0, np.nan)
    out["imerg_3x3_recovery_ratio"] = best / avamet.replace(0.0, np.nan)
    out["imerg_displacement_improves_amplitude"] = out["imerg_3x3_gain_mm"] > 0.0
    out["event_date"] = out["timestamp_utc"].dt.normalize()
    return out


def summarize_group(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "event_count": 0,
            "station_count": 0,
            "median_distance_km": None,
            "p90_distance_km": None,
            "fraction_same_cell": None,
            "fraction_displaced": None,
            "fraction_gain_positive": None,
            "median_gain_mm": None,
            "median_gain_ratio_vs_point": None,
            "median_point_recovery_ratio": None,
            "median_3x3_recovery_ratio": None,
            "sum_avamet_event_mm": None,
            "sum_imerg_point_mm": None,
            "sum_imerg_best3x3_mm": None,
        }

    distance = pd.to_numeric(df["imerg_best_3x3_distance_km"], errors="coerce").astype(float)
    gain_mm = pd.to_numeric(df["imerg_3x3_gain_mm"], errors="coerce").astype(float)
    gain_ratio = pd.to_numeric(df["imerg_3x3_gain_ratio_vs_point"], errors="coerce").astype(float)
    point_ratio = pd.to_numeric(df["imerg_point_recovery_ratio"], errors="coerce").astype(float)
    best_ratio = pd.to_numeric(df["imerg_3x3_recovery_ratio"], errors="coerce").astype(float)
    same_cell = df["imerg_best_3x3_same_cell"].fillna(False).astype(bool)
    gain_positive = df["imerg_displacement_improves_amplitude"].fillna(False).astype(bool)

    return {
        "event_count": int(df.shape[0]),
        "station_count": int(df["station_id"].astype(str).nunique()),
        "median_distance_km": float(distance.median()) if distance.notna().any() else None,
        "p90_distance_km": float(distance.quantile(0.9)) if distance.notna().any() else None,
        "fraction_same_cell": float(same_cell.mean()),
        "fraction_displaced": float((~same_cell).mean()),
        "fraction_gain_positive": float(gain_positive.mean()),
        "median_gain_mm": float(gain_mm.median()) if gain_mm.notna().any() else None,
        "median_gain_ratio_vs_point": float(gain_ratio.median()) if gain_ratio.notna().any() else None,
        "median_point_recovery_ratio": float(point_ratio.median()) if point_ratio.notna().any() else None,
        "median_3x3_recovery_ratio": float(best_ratio.median()) if best_ratio.notna().any() else None,
        "sum_avamet_event_mm": float(pd.to_numeric(df["event_accum_mm"], errors="coerce").sum()),
        "sum_imerg_point_mm": float(pd.to_numeric(df["imerg_event_accum_mm"], errors="coerce").sum()),
        "sum_imerg_best3x3_mm": float(pd.to_numeric(df["imerg_3x3_event_accum_max_mm"], errors="coerce").sum()),
    }


def build_by_scale_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"scope": "all", **summarize_group(df)})
    for window_min, group in df.groupby("window_min", sort=True):
        rows.append({"scope": str(int(window_min)), "window_min": int(window_min), **summarize_group(group)})
    summary_df = pd.DataFrame(rows)
    if "window_min" not in summary_df.columns:
        summary_df["window_min"] = np.nan
    return summary_df


def build_case_summary(df: pd.DataFrame, candidate_days_csv: Path) -> pd.DataFrame:
    if not candidate_days_csv.exists():
        return pd.DataFrame()

    days = pd.read_csv(candidate_days_csv)
    if "date" not in days.columns or days.empty:
        return pd.DataFrame()
    days["date"] = pd.to_datetime(days["date"], errors="coerce", utc=False).dt.normalize()

    rows: list[dict[str, object]] = []
    for date in days["date"].dropna().tolist():
        date_df = df.loc[df["event_date"] == date].copy()
        for window_min, group in date_df.groupby("window_min", sort=True):
            rows.append({"date": pd.Timestamp(date), "window_min": int(window_min), **summarize_group(group)})
    return pd.DataFrame(rows).sort_values(["date", "window_min"]).reset_index(drop=True) if rows else pd.DataFrame()


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_by_scale_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_case_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(args.events_input)
    if args.limit_events > 0:
        events = events.head(int(args.limit_events)).copy()
    if events.empty:
        raise RuntimeError(f"No events found in {args.events_input}")

    subset_index = build_subset_index(args.subset_root)
    station_meta = load_station_meta(args.station_alignment_csv, args.metadata_json)
    enriched = enrich_events(events, subset_index, station_meta, int(args.cache_size))

    enriched.to_parquet(args.output, index=False)

    by_scale = build_by_scale_summary(enriched)
    by_scale.to_csv(args.output_by_scale_csv, index=False)

    case_df = build_case_summary(enriched, args.candidate_days_csv)
    if not case_df.empty:
        case_df.to_csv(args.output_case_csv, index=False)
    else:
        args.output_case_csv.write_text("", encoding="utf-8")

    summary = {
        "event_count": int(enriched.shape[0]),
        "by_scale_rows": by_scale.to_dict(orient="records"),
        "candidate_case_row_count": int(case_df.shape[0]),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.output_by_scale_csv}")
    print(f"Wrote: {args.output_case_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Events processed: {enriched.shape[0]}")


if __name__ == "__main__":
    main()

