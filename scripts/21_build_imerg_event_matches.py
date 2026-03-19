from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


FILE_TIME_RE = re.compile(r"(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match AVAMET event windows against IMERG CV subsets."
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_supported.parquet"),
        help="AVAMET event catalog parquet.",
    )
    parser.add_argument(
        "--station-alignment-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_imerg.csv"),
        help="AVAMET-to-IMERG alignment CSV.",
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
        help="Root directory with IMERG .cv.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/avamet_cv_events_imerg.parquet"),
        help="Output parquet with AVAMET events enriched with IMERG metrics.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_summary.json"),
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=256,
        help="Maximum number of .npz files kept in memory.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional limit on number of events to process.",
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
    index_df["end_min"] = index_df["start_min"] + 30
    return index_df


def load_station_cells(alignment_csv: Path, metadata_json: Path) -> dict[str, dict[str, int | float | str]]:
    alignment = pd.read_csv(alignment_csv)
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    lon_start = int(metadata["lon_index_range"]["start"])
    lat_start = int(metadata["lat_index_range"]["start"])

    required = [
        "station_id",
        "imerg_cv_lon_idx",
        "imerg_cv_lat_idx",
        "imerg_cv_lon",
        "imerg_cv_lat",
        "imerg_cell_in_cv_polygon",
    ]
    missing = [column for column in required if column not in alignment.columns]
    if missing:
        raise KeyError(f"Missing required columns in {alignment_csv}: {missing}")

    station_cells: dict[str, dict[str, int | float | str]] = {}
    for row in alignment.itertuples(index=False):
        station_id = str(row.station_id)
        lon_idx_local = int(row.imerg_cv_lon_idx) - lon_start
        lat_idx_local = int(row.imerg_cv_lat_idx) - lat_start
        station_cells[station_id] = {
            "lon_idx_local": lon_idx_local,
            "lat_idx_local": lat_idx_local,
            "lon_idx_global": int(row.imerg_cv_lon_idx),
            "lat_idx_global": int(row.imerg_cv_lat_idx),
            "imerg_lon": float(row.imerg_cv_lon),
            "imerg_lat": float(row.imerg_cv_lat),
            "fallback_from_outside_polygon": not bool(row.imerg_cell_in_cv_polygon),
        }
    return station_cells


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


def expected_imerg_step_count(event_start_min: int, event_end_min: int) -> int:
    if event_end_min <= event_start_min:
        return 0
    left = math.ceil((event_start_min - 29) / 30.0)
    right = math.floor((event_end_min - 1) / 30)
    return max(0, right - left + 1)


def candidate_subset_slice(start_min_values: np.ndarray, event_start_min: int, event_end_min: int) -> slice:
    left = int(np.searchsorted(start_min_values, event_start_min - 30, side="right"))
    right = int(np.searchsorted(start_min_values, event_end_min, side="left"))
    return slice(left, right)


def process_events(
    events_df: pd.DataFrame,
    subset_index: pd.DataFrame,
    station_cells: dict[str, dict[str, int | float | str]],
    cache_size: int,
) -> pd.DataFrame:
    start_min_values = subset_index["start_min"].to_numpy(dtype=np.int64, copy=False)
    end_min_values = subset_index["end_min"].to_numpy(dtype=np.int64, copy=False)
    path_values = subset_index["path"].to_numpy(dtype=object, copy=False)
    cache = NPZCache(cache_size)

    out = events_df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    event_end_min = to_minute_index(out["timestamp_utc"])
    event_start_min = event_end_min - out["window_min"].astype(int).to_numpy(dtype=np.int64, copy=False)
    out["imerg_window_start_utc"] = out["timestamp_utc"] - pd.to_timedelta(out["window_min"].astype(int), unit="m")

    station_meta = out["station_id"].astype(str).map(station_cells)
    if station_meta.isna().any():
        missing_ids = sorted(out.loc[station_meta.isna(), "station_id"].astype(str).unique().tolist())
        raise KeyError(f"Missing IMERG mapping for stations: {missing_ids[:10]}")

    direct_accum = np.full(out.shape[0], np.nan, dtype=np.float64)
    direct_rate = np.full(out.shape[0], np.nan, dtype=np.float64)
    direct_peak_rate = np.full(out.shape[0], np.nan, dtype=np.float64)
    neighbor_accum_max = np.full(out.shape[0], np.nan, dtype=np.float64)
    neighbor_peak_rate = np.full(out.shape[0], np.nan, dtype=np.float64)
    coverage_ratio = np.zeros(out.shape[0], dtype=np.float64)
    covered_minutes = np.zeros(out.shape[0], dtype=np.float64)
    overlap_file_count = np.zeros(out.shape[0], dtype=np.int32)
    expected_file_count = np.zeros(out.shape[0], dtype=np.int32)
    positive_steps_direct = np.zeros(out.shape[0], dtype=np.int32)
    positive_steps_neighbor = np.zeros(out.shape[0], dtype=np.int32)
    missing_files = np.zeros(out.shape[0], dtype=np.int32)
    used_fallback_cell = np.zeros(out.shape[0], dtype=bool)

    for idx, meta in enumerate(station_meta.tolist()):
        station_id = str(out.iloc[idx]["station_id"])
        lon_idx_local = int(meta["lon_idx_local"])
        lat_idx_local = int(meta["lat_idx_local"])
        used_fallback_cell[idx] = bool(meta["fallback_from_outside_polygon"])

        event_start = int(event_start_min[idx])
        event_end = int(event_end_min[idx])
        window_min = int(out.iloc[idx]["window_min"])
        expected_steps = expected_imerg_step_count(event_start, event_end)
        expected_file_count[idx] = expected_steps

        candidate_slice = candidate_subset_slice(start_min_values, event_start, event_end)
        if candidate_slice.start == candidate_slice.stop:
            missing_files[idx] = expected_steps
            continue

        neighbor_accum_grid: np.ndarray | None = None
        neighbor_valid_grid: np.ndarray | None = None
        direct_total = 0.0
        peak_direct = np.nan
        peak_neighbor = np.nan
        positive_direct = 0
        positive_neighbor = 0
        overlap_count = 0
        covered = 0.0

        for pos in range(candidate_slice.start, candidate_slice.stop):
            file_start = int(start_min_values[pos])
            file_end = int(end_min_values[pos])
            overlap_min = min(file_end, event_end) - max(file_start, event_start)
            if overlap_min <= 0:
                continue

            overlap_count += 1
            covered += float(overlap_min)

            subset = cache.get(str(path_values[pos]))
            precip = subset["precipitation"]
            if (
                lon_idx_local < 0
                or lat_idx_local < 0
                or lon_idx_local >= precip.shape[0]
                or lat_idx_local >= precip.shape[1]
            ):
                raise IndexError(
                    f"Local IMERG cell for station {station_id} falls outside subset bounds: "
                    f"({lon_idx_local}, {lat_idx_local}) vs {precip.shape}"
                )

            factor = float(overlap_min) / 60.0

            cell_rate = float(precip[lon_idx_local, lat_idx_local])
            if np.isfinite(cell_rate):
                direct_total += cell_rate * factor
                peak_direct = cell_rate if np.isnan(peak_direct) else max(peak_direct, cell_rate)
                if cell_rate > 0.0:
                    positive_direct += 1

            lon_lo = max(0, lon_idx_local - 1)
            lon_hi = min(precip.shape[0], lon_idx_local + 2)
            lat_lo = max(0, lat_idx_local - 1)
            lat_hi = min(precip.shape[1], lat_idx_local + 2)
            neighborhood = precip[lon_lo:lon_hi, lat_lo:lat_hi]
            if neighbor_accum_grid is None:
                neighbor_accum_grid = np.zeros_like(neighborhood, dtype=np.float64)
                neighbor_valid_grid = np.zeros_like(neighborhood, dtype=bool)

            finite_mask = np.isfinite(neighborhood)
            if finite_mask.any():
                neighbor_accum_grid[finite_mask] += neighborhood[finite_mask].astype(np.float64) * factor
                neighbor_valid_grid[finite_mask] = True
                neighborhood_peak = float(np.nanmax(neighborhood))
                peak_neighbor = neighborhood_peak if np.isnan(peak_neighbor) else max(peak_neighbor, neighborhood_peak)
                if np.any(neighborhood[finite_mask] > 0.0):
                    positive_neighbor += 1

        overlap_file_count[idx] = overlap_count
        covered_minutes[idx] = covered
        coverage_ratio[idx] = min(1.0, covered / float(window_min)) if window_min > 0 else 0.0
        missing_files[idx] = max(0, expected_steps - overlap_count)
        direct_accum[idx] = direct_total
        direct_rate[idx] = direct_total * 60.0 / float(window_min) if window_min > 0 else np.nan
        direct_peak_rate[idx] = peak_direct
        positive_steps_direct[idx] = positive_direct
        positive_steps_neighbor[idx] = positive_neighbor
        neighbor_peak_rate[idx] = peak_neighbor

        if neighbor_accum_grid is not None and neighbor_valid_grid is not None and neighbor_valid_grid.any():
            neighbor_accum_grid = np.where(neighbor_valid_grid, neighbor_accum_grid, np.nan)
            neighbor_accum_max[idx] = float(np.nanmax(neighbor_accum_grid))

    out["imerg_event_accum_mm"] = direct_accum
    out["imerg_event_rate_mmh"] = direct_rate
    out["imerg_peak_rate_mmh"] = direct_peak_rate
    out["imerg_3x3_event_accum_max_mm"] = neighbor_accum_max
    out["imerg_3x3_peak_rate_mmh"] = neighbor_peak_rate
    out["imerg_coverage_ratio"] = coverage_ratio
    out["imerg_covered_minutes"] = covered_minutes
    out["imerg_overlap_file_count"] = overlap_file_count
    out["imerg_expected_file_count"] = expected_file_count
    out["imerg_missing_file_count"] = missing_files
    out["imerg_positive_step_count"] = positive_steps_direct
    out["imerg_3x3_positive_step_count"] = positive_steps_neighbor
    out["imerg_any_positive"] = positive_steps_direct > 0
    out["imerg_3x3_any_positive"] = positive_steps_neighbor > 0
    out["imerg_used_fallback_cv_cell"] = used_fallback_cell
    out["imerg_complete_window"] = (
        (out["imerg_expected_file_count"] > 0)
        & (out["imerg_missing_file_count"] == 0)
        & (out["imerg_coverage_ratio"] >= 0.999)
    )
    return out


def build_summary(events_df: pd.DataFrame, subset_index: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "event_count": int(events_df.shape[0]),
        "subset_file_count_available": int(subset_index.shape[0]),
        "matched_complete_window_count": int(events_df["imerg_complete_window"].sum()),
        "matched_any_positive_count": int(events_df["imerg_any_positive"].sum()),
        "matched_any_positive_3x3_count": int(events_df["imerg_3x3_any_positive"].sum()),
    }

    for window_min, group in events_df.groupby("window_min", sort=True):
        key = str(int(window_min))
        summary[key] = {
            "event_count": int(group.shape[0]),
            "complete_window_count": int(group["imerg_complete_window"].sum()),
            "any_positive_count": int(group["imerg_any_positive"].sum()),
            "any_positive_3x3_count": int(group["imerg_3x3_any_positive"].sum()),
            "median_coverage_ratio": float(group["imerg_coverage_ratio"].median()),
            "median_event_mm_avamet": float(group["event_accum_mm"].median()),
            "median_event_mm_imerg": float(group["imerg_event_accum_mm"].median()),
        }

    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(args.events_input)
    events = events.sort_values(["timestamp_utc", "window_min", "station_id"]).reset_index(drop=True)
    if args.limit_events > 0:
        events = events.head(int(args.limit_events)).copy()
    if events.empty:
        raise RuntimeError(f"No events found in {args.events_input}")

    subset_index = build_subset_index(args.subset_root)
    station_cells = load_station_cells(args.station_alignment_csv, args.metadata_json)
    matched = process_events(events, subset_index, station_cells, int(args.cache_size))

    matched.to_parquet(args.output, index=False)
    summary = build_summary(matched, subset_index)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.summary_json}")
    print(f"Events processed: {matched.shape[0]}")
    print(f"Subset files available: {subset_index.shape[0]}")


if __name__ == "__main__":
    main()

