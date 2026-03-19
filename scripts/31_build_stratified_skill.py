from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


COASTLINE_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
PROJECTED_CRS = "EPSG:32630"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build stratified AVAMET-IMERG event-based skill summaries for the filtered "
            "common-period event table."
        )
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common_excluding_low_plausibility.parquet"),
        help="Filtered common-period event parquet.",
    )
    parser.add_argument(
        "--station-inventory-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_imerg.csv"),
        help="Station inventory CSV.",
    )
    parser.add_argument(
        "--output-stations-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_strata.csv"),
        help="Output station inventory with stratification fields.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_stratified_skill.csv"),
        help="Output long-form stratified skill CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_stratified_skill_summary.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--coastal-threshold-km",
        type=float,
        default=25.0,
        help="Distance-to-coast threshold used to label a station as coastal.",
    )
    return parser


def scalar_or_none(value: float | int | np.floating | np.integer | None) -> float | int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den > 0.0
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value)
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def load_station_strata(station_inventory_csv: Path, coastal_threshold_km: float) -> pd.DataFrame:
    stations = pd.read_csv(station_inventory_csv).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["province_name"] = stations["province_name"].map(normalize_text)
    stations["alt"] = pd.to_numeric(stations["alt"], errors="coerce")

    points = gpd.GeoDataFrame(
        stations[["station_id", "lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    )

    coastline = gpd.read_file(COASTLINE_URL)
    minx, miny, maxx, maxy = points.total_bounds
    bbox = (minx - 1.0, miny - 1.0, maxx + 1.0, maxy + 1.0)
    coastline = coastline.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
    if coastline.empty:
        raise ValueError("Coastline subset is empty; cannot compute coastal distance.")

    points_projected = points.to_crs(PROJECTED_CRS)
    coastline_projected = coastline.to_crs(PROJECTED_CRS)
    coastline_union = coastline_projected.geometry.unary_union

    distances_km = points_projected.geometry.distance(coastline_union) / 1000.0
    stations["distance_to_coast_km"] = distances_km.to_numpy(dtype=float, copy=False)
    threshold = float(coastal_threshold_km)
    stations["coast_group"] = np.where(stations["distance_to_coast_km"] <= threshold, "Coastal", "Interior")

    alt = stations["alt"]
    stations["altitude_group"] = np.select(
        [
            alt <= 200.0,
            (alt > 200.0) & (alt <= 800.0),
            alt > 800.0,
        ],
        [
            "<=200 m",
            "200-800 m",
            ">800 m",
        ],
        default="Unknown",
    )
    return stations


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    event_count = int(group.shape[0])
    station_count = int(group["station_id"].astype(str).nunique()) if event_count else 0
    if event_count == 0:
        return {
            "event_count": 0,
            "station_count": 0,
            "pod_point": None,
            "pod_3x3": None,
            "median_ratio_point": None,
            "median_ratio_3x3": None,
            "median_bias_mm_point": None,
            "median_bias_mm_3x3": None,
            "mae_mm_point": None,
            "mae_mm_3x3": None,
            "rmse_mm_point": None,
            "rmse_mm_3x3": None,
            "median_distance_km": None,
            "underestimate_fraction_point": None,
            "underestimate_fraction_3x3": None,
            "zero_response_fraction_point": None,
            "zero_response_fraction_3x3": None,
        }

    avamet = pd.to_numeric(group["event_accum_mm"], errors="coerce").astype(float)
    imerg_point = pd.to_numeric(group["imerg_event_accum_mm"], errors="coerce").astype(float)
    imerg_3x3 = pd.to_numeric(group["imerg_3x3_event_accum_max_mm"], errors="coerce").astype(float)
    distance = pd.to_numeric(group["imerg_best_3x3_distance_km"], errors="coerce").astype(float)

    ratio_point = safe_ratio(imerg_point, avamet)
    ratio_3x3 = safe_ratio(imerg_3x3, avamet)
    hits_point = group["imerg_any_positive"].fillna(False).astype(bool)
    hits_3x3 = group["imerg_3x3_any_positive"].fillna(False).astype(bool)
    bias_point = imerg_point - avamet
    bias_3x3 = imerg_3x3 - avamet

    return {
        "event_count": event_count,
        "station_count": station_count,
        "pod_point": scalar_or_none(hits_point.mean()),
        "pod_3x3": scalar_or_none(hits_3x3.mean()),
        "median_ratio_point": scalar_or_none(ratio_point.median()),
        "median_ratio_3x3": scalar_or_none(ratio_3x3.median()),
        "median_bias_mm_point": scalar_or_none(bias_point.median()),
        "median_bias_mm_3x3": scalar_or_none(bias_3x3.median()),
        "mae_mm_point": scalar_or_none(bias_point.abs().mean()),
        "mae_mm_3x3": scalar_or_none(bias_3x3.abs().mean()),
        "rmse_mm_point": scalar_or_none(np.sqrt(np.mean(np.square(bias_point)))),
        "rmse_mm_3x3": scalar_or_none(np.sqrt(np.mean(np.square(bias_3x3)))),
        "median_distance_km": scalar_or_none(distance.median()),
        "underestimate_fraction_point": scalar_or_none((imerg_point < avamet).mean()),
        "underestimate_fraction_3x3": scalar_or_none((imerg_3x3 < avamet).mean()),
        "zero_response_fraction_point": scalar_or_none((imerg_point <= 0.0).mean()),
        "zero_response_fraction_3x3": scalar_or_none((imerg_3x3 <= 0.0).mean()),
    }


def build_rows(events: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def append_family_rows(frame: pd.DataFrame, family: str, value: str) -> None:
        base = {
            "strata_family": family,
            "strata_value": value,
            "window_min": "all",
        }
        rows.append({**base, **summarize_group(frame)})
        for window_min, group in frame.groupby("window_min", sort=True):
            rows.append(
                {
                    "strata_family": family,
                    "strata_value": value,
                    "window_min": str(int(window_min)),
                    **summarize_group(group),
                }
            )

    append_family_rows(events, "all", "All events")

    for value, group in events.groupby("season_group", sort=False):
        append_family_rows(group, "season", str(value))
    for value, group in events.groupby("coast_group", sort=False):
        append_family_rows(group, "coast", str(value))
    for value, group in events.groupby("province_name", sort=False):
        append_family_rows(group, "province", str(value))
    for value, group in events.groupby("altitude_group", sort=False):
        append_family_rows(group, "altitude", str(value))

    return rows


def main() -> None:
    args = build_parser().parse_args()
    args.output_stations_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    stations = load_station_strata(args.station_inventory_csv, args.coastal_threshold_km)
    stations.to_csv(args.output_stations_csv, index=False)

    events = pd.read_parquet(args.events_input).copy()
    events["station_id"] = events["station_id"].astype(str)
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], errors="coerce", utc=False)
    events["month"] = events["timestamp_utc"].dt.month
    events["season_group"] = np.where(events["month"].isin([9, 10, 11]), "SON", "Rest of year")

    amplitude_cols = ["station_id", "window_min", "timestamp_utc", "imerg_best_3x3_distance_km"]
    amplitude = pd.read_parquet(
        Path("results/avamet_cv_events_imerg_amplitude_displacement.parquet"),
        columns=amplitude_cols,
    ).copy()
    amplitude["station_id"] = amplitude["station_id"].astype(str)
    amplitude["timestamp_utc"] = pd.to_datetime(amplitude["timestamp_utc"], errors="coerce", utc=False)
    amplitude["window_min"] = pd.to_numeric(amplitude["window_min"], errors="coerce").astype(int)
    amplitude = amplitude.drop_duplicates(subset=["station_id", "window_min", "timestamp_utc"], keep="first")

    events["window_min"] = pd.to_numeric(events["window_min"], errors="coerce").astype(int)
    events = events.merge(
        amplitude,
        on=["station_id", "window_min", "timestamp_utc"],
        how="left",
    )
    events = events.merge(
        stations[
            [
                "station_id",
                "province_name",
                "distance_to_coast_km",
                "coast_group",
                "altitude_group",
                "alt",
            ]
        ],
        on="station_id",
        how="left",
    )

    rows = build_rows(events)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_csv, index=False)

    json_summary = {
        "metadata": {
            "events_input": str(args.events_input),
            "station_inventory_csv": str(args.station_inventory_csv),
            "coastal_threshold_km": float(args.coastal_threshold_km),
            "coast_group_rule": f"Coastal if distance_to_coast_km <= {float(args.coastal_threshold_km):.1f}",
            "altitude_group_rule": {
                "<=200 m": "alt <= 200",
                "200-800 m": "200 < alt <= 800",
                ">800 m": "alt > 800",
            },
        },
        "station_group_counts": {
            "coast": stations["coast_group"].value_counts().to_dict(),
            "province": stations["province_name"].value_counts().to_dict(),
            "altitude": stations["altitude_group"].value_counts().to_dict(),
        },
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_stations_csv}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Rows: {summary_df.shape[0]}")


if __name__ == "__main__":
    main()

