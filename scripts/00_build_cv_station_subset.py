from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


CV_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
CV_ISO_CODES = ["ES-A", "ES-CS", "ES-V"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build AVAMET station inventory and Valencian Community subset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/avamet_all.parquet"),
        help="Input AVAMET parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where station subset outputs will be written.",
    )
    parser.add_argument(
        "--write-filtered-parquet",
        action="store_true",
        help="If set, write an observations parquet filtered to the CV station list.",
    )
    parser.add_argument(
        "--filtered-parquet",
        type=Path,
        default=Path("results/avamet_cv.parquet"),
        help="Output parquet path when --write-filtered-parquet is enabled.",
    )
    return parser


def load_station_inventory(parquet_path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            station_id,
            AVG(lat) AS lat,
            AVG(lon) AS lon,
            AVG(alt) AS alt,
            COUNT(*) AS n_obs
        FROM read_parquet('{parquet_path.as_posix()}')
        WHERE lat IS NOT NULL
          AND lon IS NOT NULL
        GROUP BY station_id
        ORDER BY station_id
    """
    return con.execute(query).fetchdf()


def load_cv_polygon() -> gpd.GeoDataFrame:
    admin1 = gpd.read_file(CV_ADMIN1_URL)
    spain = admin1[admin1["adm0_a3"] == "ESP"].copy()
    cv_provinces = spain[spain["iso_3166_2"].isin(CV_ISO_CODES)].copy()
    cv_union = cv_provinces.dissolve()
    cv_union["name"] = "Valencian Community"
    return cv_union


def classify_stations(stations_df: pd.DataFrame, cv_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(stations_df["lon"], stations_df["lat"])]
    stations = gpd.GeoDataFrame(stations_df.copy(), geometry=geometry, crs="EPSG:4326")
    cv_geom = cv_polygon.geometry.iloc[0]
    stations["inside_cv"] = stations.geometry.intersects(cv_geom)
    bounds = cv_polygon.total_bounds
    stations["inside_cv_bbox"] = (
        (stations["lon"] >= bounds[0])
        & (stations["lon"] <= bounds[2])
        & (stations["lat"] >= bounds[1])
        & (stations["lat"] <= bounds[3])
    )
    return stations


def write_station_outputs(stations: gpd.GeoDataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv = output_dir / "avamet_station_inventory_all.csv"
    cv_csv = output_dir / "avamet_station_inventory_cv.csv"
    cv_ids_csv = output_dir / "avamet_station_ids_cv.csv"

    export_cols = ["station_id", "lat", "lon", "alt", "n_obs", "inside_cv", "inside_cv_bbox"]
    stations[export_cols].to_csv(all_csv, index=False)
    stations.loc[stations["inside_cv"], export_cols].to_csv(cv_csv, index=False)
    stations.loc[stations["inside_cv"], ["station_id"]].rename(columns={"station_id": "st"}).to_csv(
        cv_ids_csv, index=False
    )
    return all_csv, cv_csv, cv_ids_csv


def write_filtered_parquet(
    parquet_path: Path,
    inside_station_ids: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    station_df = pd.DataFrame({"station_id": inside_station_ids})
    con.register("cv_stations", station_df)
    query = f"""
        COPY (
            SELECT src.*
            FROM read_parquet('{parquet_path.as_posix()}') AS src
            INNER JOIN cv_stations AS cv
                ON src.station_id = cv.station_id
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)


def main() -> None:
    args = build_parser().parse_args()

    stations_df = load_station_inventory(args.input)
    cv_polygon = load_cv_polygon()
    stations = classify_stations(stations_df, cv_polygon)
    all_csv, cv_csv, cv_ids_csv = write_station_outputs(stations, args.output_dir)

    total = len(stations)
    inside = int(stations["inside_cv"].sum())
    outside = total - inside
    bbox_hits = int(stations["inside_cv_bbox"].sum())
    minx, miny, maxx, maxy = (float(v) for v in cv_polygon.total_bounds)

    print(f"Total stations: {total}")
    print(f"Inside CV polygon: {inside}")
    print(f"Outside CV polygon: {outside}")
    print(f"Inside CV bbox: {bbox_hits}")
    print(f"CV bbox: lon [{minx:.6f}, {maxx:.6f}] | lat [{miny:.6f}, {maxy:.6f}]")
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {cv_csv}")
    print(f"Wrote: {cv_ids_csv}")

    if args.write_filtered_parquet:
        inside_station_ids = stations.loc[stations["inside_cv"], "station_id"].astype(str).tolist()
        write_filtered_parquet(args.input, inside_station_ids, args.filtered_parquet)
        print(f"Wrote: {args.filtered_parquet}")


if __name__ == "__main__":
    main()

