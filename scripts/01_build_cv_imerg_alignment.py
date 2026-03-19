from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Point


CV_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
CV_ISO_CODES = ["ES-A", "ES-CS", "ES-V"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build AVAMET to IMERG alignment products for the Valencian Community."
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv.csv"),
        help="CSV with AVAMET stations inside the Valencian Community.",
    )
    parser.add_argument(
        "--imerg-root",
        type=Path,
        default=Path("data/imerg"),
        help="IMERG download root. The first available HDF5 is used to read the grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where alignment outputs will be written.",
    )
    return parser


def load_cv_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    def normalize_text(value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            return value.encode("latin-1").decode("utf-8")
        except UnicodeError:
            return value

    admin1 = gpd.read_file(CV_ADMIN1_URL)
    spain = admin1[admin1["adm0_a3"] == "ESP"].copy()
    provinces = spain[spain["iso_3166_2"].isin(CV_ISO_CODES)].copy()
    provinces = provinces[["iso_3166_2", "name", "geometry"]].rename(
        columns={"iso_3166_2": "province_iso", "name": "province_name"}
    )
    provinces["province_name"] = provinces["province_name"].map(normalize_text)
    cv_union = provinces.dissolve()
    cv_union["name"] = "Valencian Community"
    return provinces, cv_union


def find_sample_imerg(imerg_root: Path) -> Path:
    matches = sorted(imerg_root.rglob("*.HDF5"))
    if not matches:
        raise FileNotFoundError(f"No IMERG HDF5 files found under {imerg_root}")
    return matches[0]


def load_imerg_grid(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(sample_path, "r") as handle:
        lon = handle["Grid/lon"][:]
        lat = handle["Grid/lat"][:]
    return lon.astype(np.float64), lat.astype(np.float64)


def assign_station_province(stations: pd.DataFrame, provinces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(stations["lon"], stations["lat"])]
    gdf = gpd.GeoDataFrame(stations.copy(), geometry=geometry, crs="EPSG:4326")

    gdf["province_iso"] = None
    gdf["province_name"] = None
    for _, province in provinces.iterrows():
        mask = gdf.geometry.intersects(province.geometry)
        gdf.loc[mask, "province_iso"] = province["province_iso"]
        gdf.loc[mask, "province_name"] = province["province_name"]
    return gdf


def build_imerg_subset_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    cv_polygon: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    minx, miny, maxx, maxy = (float(v) for v in cv_polygon.total_bounds)
    lon_idx = np.where((lon >= minx) & (lon <= maxx))[0]
    lat_idx = np.where((lat >= miny) & (lat <= maxy))[0]
    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError("CV bounding box does not intersect the IMERG grid.")

    cv_geom = cv_polygon.geometry.iloc[0]
    rows: list[dict[str, object]] = []
    for i in lon_idx:
        for j in lat_idx:
            cell_lon = float(lon[i])
            cell_lat = float(lat[j])
            inside = Point(cell_lon, cell_lat).intersects(cv_geom)
            rows.append(
                {
                    "lon_idx": int(i),
                    "lat_idx": int(j),
                    "lon": cell_lon,
                    "lat": cell_lat,
                    "inside_cv_polygon": bool(inside),
                }
            )

    grid_df = pd.DataFrame(rows).sort_values(["lon_idx", "lat_idx"]).reset_index(drop=True)
    metadata = {
        "bbox": {
            "min_lon": minx,
            "max_lon": maxx,
            "min_lat": miny,
            "max_lat": maxy,
        },
        "lon_index_range": {
            "start": int(lon_idx[0]),
            "end": int(lon_idx[-1]),
            "count": int(lon_idx.size),
        },
        "lat_index_range": {
            "start": int(lat_idx[0]),
            "end": int(lat_idx[-1]),
            "count": int(lat_idx.size),
        },
        "bbox_cell_count": int(grid_df.shape[0]),
        "polygon_cell_count": int(grid_df["inside_cv_polygon"].sum()),
        "grid_resolution_deg": {
            "lon": float(lon[1] - lon[0]),
            "lat": float(lat[1] - lat[0]),
        },
    }
    return grid_df, metadata


def attach_nearest_imerg_cell(
    stations: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    inside_grid = grid_df[grid_df["inside_cv_polygon"]].copy()
    inside_lon = inside_grid["lon"].to_numpy(dtype=np.float64)
    inside_lat = inside_grid["lat"].to_numpy(dtype=np.float64)
    inside_lon_idx = inside_grid["lon_idx"].to_numpy(dtype=np.int32)
    inside_lat_idx = inside_grid["lat_idx"].to_numpy(dtype=np.int32)

    grid_lookup = {
        (int(row.lon_idx), int(row.lat_idx)): bool(row.inside_cv_polygon)
        for row in grid_df.itertuples(index=False)
    }

    records: list[dict[str, object]] = []
    for row in stations.itertuples(index=False):
        lon_idx = int(np.abs(lon - float(row.lon)).argmin())
        lat_idx = int(np.abs(lat - float(row.lat)).argmin())
        imerg_lon = float(lon[lon_idx])
        imerg_lat = float(lat[lat_idx])

        inside_dist2 = (inside_lon - float(row.lon)) ** 2 + (inside_lat - float(row.lat)) ** 2
        inside_pos = int(np.argmin(inside_dist2))
        cv_lon_idx = int(inside_lon_idx[inside_pos])
        cv_lat_idx = int(inside_lat_idx[inside_pos])
        cv_imerg_lon = float(inside_lon[inside_pos])
        cv_imerg_lat = float(inside_lat[inside_pos])

        records.append(
            {
                "station_id": row.station_id,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "alt": float(row.alt) if pd.notna(row.alt) else None,
                "n_obs": int(row.n_obs),
                "province_iso": row.province_iso,
                "province_name": row.province_name,
                "imerg_lon_idx": lon_idx,
                "imerg_lat_idx": lat_idx,
                "imerg_lon": imerg_lon,
                "imerg_lat": imerg_lat,
                "delta_lon_deg": float(row.lon) - imerg_lon,
                "delta_lat_deg": float(row.lat) - imerg_lat,
                "imerg_cell_in_cv_polygon": grid_lookup.get((lon_idx, lat_idx), False),
                "imerg_cv_lon_idx": cv_lon_idx,
                "imerg_cv_lat_idx": cv_lat_idx,
                "imerg_cv_lon": cv_imerg_lon,
                "imerg_cv_lat": cv_imerg_lat,
                "delta_cv_lon_deg": float(row.lon) - cv_imerg_lon,
                "delta_cv_lat_deg": float(row.lat) - cv_imerg_lat,
            }
        )
    return pd.DataFrame(records).sort_values("station_id").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(args.station_csv)
    provinces, cv_polygon = load_cv_layers()
    sample_imerg = find_sample_imerg(args.imerg_root)
    lon, lat = load_imerg_grid(sample_imerg)

    stations_gdf = assign_station_province(stations, provinces)
    grid_df, metadata = build_imerg_subset_grid(lon, lat, cv_polygon)
    mapping_df = attach_nearest_imerg_cell(stations_gdf, lon, lat, grid_df)

    station_out = args.output_dir / "avamet_station_inventory_cv_imerg.csv"
    grid_out = args.output_dir / "imerg_cv_grid_cells.csv"
    metadata_out = args.output_dir / "imerg_cv_subset_metadata.json"

    mapping_df.to_csv(station_out, index=False, encoding="utf-8-sig")
    grid_df.to_csv(grid_out, index=False)

    metadata["sample_imerg_file"] = str(sample_imerg)
    metadata["station_count"] = int(mapping_df.shape[0])
    metadata["stations_with_nearest_cell_inside_cv_polygon"] = int(mapping_df["imerg_cell_in_cv_polygon"].sum())
    metadata["stations_with_cv_fallback_mapping"] = int(mapping_df.shape[0])
    metadata["province_counts"] = (
        mapping_df["province_name"].fillna("Unknown").value_counts().sort_index().to_dict()
    )
    with metadata_out.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Sample IMERG file: {sample_imerg}")
    print(f"Stations written: {station_out}")
    print(f"Grid written: {grid_out}")
    print(f"Metadata written: {metadata_out}")
    print(f"IMERG bbox cells: {metadata['bbox_cell_count']}")
    print(f"IMERG polygon cells: {metadata['polygon_cell_count']}")
    print(
        "Stations with nearest IMERG cell already inside CV polygon: "
        f"{metadata['stations_with_nearest_cell_inside_cv_polygon']}"
    )


if __name__ == "__main__":
    main()

