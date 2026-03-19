from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import Normalize, TwoSlopeNorm


CV_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"
CV_ISO_CODES = ["ES-A", "ES-CS", "ES-V"]
FILE_TIME_RE = re.compile(r"(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparative AVAMET vs IMERG maps for selected DANA case-study days in the Comunitat Valenciana (eastern Spain)."
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common.parquet"),
        help="Event parquet used to rank DANA case-study days.",
    )
    parser.add_argument(
        "--qc-input",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="AVAMET QC parquet used to build daily station accumulations.",
    )
    parser.add_argument(
        "--station-inventory-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv.csv"),
        help="Station inventory CSV with coordinates.",
    )
    parser.add_argument(
        "--station-alignment-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_imerg.csv"),
        help="Station-to-IMERG alignment CSV used to project AVAMET onto the IMERG grid.",
    )
    parser.add_argument(
        "--subset-root",
        type=Path,
        default=Path("data/imerg_cv"),
        help="Root directory with IMERG CV subsets.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/dana_comparison_maps_paper.png"),
        help="Output PNG figure.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/dana_candidate_days.csv"),
        help="Output CSV with selected case-study days.",
    )
    parser.add_argument(
        "--top-days",
        type=int,
        default=2,
        help="Number of case-study days to map when --dates is not provided.",
    )
    parser.add_argument(
        "--dates",
        type=str,
        default="",
        help="Optional comma-separated list of dates (YYYY-MM-DD). Overrides automatic selection.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Minimum daily coverage ratio to display an AVAMET station in color.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200000,
        help="Batch size for streaming the AVAMET QC parquet.",
    )
    return parser


def to_minute_index(values: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    ts = pd.to_datetime(values, errors="coerce", utc=False)
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    return (ts_ns // 60_000_000_000).astype(np.int64, copy=False)


def parse_subset_date(path: Path) -> pd.Timestamp:
    match = FILE_TIME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse IMERG timestamp from {path.name}")
    return pd.to_datetime(match.group("date"), format="%Y%m%d", errors="raise")


def normalize_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value


def load_cv_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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


def select_candidate_days(events_path: Path, top_days: int) -> pd.DataFrame:
    events = pd.read_parquet(
        events_path,
        columns=["timestamp_utc", "window_min", "station_id", "event_accum_mm", "imerg_event_accum_mm"],
    )
    events["timestamp_utc"] = pd.to_datetime(events["timestamp_utc"], errors="coerce", utc=False)
    events["date"] = events["timestamp_utc"].dt.normalize()
    events["month"] = events["timestamp_utc"].dt.month

    focus = events.loc[(events["window_min"] == 360) & (events["month"].isin([9, 10, 11]))].copy()
    daily = (
        focus.groupby("date")
        .agg(
            event_count=("station_id", "count"),
            station_count=("station_id", "nunique"),
            avamet_sum_mm=("event_accum_mm", "sum"),
            avamet_max_mm=("event_accum_mm", "max"),
            imerg_sum_mm=("imerg_event_accum_mm", "sum"),
        )
        .reset_index()
        .sort_values(["station_count", "avamet_sum_mm", "event_count"], ascending=[False, False, False])
        .head(max(1, int(top_days)))
        .reset_index(drop=True)
    )
    return daily


def parse_dates_arg(raw: str) -> list[pd.Timestamp]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        return []
    return [pd.to_datetime(value, format="%Y-%m-%d", errors="raise").normalize() for value in values]


def build_selected_days(events_path: Path, raw_dates: str, top_days: int) -> pd.DataFrame:
    manual_dates = parse_dates_arg(raw_dates)
    if manual_dates:
        selected = pd.DataFrame({"date": manual_dates})
        selected["event_count"] = np.nan
        selected["station_count"] = np.nan
        selected["avamet_sum_mm"] = np.nan
        selected["avamet_max_mm"] = np.nan
        selected["imerg_sum_mm"] = np.nan
        return selected
    return select_candidate_days(events_path, top_days)


def accumulate_daily_avamet(
    qc_path: Path,
    stations: pd.DataFrame,
    selected_days: pd.DataFrame,
    batch_size: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    station_ids = stations["station_id"].astype(str).tolist()
    station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
    n_days = int(selected_days.shape[0])
    n_stations = int(len(station_ids))
    daily_accum = np.zeros((n_days, n_stations), dtype=np.float64)
    daily_covered = np.zeros((n_days, n_stations), dtype=np.float64)

    day_start = to_minute_index(selected_days["date"])
    day_end = day_start + 1440

    parquet = pq.ParquetFile(qc_path)
    columns = ["station_id", "timestamp_utc", "delta_prev_min", "precip_mm_qc", "usable_interval_extremes"]

    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        batch_df = batch.to_pandas()
        if batch_df.empty:
            continue

        batch_df["station_id"] = batch_df["station_id"].astype(str)
        batch_df = batch_df.loc[batch_df["station_id"].isin(station_to_idx)].copy()
        if batch_df.empty:
            continue

        station_idx = batch_df["station_id"].map(station_to_idx).to_numpy(dtype=np.int64, copy=False)
        end_min = to_minute_index(batch_df["timestamp_utc"])
        duration_min = pd.to_numeric(batch_df["delta_prev_min"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        duration_min = np.where(np.isfinite(duration_min) & (duration_min > 0), duration_min, 0.0)
        start_min = end_min.astype(np.float64) - duration_min
        precip_mm = pd.to_numeric(batch_df["precip_mm_qc"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        precip_mm = np.where(np.isfinite(precip_mm), precip_mm, 0.0)
        usable = batch_df["usable_interval_extremes"].fillna(False).to_numpy(dtype=bool, copy=False)

        for day_idx in range(n_days):
            overlap = np.minimum(end_min.astype(np.float64), float(day_end[day_idx])) - np.maximum(
                start_min, float(day_start[day_idx])
            )
            mask = (overlap > 0.0) & usable & (duration_min > 0.0)
            if not np.any(mask):
                continue

            frac = overlap[mask] / duration_min[mask]
            np.add.at(daily_accum[day_idx], station_idx[mask], precip_mm[mask] * frac)
            np.add.at(daily_covered[day_idx], station_idx[mask], overlap[mask])

    per_day_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for day_idx, row in selected_days.iterrows():
        day_df = stations.copy()
        day_df["date"] = pd.Timestamp(row["date"])
        day_df["avamet_daily_mm"] = daily_accum[day_idx]
        day_df["avamet_coverage_ratio"] = np.clip(daily_covered[day_idx] / 1440.0, 0.0, 1.0)
        per_day_frames.append(day_df)

        valid_mask = day_df["avamet_coverage_ratio"] >= 0.8
        summary_rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "event_count_360": row.get("event_count", np.nan),
                "station_count_360": row.get("station_count", np.nan),
                "avamet_station_count_valid": int(valid_mask.sum()),
                "avamet_station_sum_mm": float(day_df.loc[valid_mask, "avamet_daily_mm"].sum()),
                "avamet_station_max_mm": float(day_df.loc[valid_mask, "avamet_daily_mm"].max()) if np.any(valid_mask) else np.nan,
            }
        )

    return per_day_frames, pd.DataFrame(summary_rows)


def accumulate_daily_imerg(subset_root: Path, selected_days: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    selected_dates = [pd.Timestamp(value).normalize() for value in selected_days["date"].tolist()]
    files_by_date: dict[pd.Timestamp, list[Path]] = {date: [] for date in selected_dates}

    for path in sorted(subset_root.rglob("*.cv.npz")):
        date = parse_subset_date(path).normalize()
        if date in files_by_date:
            files_by_date[date].append(path)

    per_day_grids: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for date in selected_dates:
        files = files_by_date.get(date, [])
        if not files:
            raise FileNotFoundError(f"No IMERG subset files found for {date.date()}")

        daily_sum: np.ndarray | None = None
        valid_mask: np.ndarray | None = None
        lon: np.ndarray | None = None
        lat: np.ndarray | None = None
        inside_mask: np.ndarray | None = None

        for path in files:
            with np.load(path) as handle:
                precip = handle["precipitation"].astype(np.float32, copy=False)
                if daily_sum is None:
                    daily_sum = np.zeros_like(precip, dtype=np.float64)
                    valid_mask = np.zeros_like(precip, dtype=bool)
                    lon = handle["lon"].astype(np.float32, copy=False)
                    lat = handle["lat"].astype(np.float32, copy=False)
                    inside_mask = handle["inside_cv_polygon"].astype(bool, copy=False)
                finite = np.isfinite(precip)
                daily_sum[finite] += precip[finite].astype(np.float64) * 0.5
                valid_mask[finite] = True

        assert daily_sum is not None
        assert valid_mask is not None
        assert lon is not None
        assert lat is not None
        assert inside_mask is not None

        daily_sum[~valid_mask] = np.nan
        per_day_grids.append(
            {
                "date": date,
                "daily_grid_mm": daily_sum,
                "lon": lon,
                "lat": lat,
                "inside_mask": inside_mask,
            }
        )

        finite = np.isfinite(daily_sum)
        summary_rows.append(
            {
                "date": date,
                "imerg_file_count": len(files),
                "imerg_grid_max_mm": float(np.nanmax(daily_sum)) if np.any(finite) else np.nan,
                "imerg_grid_mean_mm": float(np.nanmean(daily_sum)) if np.any(finite) else np.nan,
            }
        )

    return per_day_grids, pd.DataFrame(summary_rows)


def build_avamet_cell_grids(
    avamet_frames: list[pd.DataFrame],
    imerg_grids: list[dict[str, object]],
    coverage_threshold: float,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    grids: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []

    for day_idx, frame in enumerate(avamet_frames):
        template = imerg_grids[day_idx]["daily_grid_mm"]
        cell_grid = np.full(template.shape, np.nan, dtype=np.float64)
        valid = (
            (frame["avamet_coverage_ratio"] >= coverage_threshold)
            & frame["imerg_cv_lon_local"].notna()
            & frame["imerg_cv_lat_local"].notna()
        )
        valid_frame = frame.loc[valid].copy()
        if not valid_frame.empty:
            grouped = (
                valid_frame.groupby(["imerg_cv_lon_local", "imerg_cv_lat_local"])["avamet_daily_mm"]
                .median()
                .reset_index()
            )
            lon_idx = grouped["imerg_cv_lon_local"].astype(int).to_numpy(dtype=np.int64, copy=False)
            lat_idx = grouped["imerg_cv_lat_local"].astype(int).to_numpy(dtype=np.int64, copy=False)
            cell_grid[lon_idx, lat_idx] = grouped["avamet_daily_mm"].to_numpy(dtype=np.float64, copy=False)

        grids.append(cell_grid)
        finite = np.isfinite(cell_grid)
        summary_rows.append(
            {
                "date": pd.Timestamp(frame["date"].iloc[0]),
                "avamet_cell_count_valid": int(np.count_nonzero(finite)),
                "avamet_cell_max_mm": float(np.nanmax(cell_grid)) if np.any(finite) else np.nan,
                "avamet_cell_mean_mm": float(np.nanmean(cell_grid)) if np.any(finite) else np.nan,
            }
        )

    return grids, pd.DataFrame(summary_rows)


def annotate_provinces(ax: plt.Axes, provinces: gpd.GeoDataFrame) -> None:
    for row in provinces.itertuples(index=False):
        point = row.geometry.representative_point()
        ax.text(
            float(point.x),
            float(point.y),
            str(row.province_name),
            fontsize=8,
            ha="center",
            va="center",
            color="#2a2a2a",
            alpha=0.8,
            zorder=5,
        )


def style_map_axes(
    ax: plt.Axes,
    provinces: gpd.GeoDataFrame,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    x_pad: float,
    y_pad: float,
) -> None:
    provinces.boundary.plot(ax=ax, color="#202020", linewidth=0.8)
    annotate_provinces(ax, provinces)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_facecolor("#f6f3ef")


def plot_maps(
    selected_days: pd.DataFrame,
    avamet_frames: list[pd.DataFrame],
    avamet_cell_grids: list[np.ndarray],
    imerg_grids: list[dict[str, object]],
    provinces: gpd.GeoDataFrame,
    cv_union: gpd.GeoDataFrame,
    output_png: Path,
    coverage_threshold: float,
) -> None:
    positive_values: list[np.ndarray] = []
    for frame in avamet_frames:
        vals = frame.loc[frame["avamet_coverage_ratio"] >= coverage_threshold, "avamet_daily_mm"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            positive_values.append(vals)
    for grid in imerg_grids:
        vals = grid["daily_grid_mm"]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size:
            positive_values.append(vals)
    for grid in avamet_cell_grids:
        vals = grid[np.isfinite(grid) & (grid > 0)]
        if vals.size:
            positive_values.append(vals)

    if positive_values:
        all_values = np.concatenate(positive_values)
        vmax = float(np.nanquantile(all_values, 0.99))
        vmax = max(vmax, 10.0)
    else:
        vmax = 10.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    diff_values: list[np.ndarray] = []
    for day_idx, imerg in enumerate(imerg_grids):
        diff = imerg["daily_grid_mm"] - avamet_cell_grids[day_idx]
        vals = diff[np.isfinite(diff)]
        if vals.size:
            diff_values.append(vals)
    if diff_values:
        all_diff = np.concatenate(diff_values)
        diff_abs = float(np.nanquantile(np.abs(all_diff), 0.98))
        diff_abs = max(diff_abs, 10.0)
    else:
        diff_abs = 10.0
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    diff_cmap = plt.get_cmap("RdBu_r")

    n_days = len(avamet_frames)
    fig, axes = plt.subplots(
        nrows=n_days,
        ncols=3,
        figsize=(16, max(5, 4.8 * n_days)),
        constrained_layout=True,
    )
    if n_days == 1:
        axes = np.array([axes])

    bounds = cv_union.total_bounds
    x_min, y_min, x_max, y_max = bounds
    x_pad = 0.15
    y_pad = 0.15

    mappable_main = None
    mappable_diff = None
    for day_idx in range(n_days):
        day_label = pd.Timestamp(selected_days.iloc[day_idx]["date"]).strftime("%Y-%m-%d")
        avamet_df = avamet_frames[day_idx]
        avamet_cell_grid = avamet_cell_grids[day_idx]
        imerg = imerg_grids[day_idx]
        diff_grid = imerg["daily_grid_mm"] - avamet_cell_grid

        ax_left = axes[day_idx, 0]
        style_map_axes(ax_left, provinces, x_min, y_min, x_max, y_max, x_pad, y_pad)
        valid = avamet_df["avamet_coverage_ratio"] >= coverage_threshold
        if np.any(valid):
            mappable_main = ax_left.scatter(
                avamet_df.loc[valid, "lon"],
                avamet_df.loc[valid, "lat"],
                c=avamet_df.loc[valid, "avamet_daily_mm"],
                cmap=cmap,
                norm=norm,
                s=20,
                edgecolors="black",
                linewidths=0.2,
                zorder=3,
            )
        invalid = ~valid
        if np.any(invalid):
            ax_left.scatter(
                avamet_df.loc[invalid, "lon"],
                avamet_df.loc[invalid, "lat"],
                color="#c7c7c7",
                s=10,
                linewidths=0,
                zorder=2,
            )
        ax_left.set_title(f"AVAMET stations | {day_label}")

        ax_mid = axes[day_idx, 1]
        style_map_axes(ax_mid, provinces, x_min, y_min, x_max, y_max, x_pad, y_pad)
        lon_grid, lat_grid = np.meshgrid(imerg["lon"], imerg["lat"], indexing="xy")
        mesh_main = ax_mid.pcolormesh(
            lon_grid,
            lat_grid,
            imerg["daily_grid_mm"].T,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
        if mappable_main is None:
            mappable_main = mesh_main
        ax_mid.set_title(f"IMERG grid | {day_label}")

        ax_right = axes[day_idx, 2]
        style_map_axes(ax_right, provinces, x_min, y_min, x_max, y_max, x_pad, y_pad)
        mesh_diff = ax_right.pcolormesh(
            lon_grid,
            lat_grid,
            diff_grid.T,
            cmap=diff_cmap,
            norm=diff_norm,
            shading="nearest",
        )
        if mappable_diff is None:
            mappable_diff = mesh_diff
        ax_right.set_title(f"IMERG - AVAMET cell median | {day_label}")

    if mappable_main is not None:
        cbar_main = fig.colorbar(mappable_main, ax=axes[:, :2], shrink=0.9, pad=0.02)
        cbar_main.set_label("Daily precipitation (mm; map context)")
    if mappable_diff is not None:
        cbar_diff = fig.colorbar(mappable_diff, ax=axes[:, 2], shrink=0.9, pad=0.02)
        cbar_diff.set_label("Difference (mm)")

    fig.suptitle(
        "Selected DANA Case-Study Days: AVAMET vs IMERG over the Comunitat Valenciana (eastern Spain)",
        fontsize=15,
        fontweight="bold",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    selected_days = build_selected_days(args.events_input, args.dates, int(args.top_days))
    if selected_days.empty:
        raise RuntimeError("No DANA case-study days were selected.")

    stations = pd.read_csv(args.station_inventory_csv)
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.sort_values("station_id").reset_index(drop=True)
    alignment = pd.read_csv(args.station_alignment_csv)
    alignment["station_id"] = alignment["station_id"].astype(str)
    lon_min = int(alignment["imerg_cv_lon_idx"].min())
    lat_min = int(alignment["imerg_cv_lat_idx"].min())
    alignment["imerg_cv_lon_local"] = alignment["imerg_cv_lon_idx"].astype(int) - lon_min
    alignment["imerg_cv_lat_local"] = alignment["imerg_cv_lat_idx"].astype(int) - lat_min
    stations = stations.merge(
        alignment[["station_id", "imerg_cv_lon_local", "imerg_cv_lat_local"]],
        on="station_id",
        how="left",
    )

    avamet_frames, avamet_summary = accumulate_daily_avamet(
        qc_path=args.qc_input,
        stations=stations,
        selected_days=selected_days,
        batch_size=int(args.batch_size),
    )
    imerg_grids, imerg_summary = accumulate_daily_imerg(args.subset_root, selected_days)
    avamet_cell_grids, avamet_cell_summary = build_avamet_cell_grids(
        avamet_frames=avamet_frames,
        imerg_grids=imerg_grids,
        coverage_threshold=float(args.coverage_threshold),
    )

    diff_rows: list[dict[str, object]] = []
    for day_idx, row in selected_days.iterrows():
        diff_grid = imerg_grids[day_idx]["daily_grid_mm"] - avamet_cell_grids[day_idx]
        finite = np.isfinite(diff_grid)
        diff_rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "diff_cell_count": int(np.count_nonzero(finite)),
                "diff_median_mm": float(np.nanmedian(diff_grid)) if np.any(finite) else np.nan,
                "diff_mean_mm": float(np.nanmean(diff_grid)) if np.any(finite) else np.nan,
                "diff_max_abs_mm": float(np.nanmax(np.abs(diff_grid))) if np.any(finite) else np.nan,
            }
        )
    diff_summary = pd.DataFrame(diff_rows)

    summary = (
        selected_days
        .merge(avamet_summary, on="date", how="left")
        .merge(avamet_cell_summary, on="date", how="left")
        .merge(imerg_summary, on="date", how="left")
        .merge(diff_summary, on="date", how="left")
    )
    summary.to_csv(args.output_csv, index=False)

    provinces, cv_union = load_cv_layers()
    plot_maps(
        selected_days=selected_days,
        avamet_frames=avamet_frames,
        avamet_cell_grids=avamet_cell_grids,
        imerg_grids=imerg_grids,
        provinces=provinces,
        cv_union=cv_union,
        output_png=args.output_png,
        coverage_threshold=float(args.coverage_threshold),
    )

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_png}")
    print("Selected dates: " + ", ".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in summary["date"]))


if __name__ == "__main__":
    main()

