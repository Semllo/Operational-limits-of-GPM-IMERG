from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a numeric summary table for the currently selected DANA case-study days."
    )
    parser.add_argument(
        "--candidate-days-csv",
        type=Path,
        default=Path("results/dana_candidate_days.csv"),
        help="CSV with the selected candidate DANA days.",
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common_excluding_low_plausibility.parquet"),
        help="Filtered common-period event parquet.",
    )
    parser.add_argument(
        "--amplitude-input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement.parquet"),
        help="Amplitude-displacement event parquet.",
    )
    parser.add_argument(
        "--station-strata-csv",
        type=Path,
        default=Path("results/avamet_station_inventory_cv_strata.csv"),
        help="Station inventory with stratification fields.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/dana_episode_summary_table.csv"),
        help="Output CSV table (one row per selected DANA day).",
    )
    parser.add_argument(
        "--output-window-csv",
        type=Path,
        default=Path("results/dana_episode_window_table.csv"),
        help="Output CSV table by day and scale.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/dana_episode_summary_table.md"),
        help="Output markdown table.",
    )
    return parser


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value)
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def build_event_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_id"] = out["station_id"].astype(str)
    out["window_min"] = pd.to_numeric(out["window_min"], errors="coerce").astype(int)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    out["event_key"] = (
        out["station_id"]
        + "|"
        + out["window_min"].astype(str)
        + "|"
        + out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    return out


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_window_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    candidate_days = pd.read_csv(args.candidate_days_csv).copy()
    candidate_days["date"] = pd.to_datetime(candidate_days["date"], errors="coerce", utc=False).dt.normalize()

    events = pd.read_parquet(args.events_input).copy()
    events = build_event_key(events)
    events["date"] = events["timestamp_utc"].dt.normalize()

    amplitude = pd.read_parquet(
        args.amplitude_input,
        columns=[
            "station_id",
            "window_min",
            "timestamp_utc",
            "imerg_best_3x3_distance_km",
            "imerg_point_recovery_ratio",
            "imerg_3x3_recovery_ratio",
            "imerg_best_3x3_same_cell",
        ],
    ).copy()
    amplitude = build_event_key(amplitude)
    amplitude = amplitude.drop_duplicates(subset=["event_key"], keep="first")

    stations = pd.read_csv(args.station_strata_csv).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["province_name"] = stations["province_name"].map(normalize_text)

    merged = events.merge(
        amplitude[
            [
                "event_key",
                "imerg_best_3x3_distance_km",
                "imerg_point_recovery_ratio",
                "imerg_3x3_recovery_ratio",
                "imerg_best_3x3_same_cell",
            ]
        ],
        on="event_key",
        how="left",
    )
    merged = merged.merge(
        stations[["station_id", "province_name", "coast_group", "altitude_group"]],
        on="station_id",
        how="left",
    )

    all_window_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for candidate in candidate_days.itertuples(index=False):
        date_value = pd.Timestamp(candidate.date)
        date_events = merged.loc[merged["date"] == date_value].copy()

        for window_min, group in date_events.groupby("window_min", sort=True):
            ratios_point = pd.to_numeric(group["imerg_event_accum_mm"], errors="coerce") / pd.to_numeric(group["event_accum_mm"], errors="coerce").replace(0.0, pd.NA)
            ratios_3x3 = pd.to_numeric(group["imerg_3x3_event_accum_max_mm"], errors="coerce") / pd.to_numeric(group["event_accum_mm"], errors="coerce").replace(0.0, pd.NA)
            all_window_rows.append(
                {
                    "date": date_value.strftime("%Y-%m-%d"),
                    "window_min": int(window_min),
                    "event_count": int(group.shape[0]),
                    "station_count": int(group["station_id"].astype(str).nunique()),
                    "avamet_sum_event_mm": float(pd.to_numeric(group["event_accum_mm"], errors="coerce").sum()),
                    "avamet_max_event_mm": float(pd.to_numeric(group["event_accum_mm"], errors="coerce").max()),
                    "imerg_sum_event_mm_point": float(pd.to_numeric(group["imerg_event_accum_mm"], errors="coerce").sum()),
                    "imerg_sum_event_mm_3x3": float(pd.to_numeric(group["imerg_3x3_event_accum_max_mm"], errors="coerce").sum()),
                    "median_point_ratio": float(pd.to_numeric(ratios_point, errors="coerce").median()),
                    "median_3x3_ratio": float(pd.to_numeric(ratios_3x3, errors="coerce").median()),
                    "median_displacement_km": float(pd.to_numeric(group["imerg_best_3x3_distance_km"], errors="coerce").median()),
                    "fraction_same_cell": float(group["imerg_best_3x3_same_cell"].fillna(False).astype(bool).mean()),
                }
            )

        focus = date_events.loc[date_events["window_min"] == 360].copy()
        if focus.empty:
            continue

        max_row = focus.sort_values(["event_accum_mm", "timestamp_utc"], ascending=[False, True]).iloc[0]
        point_ratio = float(max_row["imerg_event_accum_mm"] / max_row["event_accum_mm"]) if max_row["event_accum_mm"] else None
        best_ratio = (
            float(max_row["imerg_3x3_event_accum_max_mm"] / max_row["event_accum_mm"])
            if max_row["event_accum_mm"]
            else None
        )

        summary_rows.append(
            {
                "date": date_value.strftime("%Y-%m-%d"),
                "event_count_360": int(focus.shape[0]),
                "station_count_360": int(focus["station_id"].astype(str).nunique()),
                "avamet_sum_event_mm_360": float(pd.to_numeric(focus["event_accum_mm"], errors="coerce").sum()),
                "imerg_sum_event_mm_point_360": float(pd.to_numeric(focus["imerg_event_accum_mm"], errors="coerce").sum()),
                "imerg_sum_event_mm_3x3_360": float(pd.to_numeric(focus["imerg_3x3_event_accum_max_mm"], errors="coerce").sum()),
                "median_point_ratio_360": float(pd.to_numeric(focus["imerg_point_recovery_ratio"], errors="coerce").median()),
                "median_3x3_ratio_360": float(pd.to_numeric(focus["imerg_3x3_recovery_ratio"], errors="coerce").median()),
                "median_displacement_km_360": float(pd.to_numeric(focus["imerg_best_3x3_distance_km"], errors="coerce").median()),
                "max_station_id": str(max_row["station_id"]),
                "max_station_province": normalize_text(max_row.get("province_name")),
                "max_station_coast_group": normalize_text(max_row.get("coast_group")),
                "max_event_timestamp_utc": pd.Timestamp(max_row["timestamp_utc"]).strftime("%Y-%m-%d %H:%M:%S"),
                "max_event_mm_360": float(max_row["event_accum_mm"]),
                "max_imerg_point_mm_360": float(max_row["imerg_event_accum_mm"]),
                "max_imerg_3x3_mm_360": float(max_row["imerg_3x3_event_accum_max_mm"]),
                "max_event_point_ratio": point_ratio,
                "max_event_3x3_ratio": best_ratio,
                "max_event_displacement_km": float(max_row["imerg_best_3x3_distance_km"]),
                "max_event_same_cell": bool(max_row["imerg_best_3x3_same_cell"]),
                "daily_station_sum_mm": float(candidate.avamet_station_sum_mm) if hasattr(candidate, "avamet_station_sum_mm") else None,
                "daily_station_max_mm": float(candidate.avamet_station_max_mm) if hasattr(candidate, "avamet_station_max_mm") else None,
                "daily_imerg_grid_max_mm": float(candidate.imerg_grid_max_mm) if hasattr(candidate, "imerg_grid_max_mm") else None,
                "daily_diff_median_mm": float(candidate.diff_median_mm) if hasattr(candidate, "diff_median_mm") else None,
                "daily_diff_mean_mm": float(candidate.diff_mean_mm) if hasattr(candidate, "diff_mean_mm") else None,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("date").reset_index(drop=True)
    window_df = pd.DataFrame(all_window_rows).sort_values(["date", "window_min"]).reset_index(drop=True)

    summary_df.to_csv(args.output_csv, index=False)
    window_df.to_csv(args.output_window_csv, index=False)

    lines = [
        "# DANA Episode Summary Table",
        "",
        "Current case-study days come from `results/dana_candidate_days.csv` and match the paper-style comparison maps.",
        "",
        "| Date | 6 h events | AVAMET sum (mm) | IMERG sum (point) | IMERG sum (3x3) | Median ratio (point) | Median ratio (3x3) | Max station | Max AVAMET (mm) | Max IMERG 3x3 (mm) | Max displacement (km) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| `{row.date}` | {int(row.event_count_360)} | {float(row.avamet_sum_event_mm_360):.1f} | "
            f"{float(row.imerg_sum_event_mm_point_360):.1f} | {float(row.imerg_sum_event_mm_3x3_360):.1f} | "
            f"{float(row.median_point_ratio_360):.2f} | {float(row.median_3x3_ratio_360):.2f} | "
            f"`{row.max_station_id}` | {float(row.max_event_mm_360):.1f} | {float(row.max_imerg_3x3_mm_360):.1f} | "
            f"{float(row.max_event_displacement_km):.1f} |"
        )

    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_window_csv}")
    print(f"Wrote: {args.output_md}")
    print(f"Episodes summarized: {summary_df.shape[0]}")


if __name__ == "__main__":
    main()

