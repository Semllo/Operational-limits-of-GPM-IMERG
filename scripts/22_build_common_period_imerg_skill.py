from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter the AVAMET-IMERG event table to the common period and summarize detection/intensity skill."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg.parquet"),
        help="Input AVAMET-IMERG event parquet.",
    )
    parser.add_argument(
        "--output-common",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common.parquet"),
        help="Output parquet restricted to windows with complete IMERG coverage.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_by_scale.csv"),
        help="Output CSV with skill metrics by scale plus an overall row.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_summary.json"),
        help="Output JSON summary.",
    )
    return parser


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    ratio = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den > 0.0
    ratio.loc[valid] = num.loc[valid] / den.loc[valid]
    return ratio


def scalar_or_none(value: float | int | np.floating | np.integer | None) -> float | int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def summarize_group(group: pd.DataFrame, label: str) -> dict[str, object]:
    count = int(group.shape[0])
    if count == 0:
        return {
            "scope": label,
            "event_count": 0,
            "pod_point": None,
            "pod_3x3": None,
            "far_point": None,
            "far_3x3": None,
            "median_bias_mm_point": None,
            "median_bias_mm_3x3": None,
            "mae_mm_point": None,
            "mae_mm_3x3": None,
            "rmse_mm_point": None,
            "rmse_mm_3x3": None,
            "median_ratio_point": None,
            "median_ratio_3x3": None,
            "median_ratio_point_hits": None,
            "median_ratio_3x3_hits": None,
            "mean_ratio_point": None,
            "mean_ratio_3x3": None,
            "underestimate_fraction_point": None,
            "underestimate_fraction_3x3": None,
            "zero_response_fraction_point": None,
            "zero_response_fraction_3x3": None,
            "median_avamet_event_mm": None,
            "median_imerg_event_mm_point": None,
            "median_imerg_event_mm_3x3": None,
        }

    avamet = pd.to_numeric(group["event_accum_mm"], errors="coerce").astype(float)
    imerg_point = pd.to_numeric(group["imerg_event_accum_mm"], errors="coerce").astype(float)
    imerg_3x3 = pd.to_numeric(group["imerg_3x3_event_accum_max_mm"], errors="coerce").astype(float)

    ratio_point = safe_ratio(imerg_point, avamet)
    ratio_3x3 = safe_ratio(imerg_3x3, avamet)
    hits_point = group["imerg_any_positive"].fillna(False).astype(bool)
    hits_3x3 = group["imerg_3x3_any_positive"].fillna(False).astype(bool)

    bias_point = imerg_point - avamet
    bias_3x3 = imerg_3x3 - avamet

    row = {
        "scope": label,
        "event_count": count,
        "pod_point": float(hits_point.mean()),
        "pod_3x3": float(hits_3x3.mean()),
        "far_point": None,
        "far_3x3": None,
        "median_bias_mm_point": scalar_or_none(bias_point.median()),
        "median_bias_mm_3x3": scalar_or_none(bias_3x3.median()),
        "mae_mm_point": scalar_or_none((bias_point.abs()).mean()),
        "mae_mm_3x3": scalar_or_none((bias_3x3.abs()).mean()),
        "rmse_mm_point": scalar_or_none(np.sqrt(np.mean(np.square(bias_point)))),
        "rmse_mm_3x3": scalar_or_none(np.sqrt(np.mean(np.square(bias_3x3)))),
        "median_ratio_point": scalar_or_none(ratio_point.median()),
        "median_ratio_3x3": scalar_or_none(ratio_3x3.median()),
        "median_ratio_point_hits": scalar_or_none(ratio_point.loc[hits_point].median()),
        "median_ratio_3x3_hits": scalar_or_none(ratio_3x3.loc[hits_3x3].median()),
        "mean_ratio_point": scalar_or_none(ratio_point.mean()),
        "mean_ratio_3x3": scalar_or_none(ratio_3x3.mean()),
        "underestimate_fraction_point": scalar_or_none((imerg_point < avamet).mean()),
        "underestimate_fraction_3x3": scalar_or_none((imerg_3x3 < avamet).mean()),
        "zero_response_fraction_point": scalar_or_none((imerg_point <= 0.0).mean()),
        "zero_response_fraction_3x3": scalar_or_none((imerg_3x3 <= 0.0).mean()),
        "median_avamet_event_mm": scalar_or_none(avamet.median()),
        "median_imerg_event_mm_point": scalar_or_none(imerg_point.median()),
        "median_imerg_event_mm_3x3": scalar_or_none(imerg_3x3.median()),
    }
    return row


def main() -> None:
    args = build_parser().parse_args()
    args.output_common.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=False)
    df["imerg_window_start_utc"] = pd.to_datetime(df["imerg_window_start_utc"], errors="coerce", utc=False)

    common = df.loc[df["imerg_complete_window"].fillna(False).astype(bool)].copy()
    common = common.sort_values(["timestamp_utc", "window_min", "station_id"]).reset_index(drop=True)
    common.to_parquet(args.output_common, index=False)

    rows: list[dict[str, object]] = [summarize_group(common, "all")]
    for window_min, group in common.groupby("window_min", sort=True):
        rows.append(summarize_group(group, str(int(window_min))))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_csv, index=False)

    common_period = {
        "filter_rule": "imerg_complete_window == True",
        "event_count_input": int(df.shape[0]),
        "event_count_common": int(common.shape[0]),
        "excluded_count": int(df.shape[0] - common.shape[0]),
        "common_fraction": float(common.shape[0] / df.shape[0]) if df.shape[0] else None,
        "event_period_start_common": None if common.empty else common["timestamp_utc"].min().isoformat(),
        "event_period_end_common": None if common.empty else common["timestamp_utc"].max().isoformat(),
        "far_note": (
            "FAR is not estimable from this event-only product. "
            "It requires a full-window comparison including IMERG-positive / AVAMET-negative windows."
        ),
    }

    json_summary = {
        "common_period": common_period,
        "skill_rows": rows,
    }
    args.output_json.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_common}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Common-period events: {common.shape[0]} / {df.shape[0]}")


if __name__ == "__main__":
    main()

