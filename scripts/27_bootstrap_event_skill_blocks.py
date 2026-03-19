from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "pod_point",
    "pod_3x3",
    "median_ratio_point",
    "median_ratio_3x3",
    "median_bias_mm_point",
    "median_bias_mm_3x3",
    "zero_response_fraction_point",
    "zero_response_fraction_3x3",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap event-based IMERG skill using daily blocks from the common-period event table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common.parquet"),
        help="Input common-period event parquet.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_bootstrap.csv"),
        help="Output CSV with bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_bootstrap.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of block bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den > 0.0
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {metric: None for metric in METRICS} | {"event_count": 0}

    avamet = pd.to_numeric(df["event_accum_mm"], errors="coerce").astype(float)
    point = pd.to_numeric(df["imerg_event_accum_mm"], errors="coerce").astype(float)
    grid = pd.to_numeric(df["imerg_3x3_event_accum_max_mm"], errors="coerce").astype(float)
    hits_point = df["imerg_any_positive"].fillna(False).astype(bool)
    hits_grid = df["imerg_3x3_any_positive"].fillna(False).astype(bool)

    ratio_point = safe_ratio(point, avamet)
    ratio_grid = safe_ratio(grid, avamet)
    bias_point = point - avamet
    bias_grid = grid - avamet

    return {
        "event_count": int(df.shape[0]),
        "pod_point": float(hits_point.mean()),
        "pod_3x3": float(hits_grid.mean()),
        "median_ratio_point": float(ratio_point.median()),
        "median_ratio_3x3": float(ratio_grid.median()),
        "median_bias_mm_point": float(bias_point.median()),
        "median_bias_mm_3x3": float(bias_grid.median()),
        "zero_response_fraction_point": float((point <= 0.0).mean()),
        "zero_response_fraction_3x3": float((grid <= 0.0).mean()),
    }


def bootstrap_group(df: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, object]:
    base = compute_metrics(df)
    if df.empty:
        return {"point_estimate": base, "confidence_intervals": {metric: {"low": None, "high": None} for metric in METRICS}}

    work = df.copy()
    work["block_date"] = pd.to_datetime(work["timestamp_utc"], errors="coerce", utc=False).dt.normalize()
    block_frames = [group.copy() for _date, group in work.groupby("block_date", sort=True)]
    n_blocks = len(block_frames)
    if n_blocks == 0:
        return {"point_estimate": base, "confidence_intervals": {metric: {"low": None, "high": None} for metric in METRICS}}

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for _ in range(int(n_bootstrap)):
        choice = rng.integers(0, n_blocks, size=n_blocks)
        sample_df = pd.concat([block_frames[idx] for idx in choice], ignore_index=True)
        metrics = compute_metrics(sample_df)
        for metric in METRICS:
            value = metrics[metric]
            if value is not None and not pd.isna(value):
                samples[metric].append(float(value))

    intervals: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = np.asarray(samples[metric], dtype=np.float64)
        if values.size == 0:
            intervals[metric] = {"low": None, "high": None}
        else:
            intervals[metric] = {
                "low": float(np.nanquantile(values, 0.025)),
                "high": float(np.nanquantile(values, 0.975)),
            }

    return {"point_estimate": base, "confidence_intervals": intervals, "block_count": n_blocks}


def rows_from_bootstrap(scope: str, result: dict[str, object]) -> dict[str, object]:
    base = result["point_estimate"]
    intervals = result["confidence_intervals"]
    row: dict[str, object] = {"scope": scope, "event_count": base["event_count"]}
    for metric in METRICS:
        row[metric] = base[metric]
        row[f"{metric}_ci_low"] = intervals[metric]["low"]
        row[f"{metric}_ci_high"] = intervals[metric]["high"]
    row["block_count"] = result.get("block_count")
    return row


def main() -> None:
    args = build_parser().parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input).copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=False)
    df = df.sort_values(["timestamp_utc", "window_min", "station_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    json_rows: list[dict[str, object]] = []

    overall = bootstrap_group(df, int(args.n_bootstrap), int(args.seed))
    rows.append(rows_from_bootstrap("all", overall))
    json_rows.append({"scope": "all", **overall})

    for window_min, group in df.groupby("window_min", sort=True):
        result = bootstrap_group(group, int(args.n_bootstrap), int(args.seed) + int(window_min))
        rows.append(rows_from_bootstrap(str(int(window_min)), result))
        json_rows.append({"scope": str(int(window_min)), **result})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_csv, index=False)

    json_summary = {
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "block_definition": "UTC day blocks from timestamp_utc",
        "rows": json_rows,
    }
    args.output_json.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    print(f"Rows: {summary_df.shape[0]}")


if __name__ == "__main__":
    main()

