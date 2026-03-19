from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MANUAL_OVERRIDE_KEYS = {
    ("c03m070e01", 180, "2024-10-31T15:20:00"),
    ("c03m070e01", 360, "2024-10-31T23:20:00"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicit outlier policy from the top-event audit and rerun "
            "event-based IMERG skill with low-plausibility events removed."
        )
    )
    parser.add_argument(
        "--top-audit-csv",
        type=Path,
        default=Path("results/avamet_cv_top10_event_audit.csv"),
        help="Top-event audit CSV (neighbor rows repeated per event).",
    )
    parser.add_argument(
        "--common-events",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common.parquet"),
        help="Common-period AVAMET-IMERG event parquet.",
    )
    parser.add_argument(
        "--output-policy-csv",
        type=Path,
        default=Path("results/avamet_top_event_policy_decisions.csv"),
        help="Output CSV with retain/exclude decisions for all audited top events.",
    )
    parser.add_argument(
        "--output-excluded-csv",
        type=Path,
        default=Path("results/avamet_low_plausibility_events.csv"),
        help="Output CSV with the excluded low-plausibility events.",
    )
    parser.add_argument(
        "--output-policy-md",
        type=Path,
        default=Path("results/avamet_outlier_policy.md"),
        help="Output markdown file documenting the outlier policy.",
    )
    parser.add_argument(
        "--output-filtered-common",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_common_excluding_low_plausibility.parquet"),
        help="Output parquet after excluding low-plausibility events.",
    )
    parser.add_argument(
        "--output-sensitivity-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_outlier_sensitivity.csv"),
        help="Output CSV comparing baseline and filtered event-based skill.",
    )
    parser.add_argument(
        "--output-sensitivity-json",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_outlier_sensitivity.json"),
        help="Output JSON summary for the outlier sensitivity rerun.",
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
    ratio = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den > 0.0
    ratio.loc[valid] = num.loc[valid] / den.loc[valid]
    return ratio


def summarize_group(group: pd.DataFrame, label: str, scenario: str) -> dict[str, object]:
    count = int(group.shape[0])
    if count == 0:
        return {
            "scenario": scenario,
            "scope": label,
            "event_count": 0,
            "pod_point": None,
            "pod_3x3": None,
            "median_bias_mm_point": None,
            "median_bias_mm_3x3": None,
            "mae_mm_point": None,
            "mae_mm_3x3": None,
            "rmse_mm_point": None,
            "rmse_mm_3x3": None,
            "median_ratio_point": None,
            "median_ratio_3x3": None,
            "zero_response_fraction_point": None,
            "zero_response_fraction_3x3": None,
        }

    avamet = pd.to_numeric(group["event_accum_mm"], errors="coerce").astype(float)
    imerg_point = pd.to_numeric(group["imerg_event_accum_mm"], errors="coerce").astype(float)
    imerg_3x3 = pd.to_numeric(group["imerg_3x3_event_accum_max_mm"], errors="coerce").astype(float)

    hits_point = group["imerg_any_positive"].fillna(False).astype(bool)
    hits_3x3 = group["imerg_3x3_any_positive"].fillna(False).astype(bool)
    ratio_point = safe_ratio(imerg_point, avamet)
    ratio_3x3 = safe_ratio(imerg_3x3, avamet)
    bias_point = imerg_point - avamet
    bias_3x3 = imerg_3x3 - avamet

    return {
        "scenario": scenario,
        "scope": label,
        "event_count": count,
        "pod_point": scalar_or_none(hits_point.mean()),
        "pod_3x3": scalar_or_none(hits_3x3.mean()),
        "median_bias_mm_point": scalar_or_none(bias_point.median()),
        "median_bias_mm_3x3": scalar_or_none(bias_3x3.median()),
        "mae_mm_point": scalar_or_none((bias_point.abs()).mean()),
        "mae_mm_3x3": scalar_or_none((bias_3x3.abs()).mean()),
        "rmse_mm_point": scalar_or_none(np.sqrt(np.mean(np.square(bias_point)))),
        "rmse_mm_3x3": scalar_or_none(np.sqrt(np.mean(np.square(bias_3x3)))),
        "median_ratio_point": scalar_or_none(ratio_point.median()),
        "median_ratio_3x3": scalar_or_none(ratio_3x3.median()),
        "zero_response_fraction_point": scalar_or_none((imerg_point <= 0.0).mean()),
        "zero_response_fraction_3x3": scalar_or_none((imerg_3x3 <= 0.0).mean()),
    }


def aggregate_top_audit(top_audit_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(top_audit_csv)
    group_cols = ["window_min", "rank_in_scale", "station_id", "timestamp_utc"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            event_start_utc=("event_start_utc", "first"),
            event_accum_mm=("event_accum_mm", "first"),
            event_rate_mmh=("event_rate_mmh", "first"),
            threshold_mm=("threshold_mm", "first"),
            threshold_quantile=("threshold_quantile", "first"),
            coverage_ratio=("coverage_ratio", "first"),
            fail_count_window=("fail_count_window", "first"),
            suspect_count_window=("suspect_count_window", "first"),
            dominant_cadence_min=("dominant_cadence_min", "first"),
            cluster_size_windows=("cluster_size_windows", "first"),
            cluster_span_min=("cluster_span_min", "first"),
            neighbor_support=("neighbor_support", "first"),
            neighbor_supporting_count=("neighbor_supporting_count", "first"),
            neighbor_candidate_count=("neighbor_candidate_count", "first"),
            nearest_support_distance_km=("nearest_support_distance_km", "first"),
            max_neighbor_accum_mm=("neighbor_accum_mm", "max"),
            max_neighbor_rate_mmh=("neighbor_rate_mmh", "max"),
            min_neighbor_distance_km=("neighbor_distance_km", "min"),
        )
        .copy()
    )

    agg["timestamp_utc"] = pd.to_datetime(agg["timestamp_utc"], errors="coerce", utc=False)
    agg["event_start_utc"] = pd.to_datetime(agg["event_start_utc"], errors="coerce", utc=False)
    agg["max_neighbor_accum_mm"] = pd.to_numeric(agg["max_neighbor_accum_mm"], errors="coerce").fillna(0.0)
    agg["max_neighbor_rate_mmh"] = pd.to_numeric(agg["max_neighbor_rate_mmh"], errors="coerce").fillna(0.0)
    agg["neighbor_max_ratio"] = agg["max_neighbor_accum_mm"] / agg["event_accum_mm"].replace(0.0, np.nan)
    return agg


def select_persistent_zero_support_stations(audit_summary: pd.DataFrame) -> set[str]:
    grouped = (
        audit_summary.groupby("station_id")
        .agg(
            audited_event_count=("station_id", "size"),
            all_no_support=("neighbor_support", lambda s: bool((~s.fillna(False)).all())),
            max_neighbor_any_mm=("max_neighbor_accum_mm", "max"),
        )
        .reset_index()
    )

    flagged = grouped.loc[
        (grouped["audited_event_count"] >= 2)
        & grouped["all_no_support"].fillna(False)
        & (pd.to_numeric(grouped["max_neighbor_any_mm"], errors="coerce").fillna(0.0) <= 0.0)
    ]
    return set(flagged["station_id"].astype(str))


def apply_policy(audit_summary: pd.DataFrame) -> pd.DataFrame:
    out = audit_summary.copy()
    persistent_zero_support_stations = select_persistent_zero_support_stations(out)

    out["station_id"] = out["station_id"].astype(str)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    event_keys = set(
        zip(
            out["station_id"].astype(str),
            out["window_min"].astype(int),
            out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
    )

    policy_bucket: list[str] = []
    decision: list[str] = []
    decision_reason: list[str] = []
    evidence_summary: list[str] = []

    for row in out.itertuples(index=False):
        key = (str(row.station_id), int(row.window_min), pd.Timestamp(row.timestamp_utc).strftime("%Y-%m-%dT%H:%M:%S"))

        if str(row.station_id) in persistent_zero_support_stations:
            bucket = "persistent_zero_support_station"
            keep = False
            reason = (
                "Repeated top-tail anomalies at the same station, all with no AVAMET network support "
                "and zero accumulation among the three nearest audited neighbors."
            )
        elif key in MANUAL_OVERRIDE_KEYS:
            bucket = "manual_global_max_override"
            keep = False
            reason = (
                "Global maximum case kept in the audit only: nearest audited neighbors remain near-zero "
                "(<= 0.2 mm) despite very large station totals, so it is excluded from the main-skill sensitivity set."
            )
        else:
            bucket = "retain_main"
            keep = True
            reason = "Retained in the main analysis; no explicit low-plausibility exclusion rule triggered."

        policy_bucket.append(bucket)
        decision.append("retain" if keep else "exclude")
        decision_reason.append(reason)
        evidence_summary.append(
            (
                f"suspect={int(row.suspect_count_window)}; "
                f"support={bool(row.neighbor_support)} ({int(row.neighbor_supporting_count)}/{int(row.neighbor_candidate_count)}); "
                f"nearest3_max={float(row.max_neighbor_accum_mm):.1f} mm; "
                f"neighbor_max_ratio={float(row.neighbor_max_ratio):.6f}"
            )
        )

    out["policy_bucket"] = policy_bucket
    out["decision"] = decision
    out["decision_reason"] = decision_reason
    out["evidence_summary"] = evidence_summary
    out["decision_rank"] = np.where(out["decision"] == "exclude", 0, 1)
    out = out.sort_values(["decision_rank", "window_min", "rank_in_scale", "station_id"]).drop(columns=["decision_rank"])
    return out


def build_skill_rows(df: pd.DataFrame, scenario: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [summarize_group(df, "all", scenario)]
    for window_min, group in df.groupby("window_min", sort=True):
        rows.append(summarize_group(group, str(int(window_min)), scenario))
    return rows


def make_event_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_id"] = out["station_id"].astype(str)
    out["window_min"] = pd.to_numeric(out["window_min"], errors="coerce").astype(int)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=False)
    out["event_key"] = (
        out["station_id"].astype(str)
        + "|"
        + out["window_min"].astype(str)
        + "|"
        + out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    return out


def write_policy_markdown(
    path: Path,
    policy_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    baseline_count: int,
    filtered_count: int,
) -> None:
    excluded_count = int(excluded_df.shape[0])
    excluded_fraction = (excluded_count / baseline_count) if baseline_count else None
    bucket_counts = excluded_df["policy_bucket"].value_counts().to_dict()
    station_counts = excluded_df["station_id"].astype(str).value_counts().to_dict()

    lines: list[str] = [
        "# AVAMET Outlier Policy",
        "",
        "## Scope",
        "",
        "This policy applies only to the audited top-tail events used for plausibility control.",
        "The full AVAMET catalog is preserved; the exclusion list is used only for a documented sensitivity rerun of the main event-based IMERG skill.",
        "",
        "## Decision Rules",
        "",
        "1. `persistent_zero_support_station`",
        "Events are excluded when they belong to a station that appears at least twice in the top-10 audit,",
        "and every audited top-tail event from that station shows `neighbor_support = False` with `0.0 mm` among the three nearest audited neighbors.",
        "",
        "2. `manual_global_max_override`",
        "The two dominant maxima at `c03m070e01` on `2024-10-31` (`180 min` and `360 min`) are excluded from the main-skill sensitivity set.",
        "They remain in the audit archive, but the nearest audited neighbors stay near-zero (`<= 0.2 mm`) while the station reports the catalog maxima.",
        "",
        "## Counts",
        "",
        f"- Audited top-tail events reviewed: `{int(policy_df.shape[0])}`",
        f"- Excluded low-plausibility events: `{excluded_count}`",
        f"- Excluded fraction of common-period event table: `{excluded_fraction:.3%}`" if excluded_fraction is not None else "- Excluded fraction of common-period event table: `n/a`",
        f"- Event count after exclusion (common-period sensitivity set): `{filtered_count}`",
        "",
        "### Excluded By Bucket",
        "",
    ]

    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: `{int(count)}`")

    lines.extend(
        [
            "",
            "### Excluded By Station",
            "",
        ]
    )
    for station_id, count in station_counts.items():
        lines.append(f"- `{station_id}`: `{int(count)}`")

    lines.extend(
        [
            "",
            "## Excluded Events",
            "",
            "| Station | Window | Timestamp UTC | Event (mm) | Suspect | Nearest-3 max (mm) | Policy bucket |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )

    display_cols = [
        "station_id",
        "window_min",
        "timestamp_utc",
        "event_accum_mm",
        "suspect_count_window",
        "max_neighbor_accum_mm",
        "policy_bucket",
    ]
    for row in excluded_df[display_cols].itertuples(index=False):
        ts = pd.Timestamp(row.timestamp_utc).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"| `{row.station_id}` | {int(row.window_min)} | `{ts}` | {float(row.event_accum_mm):.1f} | "
            f"{int(row.suspect_count_window)} | {float(row.max_neighbor_accum_mm):.1f} | `{row.policy_bucket}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    for output_path in [
        args.output_policy_csv,
        args.output_excluded_csv,
        args.output_policy_md,
        args.output_filtered_common,
        args.output_sensitivity_csv,
        args.output_sensitivity_json,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    audit_summary = aggregate_top_audit(args.top_audit_csv)
    policy_df = apply_policy(audit_summary)
    excluded_df = policy_df.loc[policy_df["decision"] == "exclude"].copy()

    common = pd.read_parquet(args.common_events).copy()
    common = make_event_key_frame(common)

    excluded_keys = (
        excluded_df["station_id"].astype(str)
        + "|"
        + excluded_df["window_min"].astype(int).astype(str)
        + "|"
        + pd.to_datetime(excluded_df["timestamp_utc"], errors="coerce", utc=False).dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    excluded_key_set = set(excluded_keys.tolist())

    common["excluded_low_plausibility"] = common["event_key"].isin(excluded_key_set)
    filtered = common.loc[~common["excluded_low_plausibility"]].copy()

    policy_df.to_csv(args.output_policy_csv, index=False)
    excluded_df.to_csv(args.output_excluded_csv, index=False)
    filtered.drop(columns=["event_key"]).to_parquet(args.output_filtered_common, index=False)

    baseline_rows = build_skill_rows(common, "baseline")
    filtered_rows = build_skill_rows(filtered, "exclude_low_plausibility")

    combined_df = pd.DataFrame(baseline_rows + filtered_rows)
    pivot = combined_df.pivot(index="scope", columns="scenario")

    delta_rows: list[dict[str, object]] = []
    if {"baseline", "exclude_low_plausibility"}.issubset(set(combined_df["scenario"].unique())):
        for scope in pivot.index:
            row: dict[str, object] = {"scenario": "delta_filtered_minus_baseline", "scope": scope}
            for metric in [
                "event_count",
                "pod_point",
                "pod_3x3",
                "median_bias_mm_point",
                "median_bias_mm_3x3",
                "mae_mm_point",
                "mae_mm_3x3",
                "rmse_mm_point",
                "rmse_mm_3x3",
                "median_ratio_point",
                "median_ratio_3x3",
                "zero_response_fraction_point",
                "zero_response_fraction_3x3",
            ]:
                baseline_value = pivot.loc[scope, (metric, "baseline")]
                filtered_value = pivot.loc[scope, (metric, "exclude_low_plausibility")]
                if pd.isna(baseline_value) or pd.isna(filtered_value):
                    row[metric] = None
                else:
                    row[metric] = scalar_or_none(filtered_value - baseline_value)
            delta_rows.append(row)

    sensitivity_df = pd.DataFrame(baseline_rows + filtered_rows + delta_rows)
    sensitivity_df.to_csv(args.output_sensitivity_csv, index=False)

    write_policy_markdown(
        path=args.output_policy_md,
        policy_df=policy_df,
        excluded_df=excluded_df,
        baseline_count=int(common.shape[0]),
        filtered_count=int(filtered.shape[0]),
    )

    json_summary = {
        "policy_summary": {
            "audited_top_event_count": int(policy_df.shape[0]),
            "excluded_low_plausibility_count": int(excluded_df.shape[0]),
            "excluded_low_plausibility_keys_in_common": int(common["excluded_low_plausibility"].sum()),
            "filtered_common_event_count": int(filtered.shape[0]),
            "policy_buckets": excluded_df["policy_bucket"].value_counts().to_dict(),
        },
        "excluded_events": excluded_df[
            [
                "station_id",
                "window_min",
                "timestamp_utc",
                "event_accum_mm",
                "suspect_count_window",
                "neighbor_support",
                "neighbor_supporting_count",
                "max_neighbor_accum_mm",
                "neighbor_max_ratio",
                "policy_bucket",
                "decision_reason",
            ]
        ]
        .assign(timestamp_utc=lambda df: pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=False).dt.strftime("%Y-%m-%dT%H:%M:%S"))
        .to_dict(orient="records"),
        "sensitivity_rows": sensitivity_df.to_dict(orient="records"),
    }
    args.output_sensitivity_json.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output_policy_csv}")
    print(f"Wrote: {args.output_excluded_csv}")
    print(f"Wrote: {args.output_policy_md}")
    print(f"Wrote: {args.output_filtered_common}")
    print(f"Wrote: {args.output_sensitivity_csv}")
    print(f"Wrote: {args.output_sensitivity_json}")
    print(f"Excluded low-plausibility events in common period: {int(common['excluded_low_plausibility'].sum())}")


if __name__ == "__main__":
    main()

