from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


SUPPORTED_DELTAS = (4, 5, 6, 10, 15, 20, 30)
CORE_DELTAS = (5, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a conservative, traceable AVAMET QC parquet for sub-hourly precipitation extremes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/avamet_cv.parquet"),
        help="Input AVAMET parquet (defaults to the Valencian Community subset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/avamet_cv_qc.parquet"),
        help="Output QC parquet.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/avamet_cv_qc_summary.json"),
        help="QC summary JSON.",
    )
    parser.add_argument(
        "--station-summary-csv",
        type=Path,
        default=Path("results/avamet_cv_qc_station_summary.csv"),
        help="Per-station QC summary CSV.",
    )
    parser.add_argument(
        "--precsum-diff-tol",
        type=float,
        default=0.2,
        help="Absolute tolerance (mm) for prec tot_sum first-difference vs prec tot.",
    )
    parser.add_argument(
        "--suspect-rate-threshold",
        type=float,
        default=200.0,
        help="Equivalent intensity threshold (mm/h) to flag a precipitation candidate as suspect.",
    )
    parser.add_argument(
        "--hard-rate-threshold",
        type=float,
        default=1000.0,
        help="Equivalent intensity threshold (mm/h) to treat a precipitation value as hard-fail.",
    )
    parser.add_argument(
        "--gap-threshold-min",
        type=int,
        default=60,
        help="Gap threshold (minutes) flagged as a large temporal gap.",
    )
    return parser


def _numeric_extract(json_key: str, alias: str) -> str:
    escaped = json_key.replace('"', '\\"')
    return (
        f"TRY_CAST(REPLACE(NULLIF(TRIM(json_extract_string(payload_json, '$.\"{escaped}\"')), 'None'), ',', '.') "
        f"AS DOUBLE) AS {alias}"
    )


def _as_sql_list(values: tuple[int, ...]) -> str:
    return ", ".join(str(v) for v in values)


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.station_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    input_path = args.input.resolve().as_posix()
    output_path = args.output.resolve().as_posix()
    station_summary_path = args.station_summary_csv.resolve().as_posix()

    supported_sql = _as_sql_list(SUPPORTED_DELTAS)
    core_sql = _as_sql_list(CORE_DELTAS)

    raw_sql = f"""
        CREATE OR REPLACE TABLE raw_window AS
        SELECT
          ROW_NUMBER() OVER () AS ingest_order,
          station_id,
          TRY_CAST(obs_time AS TIMESTAMP) AS timestamp_utc,
          TRY_CAST(inserted_at AS TIMESTAMP) AS inserted_ts,
          lat,
          lon,
          alt,
          {_numeric_extract("prec tot", "precip_raw_mm")},
          {_numeric_extract("prec tot_sum", "precip_sum_raw_mm")},
          {_numeric_extract("temp mit_mit", "temp_raw_c")},
          {_numeric_extract("hrel mit_mit", "rh_raw_pct")},
          {_numeric_extract("pres mit_mit", "pressure_raw_hpa")},
          {_numeric_extract("vent vel_mit", "wind_mean_raw")},
          {_numeric_extract("vent vel_max", "wind_gust_raw")},
          {_numeric_extract("vent gra_mit", "wind_dir_raw_deg")}
        FROM read_parquet('{input_path}')
        WHERE station_id IS NOT NULL
          AND TRY_CAST(obs_time AS TIMESTAMP) IS NOT NULL
    """
    con.execute(raw_sql)

    dedup_sql = """
        CREATE OR REPLACE TABLE dedup_window AS
        SELECT * EXCLUDE (row_keep)
        FROM (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY station_id, timestamp_utc
              ORDER BY inserted_ts DESC NULLS LAST, ingest_order DESC
            ) AS row_keep
          FROM raw_window
        )
        WHERE row_keep = 1
    """
    con.execute(dedup_sql)

    qc_sql = f"""
        CREATE OR REPLACE TABLE qc_window AS
        WITH seq AS (
          SELECT
            station_id,
            timestamp_utc,
            inserted_ts,
            lat,
            lon,
            alt,
            precip_raw_mm,
            precip_sum_raw_mm,
            temp_raw_c,
            rh_raw_pct,
            pressure_raw_hpa,
            wind_mean_raw,
            wind_gust_raw,
            wind_dir_raw_deg,
            LAG(timestamp_utc) OVER (PARTITION BY station_id ORDER BY timestamp_utc) AS prev_ts,
            LEAD(timestamp_utc) OVER (PARTITION BY station_id ORDER BY timestamp_utc) AS next_ts,
            LAG(precip_raw_mm) OVER (PARTITION BY station_id ORDER BY timestamp_utc) AS prev_precip_raw_mm,
            LEAD(precip_raw_mm) OVER (PARTITION BY station_id ORDER BY timestamp_utc) AS next_precip_raw_mm,
            LAG(precip_sum_raw_mm) OVER (PARTITION BY station_id ORDER BY timestamp_utc) AS prev_precip_sum_raw_mm
          FROM dedup_window
        ),
        flags AS (
          SELECT
            *,
            DATEDIFF('minute', prev_ts, timestamp_utc) AS delta_prev_min,
            DATEDIFF('minute', timestamp_utc, next_ts) AS delta_next_min,
            CASE
              WHEN prev_precip_sum_raw_mm IS NOT NULL AND precip_sum_raw_mm IS NOT NULL
              THEN precip_sum_raw_mm - prev_precip_sum_raw_mm
            END AS d_precsum_mm,
            CASE
              WHEN precip_raw_mm IS NOT NULL AND DATEDIFF('minute', prev_ts, timestamp_utc) > 0
              THEN precip_raw_mm * 60.0 / DATEDIFF('minute', prev_ts, timestamp_utc)
            END AS precip_rate_mmh_raw
          FROM seq
        )
        SELECT
          station_id,
          timestamp_utc,
          inserted_ts,
          lat,
          lon,
          alt,
          temp_raw_c,
          rh_raw_pct,
          pressure_raw_hpa,
          wind_mean_raw,
          wind_gust_raw,
          wind_dir_raw_deg,
          precip_raw_mm,
          precip_sum_raw_mm,
          prev_ts,
          next_ts,
          delta_prev_min,
          delta_next_min,
          d_precsum_mm,
          precip_rate_mmh_raw,
          CASE WHEN delta_prev_min IN ({core_sql}) THEN TRUE ELSE FALSE END AS is_core_cadence,
          CASE WHEN delta_prev_min IN ({supported_sql}) THEN TRUE ELSE FALSE END AS is_supported_cadence,
          CASE WHEN delta_prev_min IS NOT NULL AND delta_prev_min <= 0 THEN TRUE ELSE FALSE END AS flag_delta_nonpositive,
          CASE WHEN delta_prev_min IS NOT NULL AND delta_prev_min >= {int(args.gap_threshold_min)} THEN TRUE ELSE FALSE END AS flag_gap_large,
          CASE WHEN precip_raw_mm < 0 THEN TRUE ELSE FALSE END AS flag_precip_negative,
          CASE WHEN precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)} THEN TRUE ELSE FALSE END AS flag_precip_suspect_rate,
          CASE WHEN precip_rate_mmh_raw >= {float(args.hard_rate_threshold)} THEN TRUE ELSE FALSE END AS flag_precip_hard_rate,
          CASE
            WHEN precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)}
             AND COALESCE(prev_precip_raw_mm, 0.0) <= 0.1
             AND COALESCE(next_precip_raw_mm, 0.0) <= 0.1
            THEN TRUE ELSE FALSE
          END AS flag_precip_isolated_spike,
          CASE WHEN precip_sum_raw_mm < 0 THEN TRUE ELSE FALSE END AS flag_precsum_negative,
          CASE WHEN precip_sum_raw_mm >= 100000 THEN TRUE ELSE FALSE END AS flag_precsum_absurd,
          CASE WHEN d_precsum_mm < -0.001 THEN TRUE ELSE FALSE END AS flag_precsum_decrease,
          CASE
            WHEN d_precsum_mm IS NOT NULL
             AND precip_raw_mm IS NOT NULL
             AND ABS(d_precsum_mm - precip_raw_mm) > {float(args.precsum_diff_tol)}
            THEN TRUE ELSE FALSE
          END AS flag_precsum_mismatch,
          CASE
            WHEN precip_raw_mm < 0 OR precip_rate_mmh_raw >= {float(args.hard_rate_threshold)}
            THEN NULL
            ELSE precip_raw_mm
          END AS precip_mm_qc,
          CASE
            WHEN precip_raw_mm < 0
              OR precip_rate_mmh_raw >= {float(args.hard_rate_threshold)}
              OR delta_prev_min IS NULL
              OR delta_prev_min <= 0
            THEN NULL
            ELSE precip_raw_mm * 60.0 / delta_prev_min
          END AS precip_rate_mmh_qc,
          CASE
            WHEN precip_sum_raw_mm < 0 OR precip_sum_raw_mm >= 100000
            THEN NULL
            ELSE precip_sum_raw_mm
          END AS precip_sum_mm_qc,
          CASE
            WHEN precip_raw_mm < 0 OR precip_rate_mmh_raw >= {float(args.hard_rate_threshold)}
            THEN 'FAIL'
            WHEN precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)}
              OR (
                precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)}
                AND COALESCE(prev_precip_raw_mm, 0.0) <= 0.1
                AND COALESCE(next_precip_raw_mm, 0.0) <= 0.1
              )
            THEN 'SUSPECT'
            ELSE 'PASS'
          END AS qc_precip,
          CASE
            WHEN precip_sum_raw_mm < 0 OR precip_sum_raw_mm >= 100000
            THEN 'FAIL'
            WHEN (d_precsum_mm < -0.001)
              OR (
                d_precsum_mm IS NOT NULL
                AND precip_raw_mm IS NOT NULL
                AND ABS(d_precsum_mm - precip_raw_mm) > {float(args.precsum_diff_tol)}
              )
            THEN 'SUSPECT'
            ELSE 'PASS'
          END AS qc_precsum,
          CASE
            WHEN precip_raw_mm < 0 OR precip_rate_mmh_raw >= {float(args.hard_rate_threshold)}
            THEN 0.0
            WHEN precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)}
              OR (
                precip_rate_mmh_raw >= {float(args.suspect_rate_threshold)}
                AND COALESCE(prev_precip_raw_mm, 0.0) <= 0.1
                AND COALESCE(next_precip_raw_mm, 0.0) <= 0.1
              )
            THEN 0.5
            ELSE 1.0
          END AS q_precip,
          CASE
            WHEN precip_sum_raw_mm < 0 OR precip_sum_raw_mm >= 100000
            THEN 0.0
            WHEN (d_precsum_mm < -0.001)
              OR (
                d_precsum_mm IS NOT NULL
                AND precip_raw_mm IS NOT NULL
                AND ABS(d_precsum_mm - precip_raw_mm) > {float(args.precsum_diff_tol)}
              )
            THEN 0.5
            ELSE 1.0
          END AS q_precsum,
          CASE
            WHEN delta_prev_min IN ({supported_sql})
             AND delta_prev_min > 0
             AND precip_raw_mm >= 0
             AND NOT (precip_rate_mmh_raw >= {float(args.hard_rate_threshold)})
            THEN TRUE
            ELSE FALSE
          END AS usable_interval_extremes
        FROM flags
    """
    con.execute(qc_sql)

    con.execute(
        f"""
        COPY (
          SELECT *
          FROM qc_window
          ORDER BY station_id, timestamp_utc
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    con.execute(
        f"""
        COPY (
          SELECT
            station_id,
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (WHERE qc_precip = 'FAIL') AS n_precip_fail,
            COUNT(*) FILTER (WHERE qc_precip = 'SUSPECT') AS n_precip_suspect,
            COUNT(*) FILTER (WHERE qc_precsum = 'FAIL') AS n_precsum_fail,
            COUNT(*) FILTER (WHERE qc_precsum = 'SUSPECT') AS n_precsum_suspect,
            COUNT(*) FILTER (WHERE flag_gap_large) AS n_gap_large,
            COUNT(*) FILTER (WHERE usable_interval_extremes) AS n_usable_interval_extremes,
            QUANTILE_CONT(delta_prev_min, 0.50) FILTER (WHERE delta_prev_min IS NOT NULL) AS median_delta_min,
            QUANTILE_CONT(delta_prev_min, 0.95) FILTER (WHERE delta_prev_min IS NOT NULL) AS p95_delta_min,
            QUANTILE_CONT(precip_rate_mmh_qc, 0.99) FILTER (WHERE precip_rate_mmh_qc IS NOT NULL) AS p99_precip_rate_mmh_qc,
            MAX(precip_rate_mmh_raw) AS max_precip_rate_mmh_raw,
            MAX(precip_raw_mm) AS max_precip_raw_mm
          FROM qc_window
          GROUP BY station_id
          ORDER BY station_id
        )
        TO '{station_summary_path}'
        (HEADER, DELIMITER ',')
        """
    )

    summary_row = con.execute(
        """
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT station_id) AS n_stations,
          COUNT(*) FILTER (WHERE qc_precip = 'FAIL') AS n_precip_fail,
          COUNT(*) FILTER (WHERE qc_precip = 'SUSPECT') AS n_precip_suspect,
          COUNT(*) FILTER (WHERE qc_precsum = 'FAIL') AS n_precsum_fail,
          COUNT(*) FILTER (WHERE qc_precsum = 'SUSPECT') AS n_precsum_suspect,
          COUNT(*) FILTER (WHERE flag_precip_negative) AS n_precip_negative,
          COUNT(*) FILTER (WHERE flag_precip_hard_rate) AS n_precip_hard_rate,
          COUNT(*) FILTER (WHERE flag_precip_suspect_rate) AS n_precip_suspect_rate,
          COUNT(*) FILTER (WHERE flag_precip_isolated_spike) AS n_precip_isolated_spike,
          COUNT(*) FILTER (WHERE flag_precsum_absurd) AS n_precsum_absurd,
          COUNT(*) FILTER (WHERE flag_precsum_decrease) AS n_precsum_decrease,
          COUNT(*) FILTER (WHERE flag_precsum_mismatch) AS n_precsum_mismatch,
          COUNT(*) FILTER (WHERE flag_gap_large) AS n_gap_large,
          COUNT(*) FILTER (WHERE usable_interval_extremes) AS n_usable_interval_extremes,
          COUNT(*) FILTER (WHERE is_core_cadence) AS n_core_cadence,
          COUNT(*) FILTER (WHERE is_supported_cadence) AS n_supported_cadence,
          COUNT(*) FILTER (WHERE delta_prev_min IS NULL) AS n_first_rows,
          QUANTILE_CONT(delta_prev_min, 0.50) FILTER (WHERE delta_prev_min IS NOT NULL) AS median_delta_min,
          QUANTILE_CONT(precip_rate_mmh_qc, 0.99) FILTER (WHERE precip_rate_mmh_qc IS NOT NULL) AS p99_precip_rate_mmh_qc,
          QUANTILE_CONT(precip_rate_mmh_qc, 0.999) FILTER (WHERE precip_rate_mmh_qc IS NOT NULL) AS p999_precip_rate_mmh_qc,
          MAX(precip_rate_mmh_raw) AS max_precip_rate_mmh_raw,
          MAX(precip_raw_mm) AS max_precip_raw_mm
        FROM qc_window
        """
    ).fetchone()

    duplicate_stats = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM raw_window) AS n_raw_rows,
          (SELECT COUNT(*) FROM dedup_window) AS n_dedup_rows
        """
    ).fetchone()

    worst_station_rows = con.execute(
        """
        SELECT
          station_id,
          n_precip_fail,
          n_precip_suspect,
          n_precsum_fail,
          n_precsum_suspect
        FROM read_csv_auto(?)
        ORDER BY (n_precip_fail + n_precsum_fail) DESC, (n_precip_suspect + n_precsum_suspect) DESC, station_id
        LIMIT 10
        """,
        [station_summary_path],
    ).fetchall()

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "config": {
            "precsum_diff_tol_mm": float(args.precsum_diff_tol),
            "suspect_rate_threshold_mmh": float(args.suspect_rate_threshold),
            "hard_rate_threshold_mmh": float(args.hard_rate_threshold),
            "gap_threshold_min": int(args.gap_threshold_min),
            "supported_deltas_min": list(SUPPORTED_DELTAS),
            "core_deltas_min": list(CORE_DELTAS),
            "qc_semantics": {
                "FAIL": "Negative precip or physically absurd equivalent intensity (for precip), negative/absurd cumulative field (for prec tot_sum).",
                "SUSPECT": "Potential spike candidate or internal inconsistency kept for traceability.",
                "PASS": "No triggered QC rule in this conservative first-pass screen.",
            },
        },
        "counts": {
            "n_raw_rows": int(duplicate_stats[0]),
            "n_dedup_rows": int(duplicate_stats[1]),
            "n_duplicate_rows_removed": int(duplicate_stats[0] - duplicate_stats[1]),
            "n_rows": int(summary_row[0]),
            "n_stations": int(summary_row[1]),
            "n_precip_fail": int(summary_row[2]),
            "n_precip_suspect": int(summary_row[3]),
            "n_precsum_fail": int(summary_row[4]),
            "n_precsum_suspect": int(summary_row[5]),
            "n_precip_negative": int(summary_row[6]),
            "n_precip_hard_rate": int(summary_row[7]),
            "n_precip_suspect_rate": int(summary_row[8]),
            "n_precip_isolated_spike": int(summary_row[9]),
            "n_precsum_absurd": int(summary_row[10]),
            "n_precsum_decrease": int(summary_row[11]),
            "n_precsum_mismatch": int(summary_row[12]),
            "n_gap_large": int(summary_row[13]),
            "n_usable_interval_extremes": int(summary_row[14]),
            "n_core_cadence": int(summary_row[15]),
            "n_supported_cadence": int(summary_row[16]),
            "n_first_rows": int(summary_row[17]),
        },
        "distribution": {
            "median_delta_min": None if summary_row[18] is None else float(summary_row[18]),
            "p99_precip_rate_mmh_qc": None if summary_row[19] is None else float(summary_row[19]),
            "p999_precip_rate_mmh_qc": None if summary_row[20] is None else float(summary_row[20]),
            "max_precip_rate_mmh_raw": None if summary_row[21] is None else float(summary_row[21]),
            "max_precip_raw_mm": None if summary_row[22] is None else float(summary_row[22]),
        },
        "worst_station_fail_load_top10": [
            {
                "station_id": row[0],
                "n_precip_fail": int(row[1]),
                "n_precip_suspect": int(row[2]),
                "n_precsum_fail": int(row[3]),
                "n_precsum_suspect": int(row[4]),
            }
            for row in worst_station_rows
        ],
        "outputs": {
            "qc_parquet": str(args.output),
            "station_summary_csv": str(args.station_summary_csv),
        },
    }

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.station_summary_csv}")
    print(f"Wrote: {args.summary_json}")
    print(
        "Rows="
        f"{summary['counts']['n_rows']} | stations={summary['counts']['n_stations']} | "
        f"precip FAIL={summary['counts']['n_precip_fail']} | precip SUSPECT={summary['counts']['n_precip_suspect']}"
    )


if __name__ == "__main__":
    main()

