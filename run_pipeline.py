from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal standalone pipeline for paper 6."
    )
    parser.add_argument(
        "--avamet-input",
        type=Path,
        default=Path("data/avamet_all.parquet"),
        help="AVAMET parquet input.",
    )
    parser.add_argument(
        "--imerg-root",
        type=Path,
        default=Path("data/imerg"),
        help="Root directory with raw IMERG HDF5 files.",
    )
    parser.add_argument(
        "--imerg-cv-root",
        type=Path,
        default=Path("data/imerg_cv"),
        help="Root directory for extracted IMERG CV subsets.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip step 02 and assume data/imerg_cv already exists.",
    )
    parser.add_argument(
        "--overwrite-imerg-cv",
        action="store_true",
        help="Overwrite existing extracted IMERG CV subset files.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates for event-based uncertainty.",
    )
    return parser


def run_step(repo_root: Path, name: str, args: list[str]) -> None:
    print(f"[START] {name}")
    command = [sys.executable, *args]
    subprocess.run(command, cwd=repo_root, check=True)
    print(f"[DONE]  {name}")


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent

    (repo_root / "results").mkdir(parents=True, exist_ok=True)
    (repo_root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (repo_root / "data").mkdir(parents=True, exist_ok=True)

    steps: list[tuple[str, list[str]]] = [
        (
            "00_build_cv_station_subset.py",
            [
                "scripts/00_build_cv_station_subset.py",
                "--input",
                str(args.avamet_input),
                "--output-dir",
                "results",
                "--write-filtered-parquet",
                "--filtered-parquet",
                "results/avamet_cv.parquet",
            ],
        ),
        (
            "01_build_cv_imerg_alignment.py",
            [
                "scripts/01_build_cv_imerg_alignment.py",
                "--station-csv",
                "results/avamet_station_inventory_cv.csv",
                "--imerg-root",
                str(args.imerg_root),
                "--output-dir",
                "results",
            ],
        ),
    ]

    if not args.skip_extraction:
        extract_step = (
            "02_extract_imerg_cv_subset.py",
            [
                "scripts/02_extract_imerg_cv_subset.py",
                "--input",
                str(args.imerg_root),
                "--metadata-json",
                "results/imerg_cv_subset_metadata.json",
                "--grid-csv",
                "results/imerg_cv_grid_cells.csv",
                "--output-root",
                str(args.imerg_cv_root),
                "--progress-every",
                "500",
            ],
        )
        if args.overwrite_imerg_cv:
            extract_step[1].append("--overwrite")
        steps.append(extract_step)

    steps.extend(
        [
            ("15_build_avamet_extremes_qc.py", ["scripts/15_build_avamet_extremes_qc.py"]),
            ("16_build_avamet_window_completeness.py", ["scripts/16_build_avamet_window_completeness.py"]),
            ("17_build_avamet_event_catalog.py", ["scripts/17_build_avamet_event_catalog.py"]),
            ("18_augment_avamet_event_catalog_support.py", ["scripts/18_augment_avamet_event_catalog_support.py"]),
            ("20_build_top_event_audit.py", ["scripts/20_build_top_event_audit.py"]),
            ("21_build_imerg_event_matches.py", ["scripts/21_build_imerg_event_matches.py"]),
            ("22_build_common_period_imerg_skill.py", ["scripts/22_build_common_period_imerg_skill.py"]),
            ("25_build_imerg_window_confusion_robust.py", ["scripts/25_build_imerg_window_confusion_robust.py"]),
            ("26_build_avamet_grid_baseline.py", ["scripts/26_build_avamet_grid_baseline.py"]),
            (
                "27_bootstrap_event_skill_blocks.py",
                [
                    "scripts/27_bootstrap_event_skill_blocks.py",
                    "--n-bootstrap",
                    str(args.n_bootstrap),
                ],
            ),
            ("24_build_dana_comparison_maps.py", ["scripts/24_build_dana_comparison_maps.py"]),
            ("28_build_amplitude_displacement_metrics.py", ["scripts/28_build_amplitude_displacement_metrics.py"]),
            ("29_build_amplitude_displacement_figure.py", ["scripts/29_build_amplitude_displacement_figure.py"]),
            ("30_build_outlier_policy_sensitivity.py", ["scripts/30_build_outlier_policy_sensitivity.py"]),
            ("31_build_stratified_skill.py", ["scripts/31_build_stratified_skill.py"]),
            ("33_build_dana_episode_table.py", ["scripts/33_build_dana_episode_table.py"]),
            ("34_build_methods_schematic.py", ["scripts/34_build_methods_schematic.py"]),
            ("35_build_main_skill_figure.py", ["scripts/35_build_main_skill_figure.py"]),
            ("36_build_stratified_main_figure.py", ["scripts/36_build_stratified_main_figure.py"]),
            ("37_build_dana_case_package_figure.py", ["scripts/37_build_dana_case_package_figure.py"]),
            ("42_build_ets_four_curves_figure.py", ["scripts/42_build_ets_four_curves_figure.py"]),
        ]
    )

    for name, step_args in steps:
        run_step(repo_root, name, step_args)


if __name__ == "__main__":
    main()
