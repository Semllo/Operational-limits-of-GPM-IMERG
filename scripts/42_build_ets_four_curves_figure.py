from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCALE_ORDER = [30, 60, 180, 360]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a four-curve ETS figure by scale: IMERG point, IMERG 3x3, gauge-to-grid baseline, and LOO baseline."
    )
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_by_scale.csv"),
        help="IMERG robust window confusion CSV.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("results/avamet_grid_baseline_by_scale.csv"),
        help="Gauge-to-grid baseline CSV.",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed_threshold",
        help="Threshold mode filter for IMERG window table.",
    )
    parser.add_argument(
        "--stride-mode",
        type=str,
        default="scale_stride",
        help="Stride mode filter for IMERG window table.",
    )
    parser.add_argument(
        "--baseline-subset",
        type=str,
        default="multi_station_cells",
        help="Subset type for gauge-to-grid baseline curve.",
    )
    parser.add_argument(
        "--loo-subset",
        type=str,
        default="leave_one_out_multi_station_cells",
        help="Subset type for leave-one-out baseline curve.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_6_ets_four_curves.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output DPI.",
    )
    return parser


def load_imerg_ets(path: Path, threshold_mode: str, stride_mode: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    keep = df[(df["threshold_mode"] == threshold_mode) & (df["stride_mode"] == stride_mode)].copy()
    keep["window_min"] = pd.to_numeric(keep["window_min"], errors="coerce").astype("Int64")
    keep = keep[keep["window_min"].isin(SCALE_ORDER)].copy()
    if keep.empty:
        raise ValueError(
            f"No IMERG rows found for threshold_mode={threshold_mode!r}, stride_mode={stride_mode!r} in {path}"
        )
    keep = keep.sort_values("window_min")

    point = keep.set_index("window_min")["point_ets"].astype(float)
    grid3 = keep.set_index("window_min")["grid3x3_ets"].astype(float)
    return point, grid3


def load_baseline_ets(path: Path, subset: str) -> pd.Series:
    df = pd.read_csv(path)
    keep = df[df["subset_type"] == subset].copy()
    keep["window_min"] = pd.to_numeric(keep["window_min"], errors="coerce").astype("Int64")
    keep = keep[keep["window_min"].isin(SCALE_ORDER)].copy()
    if keep.empty:
        raise ValueError(f"No baseline rows found for subset_type={subset!r} in {path}")
    keep = keep.sort_values("window_min")
    return keep.set_index("window_min")["ets"].astype(float)


def scale_labels(values: list[int]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value < 60:
            labels.append(f"{value} min")
        else:
            labels.append(f"{value // 60} h")
    return labels


def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    imerg_point, imerg_3x3 = load_imerg_ets(args.window_csv, args.threshold_mode, args.stride_mode)
    baseline = load_baseline_ets(args.baseline_csv, args.baseline_subset)
    loo = load_baseline_ets(args.baseline_csv, args.loo_subset)

    for window in SCALE_ORDER:
        if window not in imerg_point.index or window not in imerg_3x3.index:
            raise ValueError(f"Missing IMERG ETS at window={window}")
        if window not in baseline.index:
            raise ValueError(f"Missing baseline ETS at window={window}")
        if window not in loo.index:
            raise ValueError(f"Missing LOO ETS at window={window}")

    x = list(range(len(SCALE_ORDER)))
    xticks = scale_labels(SCALE_ORDER)

    y_imerg_point = [float(imerg_point.loc[w]) for w in SCALE_ORDER]
    y_imerg_3x3 = [float(imerg_3x3.loc[w]) for w in SCALE_ORDER]
    y_baseline = [float(baseline.loc[w]) for w in SCALE_ORDER]
    y_loo = [float(loo.loc[w]) for w in SCALE_ORDER]

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "#f4f2ec",
            "axes.facecolor": "#fbfaf7",
        }
    )

    fig, ax = plt.subplots(figsize=(11.2, 6.6))

    ax.plot(x, y_imerg_point, marker="o", linewidth=2.2, color="#bc4749", label="IMERG station-cell")
    ax.plot(x, y_imerg_3x3, marker="o", linewidth=2.2, color="#2a9d8f", label="IMERG 3x3")
    ax.plot(x, y_baseline, marker="s", linewidth=2.2, color="#355070", label="Gauge-to-grid baseline")
    ax.plot(x, y_loo, marker="s", linewidth=2.2, color="#6a994e", label="Gauge-to-grid baseline (LOO)")

    ax.set_xticks(x, xticks)
    ax.set_ylabel("ETS")
    ax.set_xlabel("Accumulation window")
    ax.set_ylim(0.0, 0.55)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    title = "ETS by scale: IMERG vs gauge-to-grid references"
    subtitle = f"IMERG mode: {args.threshold_mode}, {args.stride_mode}"
    fig.suptitle(title, x=0.105, y=0.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.105, 0.952, subtitle, ha="left", va="top", fontsize=10, color="#444444")

    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.005))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()


