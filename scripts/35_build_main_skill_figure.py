from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALE_ORDER = [30, 60, 180, 360]
POINT_COLOR = "#a44a3f"
GRID_COLOR = "#1f5f8b"
FIXED_STYLE = "-"
RELATIVE_STYLE = "--"
BG = "#f3f1eb"
PANEL = "#fbfbf8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the main multi-scale skill figure for the manuscript.")
    parser.add_argument(
        "--robust-skill-csv",
        type=Path,
        default=Path("results/avamet_imerg_window_confusion_robust_by_scale.csv"),
        help="Robust window-confusion summary CSV.",
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_skill_bootstrap.csv"),
        help="Bootstrap event-skill CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_2_main_skill_multiscale.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )
    return parser


def label_scales(values: list[int]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value < 60:
            labels.append(f"{value} min")
        else:
            labels.append(f"{value // 60} h")
    return labels


def load_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[df["stride_mode"].astype(str) == "scale_stride"].copy()
    df["window_min"] = pd.to_numeric(df["window_min"], errors="coerce").astype(int)
    df = df.loc[df["window_min"].isin(SCALE_ORDER)].copy()
    return df


def load_bootstrap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["scope"] = df["scope"].astype(str)
    df = df.loc[df["scope"].isin([str(value) for value in SCALE_ORDER])].copy()
    df["window_min"] = df["scope"].astype(int)
    return df.sort_values("window_min").reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def add_panel_title(ax: plt.Axes, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}  {title}", loc="left", fontsize=12, fontweight="bold")


def get_series(df: pd.DataFrame, threshold_mode: str) -> pd.DataFrame:
    subset = df.loc[df["threshold_mode"].astype(str) == threshold_mode].copy()
    subset = subset.sort_values("window_min").reset_index(drop=True)
    return subset


def plot_metric_panel(ax: plt.Axes, robust: pd.DataFrame, point_col: str, grid_col: str, ylabel: str, prefix: str, title: str) -> None:
    x = np.arange(len(SCALE_ORDER), dtype=float)
    fixed = get_series(robust, "fixed_threshold")
    relative = get_series(robust, "relative_threshold")

    for subset, linestyle, label_suffix in [
        (fixed, FIXED_STYLE, "Fixed"),
        (relative, RELATIVE_STYLE, "Relative"),
    ]:
        ax.plot(
            x,
            pd.to_numeric(subset[point_col], errors="coerce").to_numpy(dtype=float, copy=False),
            color=POINT_COLOR,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            label=f"Station cell | {label_suffix}",
        )
        ax.plot(
            x,
            pd.to_numeric(subset[grid_col], errors="coerce").to_numpy(dtype=float, copy=False),
            color=GRID_COLOR,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            label=f"3x3 tolerance | {label_suffix}",
        )

    ax.set_xticks(x, label_scales(SCALE_ORDER))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    add_panel_title(ax, prefix, title)


def plot_bootstrap_panel(ax: plt.Axes, boot: pd.DataFrame) -> None:
    x = np.arange(len(SCALE_ORDER), dtype=float)
    point = pd.to_numeric(boot["pod_point"], errors="coerce").to_numpy(dtype=float, copy=False)
    point_lo = pd.to_numeric(boot["pod_point_ci_low"], errors="coerce").to_numpy(dtype=float, copy=False)
    point_hi = pd.to_numeric(boot["pod_point_ci_high"], errors="coerce").to_numpy(dtype=float, copy=False)
    grid = pd.to_numeric(boot["pod_3x3"], errors="coerce").to_numpy(dtype=float, copy=False)
    grid_lo = pd.to_numeric(boot["pod_3x3_ci_low"], errors="coerce").to_numpy(dtype=float, copy=False)
    grid_hi = pd.to_numeric(boot["pod_3x3_ci_high"], errors="coerce").to_numpy(dtype=float, copy=False)

    ax.fill_between(x, point_lo, point_hi, color=POINT_COLOR, alpha=0.18)
    ax.fill_between(x, grid_lo, grid_hi, color=GRID_COLOR, alpha=0.18)
    ax.plot(x, point, color=POINT_COLOR, linewidth=2.3, marker="o", label="Station cell")
    ax.plot(x, grid, color=GRID_COLOR, linewidth=2.3, marker="o", label="3x3 tolerance")

    for idx, value in enumerate(point):
        ax.text(x[idx], value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8.4, color=POINT_COLOR)
    for idx, value in enumerate(grid):
        ax.text(x[idx], value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8.4, color=GRID_COLOR)

    ax.set_xticks(x, label_scales(SCALE_ORDER))
    ax.set_ylabel("Event POD (with 95% CI)")
    ax.set_ylim(0.55, 1.01)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    add_panel_title(ax, "D.", "Event-Based Detection Improves with Scale (Bootstrapped)")


def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    robust = load_robust(args.robust_skill_csv)
    boot = load_bootstrap(args.bootstrap_csv)

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    axes = axes.ravel()

    plot_metric_panel(
        axes[0],
        robust,
        "point_pod",
        "grid3x3_pod",
        "Window POD",
        "A.",
        "Window POD Is Very Low Under Fixed Thresholds\n(Relative Thresholds Shown for Comparison)",
    )
    axes[0].set_ylim(0.0, 0.34)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.6, ncol=2)

    plot_metric_panel(
        axes[1],
        robust,
        "point_ets",
        "grid3x3_ets",
        "Equitable Threat Score",
        "B.",
        "Relative Thresholds Lift Skill,\nBut Absolute Skill Remains Modest",
    )
    axes[1].set_ylim(0.0, 0.10)

    plot_metric_panel(
        axes[2],
        robust,
        "point_bias",
        "grid3x3_bias",
        "Frequency Bias",
        "C.",
        "3x3 Tolerance Raises Event Counts\nWhile Recovering Events",
    )
    axes[2].axhline(1.0, color="#555555", linewidth=1.1, linestyle=":")
    axes[2].set_ylim(0.0, 3.4)

    plot_bootstrap_panel(axes[3], boot)

    for ax in axes:
        style_axes(ax)

    fig.suptitle(
        "Multi-Scale IMERG Skill in the Comunitat Valenciana (eastern Spain)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Panels A-C use the non-overlapping `stride = W` window comparison, with fixed thresholds "
            "emphasized and relative thresholds shown for comparison.\n"
            "In Panel C, relative-threshold BIAS reflects exceedance-rate mismatch under product-native "
            "thresholds rather than a physical frequency bias. Panel D uses the event-based common-period "
            "product with UTC-day block bootstrap."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#444444",
    )

    fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.11, wspace=0.16, hspace=0.26)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()


