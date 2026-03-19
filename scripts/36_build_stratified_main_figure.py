from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALE_ORDER = ["30", "60", "180", "360"]
SEASON_COLORS = {"SON": "#8d0801", "Rest of year": "#355070"}
COAST_COLORS = {"Coastal": "#2a9d8f", "Interior": "#577590"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the streamlined main-text stratification figure.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_stratified_skill.csv"),
        help="Long-form stratified summary CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_4_stratification_main.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )
    return parser


def label_scales(values: list[str]) -> list[str]:
    labels: list[str] = []
    for value in values:
        minutes = int(value)
        if minutes < 60:
            labels.append(f"{minutes} min")
        else:
            labels.append(f"{minutes // 60} h")
    return labels


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["strata_family"] = df["strata_family"].astype(str)
    df["strata_value"] = df["strata_value"].astype(str)
    df["window_min"] = df["window_min"].astype(str)
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def add_panel_title(ax: plt.Axes, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}  {title}", loc="left", fontsize=12, fontweight="bold")


def plot_family(ax: plt.Axes, df: pd.DataFrame, family: str, order: list[str], colors: dict[str, str], title: str, prefix: str) -> None:
    subset = df[(df["strata_family"] == family) & (df["window_min"].isin(SCALE_ORDER))].copy()
    x = np.arange(len(SCALE_ORDER), dtype=float)

    for value in order:
        data = subset[subset["strata_value"] == value].set_index("window_min").reindex(SCALE_ORDER)
        y = pd.to_numeric(data["median_ratio_3x3"], errors="coerce").to_numpy(dtype=float, copy=False)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            color=colors[value],
            label=value,
        )
        for idx, metric in enumerate(y):
            if np.isnan(metric):
                continue
            ax.text(x[idx], metric + 0.008, f"{metric:.2f}", ha="center", va="bottom", fontsize=8.5, color=colors[value])

    ax.set_xticks(x, label_scales(SCALE_ORDER))
    ax.set_ylabel("Median IMERG / AVAMET ratio (3x3)")
    ax.set_ylim(0.0, 0.38)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    add_panel_title(ax, prefix, title)


def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.summary_csv)

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "#f3f1eb",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

    plot_family(
        axes[0],
        summary,
        family="season",
        order=["SON", "Rest of year"],
        colors=SEASON_COLORS,
        title="Seasonal Contrast in Amplitude Recovery",
        prefix="A.",
    )
    plot_family(
        axes[1],
        summary,
        family="coast",
        order=["Coastal", "Interior"],
        colors=COAST_COLORS,
        title="Interior Stations Retain More Amplitude Than Coastal Ones",
        prefix="B.",
    )

    for ax in axes:
        style_axes(ax)

    fig.suptitle(
        "Stratified Amplitude Recovery for IMERG Extreme Events in the Comunitat Valenciana (eastern Spain)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Province and altitude contrasts are retained in the supplementary stratified summary tables.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#444444",
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12, wspace=0.16)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()


