from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BG = "#f3f1eb"
PANEL = "#fbfbf8"
AVAMET = "#355070"
IMERG = "#2a9d8f"
MERGE = "#bc6c25"
CHECK = "#6d597a"
TEXT = "#2f2f2f"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the methods / comparability schematic for the manuscript.")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_1_methods_schematic.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )
    return parser


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=color,
        facecolor=PANEL,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.055, title, ha="left", va="top", fontsize=12, fontweight="bold", color=color)
    ax.text(x + 0.02, y + h - 0.10, body, ha="left", va="top", fontsize=9.7, color=TEXT, linespacing=1.4)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#666666") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )

    fig = plt.figure(figsize=(13.4, 7.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Methods and Comparability Framework",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    add_box(
        ax,
        (0.05, 0.62),
        (0.26, 0.24),
        "AVAMET Reference",
        (
            "568 stations in the Comunitat Valenciana\n"
            "QC + completeness filters\n"
            "POT catalog at 30, 60, 180, 360 min"
        ),
        AVAMET,
    )
    add_box(
        ax,
        (0.05, 0.24),
        (0.26, 0.24),
        "IMERG Input",
        (
            "IMERG V07 half-hourly fields\n"
            "2019-01-01 to 2025-09-30\n"
            "CV subset extracted from the full archive"
        ),
        IMERG,
    )
    add_box(
        ax,
        (0.36, 0.41),
        (0.27, 0.31),
        "Alignment Layer",
        (
            "Common 30 min windows\n"
            "Station-to-cell alignment\n"
            "Station cell and 3x3 tolerance\n"
            "AVAMET->grid representativeness baseline"
        ),
        MERGE,
    )
    add_box(
        ax,
        (0.72, 0.61),
        (0.22, 0.25),
        "Skill Family A",
        (
            "Fixed-threshold operational skill\n"
            "AVAMET POT threshold applied to IMERG\n"
            "POD, ETS, BIAS (stride = W)"
        ),
        CHECK,
    )
    add_box(
        ax,
        (0.72, 0.30),
        (0.22, 0.25),
        "Skill Family B",
        (
            "Relative-occurrence skill\n"
            "IMERG p99.5 by cell / scale\n"
            "Separates detection from amplitude loss"
        ),
        CHECK,
    )
    add_box(
        ax,
        (0.38, 0.08),
        (0.56, 0.16),
        "Diagnostic Layers",
        (
            "Amplitude vs displacement | block-bootstrap event skill\n"
            "Stratification (season, coast, province, altitude) | DANA case studies"
        ),
        "#7c6a0a",
    )

    add_arrow(ax, (0.31, 0.74), (0.38, 0.60))
    add_arrow(ax, (0.31, 0.36), (0.38, 0.52))
    add_arrow(ax, (0.63, 0.60), (0.72, 0.73))
    add_arrow(ax, (0.63, 0.52), (0.72, 0.42))
    add_arrow(ax, (0.495, 0.41), (0.495, 0.24))

    ax.text(
        0.94,
        0.585,
        "Feeds Fig. 2",
        ha="right",
        va="center",
        fontsize=9.6,
        color=CHECK,
        fontweight="bold",
    )
    ax.text(
        0.94,
        0.275,
        "Explains Fig. 2",
        ha="right",
        va="center",
        fontsize=9.2,
        color=CHECK,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.025,
        (
            "The comparison is explicitly structured to separate point-to-pixel representativeness, "
            "amplitude loss, and spatial displacement."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()



