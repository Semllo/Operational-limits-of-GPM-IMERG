from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final main-text DANA case-study figure package.")
    parser.add_argument(
        "--maps-png",
        type=Path,
        default=Path("results/figures/dana_comparison_maps_paper.png"),
        help="Existing map package PNG.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/dana_episode_summary_table.csv"),
        help="Numeric DANA summary CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_5_dana_cases_main.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    image = plt.imread(args.maps_png)

    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.facecolor": "#f3f1eb",
        }
    )

    fig = plt.figure(figsize=(12.4, 8.0))
    ax_img = fig.add_subplot(1, 1, 1)
    ax_img.imshow(image)
    ax_img.axis("off")

    fig.suptitle(
        "DANA Case Studies: AVAMET vs IMERG over the Comunitat Valenciana (eastern Spain)",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    fig.text(
        0.5,
        0.02,
        "Maps show daily accumulations for spatial context; the event catalog and case metrics remain based on 30-360 min windows.",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#444444",
    )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.08)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()


