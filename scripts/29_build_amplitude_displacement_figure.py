from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALE_ORDER = [30, 60, 180, 360]
POINT_COLOR = "#a44a3f"
BEST3X3_COLOR = "#1f5f8b"
DISPLACED_COLOR = "#2a9d8f"
SAME_CELL_COLOR = "#d9e3ea"
DISTANCE_COLOR = "#355c7d"
P90_COLOR = "#c06c84"
AXIS_SHIFT_COLOR = "#4d7ea8"
DIAGONAL_SHIFT_COLOR = "#7b2cbf"
CASE_COLORS = ["#0f4c5c", "#bc4749", "#6a994e", "#7b2cbf"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the main-text amplitude-vs-displacement figure from the IMERG "
            "event decomposition summaries."
        )
    )
    parser.add_argument(
        "--by-scale-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement_by_scale.csv"),
        help="CSV summary by time scale.",
    )
    parser.add_argument(
        "--case-csv",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement_case_day_window.csv"),
        help="CSV summary by selected case-study day and time scale.",
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        default=Path("results/avamet_cv_events_imerg_amplitude_displacement.parquet"),
        help="Full enriched event parquet used to build displacement-type fractions.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/figures/figure_3_amplitude_displacement.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output DPI.",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default="Amplitude vs Displacement Decomposition of IMERG Extremes over the Comunitat Valenciana (eastern Spain)",
        help="Figure title.",
    )
    return parser


def load_scale_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["scope"] = df["scope"].astype(str)
    df = df.loc[df["scope"].isin([str(value) for value in SCALE_ORDER])].copy()
    if df.empty:
        raise ValueError(f"No scale rows found in {path}")
    df["window_min"] = df["scope"].astype(int)
    df = df.sort_values("window_min").reset_index(drop=True)
    return df


def load_case_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["window_min"] = pd.to_numeric(df["window_min"], errors="coerce").astype("Int64")
    df = df.loc[df["window_min"].isin(SCALE_ORDER)].copy()
    df = df.sort_values(["date", "window_min"]).reset_index(drop=True)
    return df


def load_displacement_profile(path: Path) -> pd.DataFrame:
    columns = [
        "window_min",
        "imerg_best_3x3_delta_lon_cells",
        "imerg_best_3x3_delta_lat_cells",
    ]
    df = pd.read_parquet(path, columns=columns)
    df["window_min"] = pd.to_numeric(df["window_min"], errors="coerce").astype("Int64")
    df = df.loc[df["window_min"].isin(SCALE_ORDER)].copy()
    if df.empty:
        raise ValueError(f"No displacement rows found in {path}")

    delta_lon = pd.to_numeric(df["imerg_best_3x3_delta_lon_cells"], errors="coerce")
    delta_lat = pd.to_numeric(df["imerg_best_3x3_delta_lat_cells"], errors="coerce")
    df["same_cell"] = np.isclose(delta_lon, 0.0) & np.isclose(delta_lat, 0.0)
    df["axis_shift"] = (
        (np.isclose(np.abs(delta_lon), 1.0) & np.isclose(np.abs(delta_lat), 0.0))
        | (np.isclose(np.abs(delta_lon), 0.0) & np.isclose(np.abs(delta_lat), 1.0))
    )
    df["diagonal_shift"] = np.isclose(np.abs(delta_lon), 1.0) & np.isclose(np.abs(delta_lat), 1.0)
    df["other_shift"] = ~(df["same_cell"] | df["axis_shift"] | df["diagonal_shift"])

    rows: list[dict[str, float | int]] = []
    for window_min in SCALE_ORDER:
        subset = df.loc[df["window_min"] == window_min]
        if subset.empty:
            continue
        rows.append(
            {
                "window_min": int(window_min),
                "same_cell": float(subset["same_cell"].mean()),
                "axis_shift": float(subset["axis_shift"].mean()),
                "diagonal_shift": float(subset["diagonal_shift"].mean()),
                "other_shift": float(subset["other_shift"].mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("window_min").reset_index(drop=True)
    return out


def label_scales(values: list[int]) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value < 60:
            labels.append(f"{value} min")
        else:
            labels.append(f"{value // 60} h")
    return labels


def safe_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)


def add_panel_title(ax: plt.Axes, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}  {title}", loc="left", fontsize=12, fontweight="bold")


def plot_recovery_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    scales = summary["window_min"].astype(int).tolist()
    x = np.arange(len(scales), dtype=float)
    point_ratio = safe_values(summary["median_point_recovery_ratio"])
    best_ratio = safe_values(summary["median_3x3_recovery_ratio"])

    ax.plot(x, point_ratio, marker="o", linewidth=2.4, color=POINT_COLOR, label="Station cell")
    ax.plot(x, best_ratio, marker="o", linewidth=2.4, color=BEST3X3_COLOR, label="3x3 tolerance")

    for idx, value in enumerate(point_ratio):
        ax.text(x[idx], value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=POINT_COLOR)
    for idx, value in enumerate(best_ratio):
        ax.text(x[idx], value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=BEST3X3_COLOR)

    ax.set_xticks(x, label_scales(scales))
    ax.set_ylabel("Median IMERG / AVAMET ratio")
    ax.set_ylim(0.0, max(0.34, float(np.nanmax(best_ratio)) * 1.22))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    add_panel_title(ax, "A.", "Amplitude Recovery Improves Under 3x3 Tolerance")


def plot_displacement_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    scales = summary["window_min"].astype(int).tolist()
    x = np.arange(len(scales), dtype=float)
    same = safe_values(summary["fraction_same_cell"])
    displaced = 1.0 - same
    gain = safe_values(summary["fraction_gain_positive"])
    width = 0.64

    ax.bar(x, same, width=width, color=SAME_CELL_COLOR, edgecolor="white", linewidth=0.9, label="Same cell")
    ax.bar(
        x,
        displaced,
        width=width,
        bottom=same,
        color=DISPLACED_COLOR,
        edgecolor="white",
        linewidth=0.9,
        label="Displaced",
    )
    ax.plot(x, gain, color=POINT_COLOR, linewidth=2.0, marker="D", label="Gain > 0")

    for idx, value in enumerate(displaced):
        ax.text(x[idx], same[idx] + displaced[idx] / 2.0, f"{value:.0%}", ha="center", va="center", fontsize=9, color="white")

    if len(x) > 0:
        label_x = x[0] - 0.38
        ax.text(label_x, same[0] / 2.0, "Same", ha="right", va="center", fontsize=8.3, color="#33414d")
        ax.text(
            label_x,
            same[0] + displaced[0] / 2.0 - 0.08,
            "Displ.",
            ha="right",
            va="center",
            fontsize=8.3,
            color="white",
        )
        ax.text(
            label_x,
            0.94,
            "Gain > 0",
            ha="right",
            va="center",
            fontsize=8.5,
            color=POINT_COLOR,
        )

    ax.set_xticks(x, label_scales(scales))
    ax.set_ylabel("Event fraction")
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    add_panel_title(ax, "B.", "Most Events Improve Under 3x3 Tolerance")


def plot_distance_panel(ax: plt.Axes, profile: pd.DataFrame) -> None:
    scales = profile["window_min"].astype(int).tolist()
    x = np.arange(len(scales), dtype=float)
    same = safe_values(profile["same_cell"])
    axis = safe_values(profile["axis_shift"])
    diagonal = safe_values(profile["diagonal_shift"])
    other = safe_values(profile["other_shift"])

    ax.bar(x, same, width=0.64, color=SAME_CELL_COLOR, edgecolor="white", linewidth=0.9, label="Same")
    ax.bar(x, axis, width=0.64, bottom=same, color=AXIS_SHIFT_COLOR, edgecolor="white", linewidth=0.9, label="Adjacent")
    ax.bar(
        x,
        diagonal,
        width=0.64,
        bottom=same + axis,
        color=DIAGONAL_SHIFT_COLOR,
        edgecolor="white",
        linewidth=0.9,
        label="Diagonal",
    )
    if np.any(other > 0.0):
        ax.bar(
            x,
            other,
            width=0.64,
            bottom=same + axis + diagonal,
            color=P90_COLOR,
            edgecolor="white",
            linewidth=0.9,
            label="Other",
        )

    for idx, value in enumerate(diagonal):
        if value <= 0.0:
            continue
        midpoint = same[idx] + axis[idx] + value / 2.0
        ax.text(x[idx], midpoint, f"{value:.0%}", ha="center", va="center", fontsize=9, color="white")

    if len(x) > 0:
        label_x = x[0] - 0.38
        ax.text(label_x, same[0] / 2.0, "Same", ha="right", va="center", fontsize=8.0, color="#33414d")
        ax.text(
            label_x,
            same[0] + axis[0] / 2.0,
            "1-cell",
            ha="right",
            va="center",
            fontsize=8.0,
            color="white",
        )
        ax.text(
            label_x,
            same[0] + axis[0] + diagonal[0] / 2.0,
            "Diag.",
            ha="right",
            va="center",
            fontsize=8.0,
            color="white",
        )

    ax.set_xticks(x, label_scales(scales))
    ax.set_ylabel("Fraction of events")
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8.2,
        ncol=4,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    add_panel_title(ax, "C.", "Peak Offsets Mostly Stay Within Adjacent IMERG Cells")


def plot_case_panel(ax: plt.Axes, case_df: pd.DataFrame) -> None:
    add_panel_title(ax, "D.", "Selected DANA Case-Study Days Show Larger 3x3-Tolerance Gains")
    if case_df.empty:
        ax.text(0.5, 0.5, "No candidate-day summary available", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    dates = list(dict.fromkeys(case_df["date"].dropna().astype(str).tolist()))
    for idx, date in enumerate(dates):
        subset = case_df.loc[case_df["date"] == date].copy()
        subset = subset.sort_values("window_min")
        x = np.arange(subset.shape[0], dtype=float)
        gain = safe_values(subset["median_gain_mm"])
        ax.plot(
            x,
            gain,
            marker="o",
            linewidth=2.3,
            color=CASE_COLORS[idx % len(CASE_COLORS)],
            label=date,
        )
        for jdx, value in enumerate(gain):
            ax.text(
                x[jdx],
                value + 0.35,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=CASE_COLORS[idx % len(CASE_COLORS)],
            )

    ax.set_xticks(np.arange(len(SCALE_ORDER), dtype=float), label_scales(SCALE_ORDER))
    ax.set_ylabel("Median gain under 3x3 tolerance (mm)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", title="Date")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = load_scale_summary(args.by_scale_csv)
    case_df = load_case_summary(args.case_csv)
    displacement_profile = load_displacement_profile(args.events_parquet)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "#f3f1eb",
            "axes.titlepad": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    axes = axes.ravel()

    plot_recovery_panel(axes[0], summary)
    plot_displacement_panel(axes[1], summary)
    plot_distance_panel(axes[2], displacement_profile)
    plot_case_panel(axes[3], case_df)

    for ax in axes:
        style_axes(ax)

    fig.suptitle(
        args.figure_title,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Recovery ratios compare IMERG against the AVAMET event accumulation. "
            "The 3x3 tolerance series is a spatial-tolerance upper bound, not a replacement product."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.16, wspace=0.12, hspace=0.24)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output_png}")


if __name__ == "__main__":
    main()

