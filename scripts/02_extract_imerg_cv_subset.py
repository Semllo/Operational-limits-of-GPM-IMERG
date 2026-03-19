from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Valencian Community IMERG subsets from downloaded HDF5 files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/imerg"),
        help="IMERG HDF5 file or root directory.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("results/imerg_cv_subset_metadata.json"),
        help="Subset metadata JSON produced by 01_build_cv_imerg_alignment.py.",
    )
    parser.add_argument(
        "--grid-csv",
        type=Path,
        default=Path("results/imerg_cv_grid_cells.csv"),
        help="Grid CSV produced by 01_build_cv_imerg_alignment.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/imerg_cv"),
        help="Directory where subset .npz files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of files to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing .cv.npz outputs. By default, existing outputs are skipped.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Emit a progress line every N newly processed files.",
    )
    parser.add_argument(
        "--keep-bbox",
        action="store_true",
        help="If set, keep full CV bounding box values. By default, cells outside the CV polygon are masked to NaN.",
    )
    return parser


def load_subset_spec(metadata_json: Path, grid_csv: Path) -> tuple[slice, slice, np.ndarray]:
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    lon_start = int(metadata["lon_index_range"]["start"])
    lon_end = int(metadata["lon_index_range"]["end"])
    lat_start = int(metadata["lat_index_range"]["start"])
    lat_end = int(metadata["lat_index_range"]["end"])
    lon_count = int(metadata["lon_index_range"]["count"])
    lat_count = int(metadata["lat_index_range"]["count"])

    grid_df = pd.read_csv(grid_csv)
    grid_df = grid_df.sort_values(["lon_idx", "lat_idx"]).reset_index(drop=True)
    mask = grid_df["inside_cv_polygon"].to_numpy(dtype=bool).reshape(lon_count, lat_count)

    return slice(lon_start, lon_end + 1), slice(lat_start, lat_end + 1), mask


def discover_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.HDF5"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def mask_outside_polygon(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = data.astype(np.float32, copy=True)
    masked[~mask] = np.nan
    return masked


def extract_one(
    input_file: Path,
    source_root: Path,
    output_root: Path,
    lon_slice: slice,
    lat_slice: slice,
    mask: np.ndarray,
    keep_bbox: bool,
    overwrite: bool,
) -> Path:
    rel = input_file.relative_to(source_root) if source_root.is_dir() else Path(input_file.name)
    output_path = (output_root / rel).with_suffix(".cv.npz")
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, "r") as handle:
        lon = handle["Grid/lon"][lon_slice].astype(np.float32)
        lat = handle["Grid/lat"][lat_slice].astype(np.float32)
        precipitation = handle["Grid/precipitation"][0, lon_slice, lat_slice].astype(np.float32)
        random_error = handle["Grid/randomError"][0, lon_slice, lat_slice].astype(np.float32)
        quality = handle["Grid/precipitationQualityIndex"][0, lon_slice, lat_slice].astype(np.float32)
        time_value = int(handle["Grid/time"][0])

    if not keep_bbox:
        precipitation = mask_outside_polygon(precipitation, mask)
        random_error = mask_outside_polygon(random_error, mask)
        quality = mask_outside_polygon(quality, mask)

    np.savez_compressed(
        output_path,
        lon=lon,
        lat=lat,
        inside_cv_polygon=mask,
        precipitation=precipitation,
        random_error=random_error,
        precipitation_quality_index=quality,
        time=time_value,
        source_file=str(input_file),
    )
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input
    files = discover_files(input_path)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError("No IMERG HDF5 files found to process.")

    lon_slice, lat_slice, mask = load_subset_spec(args.metadata_json, args.grid_csv)

    written = []
    skipped = 0
    for input_file in files:
        rel = input_file.relative_to(input_path) if input_path.is_dir() else Path(input_file.name)
        output_path = (args.output_root / rel).with_suffix(".cv.npz")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        out = extract_one(
            input_file=input_file,
            source_root=input_path,
            output_root=args.output_root,
            lon_slice=lon_slice,
            lat_slice=lat_slice,
            mask=mask,
            keep_bbox=args.keep_bbox,
            overwrite=bool(args.overwrite),
        )
        written.append(out)
        if args.progress_every > 0 and len(written) % int(args.progress_every) == 0:
            print(f"Processed {len(written)} files (skipped existing: {skipped})")

    print(f"Processed files: {len(written)}")
    print(f"Skipped existing: {skipped}")
    if written:
        print(f"First output: {written[0]}")
        print(f"Last output: {written[-1]}")


if __name__ == "__main__":
    main()

