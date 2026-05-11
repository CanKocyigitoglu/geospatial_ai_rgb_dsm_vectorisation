from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None


"""
Prepare real ISPRS Potsdam/Vaihingen RGB/TOP + DSM + colour-label data
for the project training pipeline.

Output format is compatible with the existing project:

processed_dir/
├── rgb/
│   └── <tile_id>_rgb.png
├── dsm/
│   └── <tile_id>_dsm.npy
├── masks/
│   └── <tile_id>_mask.png
└── metadata.csv

Default class reduction:
0 = background / ignored / clutter / car
1 = building
2 = vegetation       (low vegetation + tree)
3 = road_or_impervious

This reduced mapping keeps the project beginner-friendly while still covering
RGB imagery, DSM height data, CNN segmentation, feature extraction, and GIS
vectorisation.
"""

# Official ISPRS label colours -> original semantic class name.
# Source convention: ISPRS 2D semantic labeling benchmark.
ISPRS_COLOUR_TO_NAME = {
    (255, 255, 255): "impervious_surface",
    (0, 0, 255): "building",
    (0, 255, 255): "low_vegetation",
    (0, 255, 0): "tree",
    (255, 255, 0): "car",
    (255, 0, 0): "clutter_background",
}

# Reduced four-class mapping used by the existing U-Net in this project.
REDUCED_CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "vegetation",
    3: "road_or_impervious",
}

ISPRS_COLOUR_TO_REDUCED_ID = {
    (255, 255, 255): 3,  # impervious surfaces -> road/impervious
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low vegetation -> vegetation
    (0, 255, 0): 2,      # tree -> vegetation
    (255, 255, 0): 0,    # car -> ignore/background for this beginner project
    (255, 0, 0): 0,      # clutter/background -> background
}


def normalise_array_layout(arr: np.ndarray) -> np.ndarray:
    """Convert common TIFF layouts to H x W x C when needed."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8 RGB-like image."""
    arr = normalise_array_layout(arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {arr.shape}")

    arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


# def read_dsm(path: Path) -> np.ndarray:
#     """Read DSM/nDSM TIFF as float32 H x W array."""
#     arr = np.asarray(tiff.imread(path))
#     arr = normalise_array_layout(arr)
#     if arr.ndim == 3:
#         arr = arr[..., 0]
#     arr = arr.astype(np.float32)
#     if np.isnan(arr).any():
#         median = float(np.nanmedian(arr))
#         arr = np.nan_to_num(arr, nan=median)
#     return arr

def read_dsm(path):
    dsm = np.array(Image.open(path)).astype(np.float32)

    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]

    # Normalise DSM/nDSM to 0–1 for CNN input
    dsm = dsm - dsm.min()
    if dsm.max() > 0:
        dsm = dsm / dsm.max()

    return dsm


def label_rgb_to_reduced_mask(label_rgb: np.ndarray, tolerance: int = 5) -> np.ndarray:
    """
    Convert ISPRS colour-coded label image to reduced class IDs.

    Exact matching is used first. If a pixel colour is slightly off because of
    reading/conversion artefacts, nearest-colour matching within tolerance is used.
    """
    label_rgb = normalise_array_layout(label_rgb)
    if label_rgb.ndim == 2:
        raise ValueError("Label image should be colour-coded RGB, but got a single-band image.")
    label_rgb = label_rgb[..., :3].astype(np.int16)

    h, w, _ = label_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    assigned = np.zeros((h, w), dtype=bool)

    # Exact matching.
    for colour, class_id in ISPRS_COLOUR_TO_REDUCED_ID.items():
        colour_arr = np.array(colour, dtype=np.int16)
        hits = np.all(label_rgb == colour_arr, axis=-1)
        mask[hits] = class_id
        assigned |= hits

    # Nearest colour fallback for any remaining pixels.
    if not np.all(assigned):
        colours = np.array(list(ISPRS_COLOUR_TO_REDUCED_ID.keys()), dtype=np.int16)
        ids = np.array(list(ISPRS_COLOUR_TO_REDUCED_ID.values()), dtype=np.uint8)

        flat = label_rgb.reshape(-1, 3)
        flat = flat.astype(np.float32)
        colours = colours.astype(np.float32)
        distances = np.sqrt(((flat[:, None, :] - colours[None, :, :]) ** 2).sum(axis=2))
        nearest_idx = distances.argmin(axis=1)
        nearest_dist = distances.min(axis=1)
        mapped = ids[nearest_idx].reshape(h, w)
        close = nearest_dist.reshape(h, w) <= tolerance
        mask[~assigned & close] = mapped[~assigned & close]

    return mask


def extract_area_key(path: Path) -> Optional[str]:
    """
    Extract a loose area key from Potsdam or Vaihingen filenames.

    Potsdam examples:
      top_potsdam_2_10_RGB.tif -> potsdam_2_10
      dsm_potsdam_02_10.tif    -> potsdam_2_10

    Vaihingen examples:
      top_mosaic_09cm_area1.tif       -> area_1
      dsm_09cm_matching_area1.tif     -> area_1
      top_mosaic_09cm_area1_class.tif -> area_1
    """
    stem = path.stem.lower()

    area_match = re.search(r"area\s*0*(\d+)", stem)
    if area_match:
        return f"area_{int(area_match.group(1))}"

    nums = re.findall(r"\d+", stem)
    if "potsdam" in stem and len(nums) >= 2:
        return f"potsdam_{int(nums[-2])}_{int(nums[-1])}"
    if len(nums) >= 2:
        return f"tile_{int(nums[-2])}_{int(nums[-1])}"
    if len(nums) == 1:
        return f"tile_{int(nums[0])}"
    return None


def list_tiffs(folder: Path) -> list[Path]:
    files = []
    for pattern in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"):
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def build_key_map(folder: Path) -> dict[str, Path]:
    mapping = {}
    for path in list_tiffs(folder):
        key = extract_area_key(path)
        if key is not None and key not in mapping:
            mapping[key] = path
    return mapping


def parse_tfw(tiff_path: Path) -> Optional[tuple[float, float, float, float, float, float]]:
    """
    Parse a TIFF world file if available.

    World file order:
    A: pixel size in x direction
    D: rotation about y-axis
    B: rotation about x-axis
    E: pixel size in y direction, usually negative
    C: x-coordinate of centre of upper-left pixel
    F: y-coordinate of centre of upper-left pixel
    """
    candidates = [
        tiff_path.with_suffix(".tfw"),
        tiff_path.with_suffix(".TFW"),
        tiff_path.with_suffix(".tifw"),
    ]
    for tfw in candidates:
        if tfw.exists():
            values = [float(line.strip()) for line in tfw.read_text().splitlines() if line.strip()]
            if len(values) >= 6:
                return tuple(values[:6])  # type: ignore[return-value]
    return None


def pixel_to_projected(tfw, x: float, y: float) -> tuple[float, float]:
    A, D, B, E, C, F = tfw
    X = A * x + B * y + C
    Y = D * x + E * y + F
    return X, Y


def make_transformer(source_crs: Optional[str]):
    if source_crs is None or Transformer is None:
        return None
    return Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)


def projected_to_lonlat(transformer, x: float, y: float) -> tuple[float, float]:
    if transformer is None:
        # Fallback for non-georeferenced data: values are treated as already lon/lat.
        return float(x), float(y)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def derive_patch_georef(
    top_path: Path,
    x0: int,
    y0: int,
    fallback_index: int,
    source_crs: Optional[str],
) -> dict:
    """
    Create the metadata fields expected by the existing vectorisation pipeline.

    If a .tfw world file and source CRS are available, coordinates are transformed
    to EPSG:4326 for Folium display. Otherwise demo coordinates are used.
    """
    tfw = parse_tfw(top_path)
    transformer = make_transformer(source_crs)

    if tfw is None or transformer is None:
        return {
            "origin_lon": -1.3500 + fallback_index * 0.001,
            "origin_lat": 51.5200 + fallback_index * 0.001,
            "pixel_size_lon": 0.000006,
            "pixel_size_lat": 0.000006,
            "crs": "EPSG:4326_demo_fallback",
        }

    x_proj, y_proj = pixel_to_projected(tfw, x0, y0)
    x_proj_right, y_proj_right = pixel_to_projected(tfw, x0 + 1, y0)
    x_proj_down, y_proj_down = pixel_to_projected(tfw, x0, y0 + 1)

    lon, lat = projected_to_lonlat(transformer, x_proj, y_proj)
    lon_right, lat_right = projected_to_lonlat(transformer, x_proj_right, y_proj_right)
    lon_down, lat_down = projected_to_lonlat(transformer, x_proj_down, y_proj_down)

    return {
        "origin_lon": lon,
        "origin_lat": lat,
        "pixel_size_lon": abs(lon_right - lon),
        "pixel_size_lat": abs(lat_down - lat),
        "crs": "EPSG:4326",
    }


def save_patch(
    output_dir: Path,
    tile_id: str,
    rgb_patch: np.ndarray,
    dsm_patch: np.ndarray,
    mask_patch: np.ndarray,
) -> dict:
    rgb_dir = output_dir / "rgb"
    dsm_dir = output_dir / "dsm"
    mask_dir = output_dir / "masks"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dsm_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / f"{tile_id}_rgb.png"
    dsm_path = dsm_dir / f"{tile_id}_dsm.npy"
    mask_path = mask_dir / f"{tile_id}_mask.png"

    Image.fromarray(rgb_patch).save(rgb_path)
    np.save(dsm_path, dsm_patch.astype(np.float32))
    Image.fromarray(mask_patch.astype(np.uint8)).save(mask_path)

    return {
        "tile_id": tile_id,
        "rgb_path": str(rgb_path),
        "dsm_path": str(dsm_path),
        "mask_path": str(mask_path),
    }


def class_count(mask: np.ndarray) -> int:
    return int(len(np.unique(mask)))


def prepare_dataset(
    rgb_dir: Path,
    dsm_dir: Path,
    label_dir: Path,
    output_dir: Path,
    patch_size: int = 512,
    stride: int = 512,
    max_patches: Optional[int] = None,
    min_classes: int = 2,
    source_crs: Optional[str] = None,
) -> None:
    rgb_map = build_key_map(rgb_dir)
    dsm_map = build_key_map(dsm_dir)
    label_map = build_key_map(label_dir)

    shared_keys = sorted(set(rgb_map) & set(dsm_map) & set(label_map))
    if not shared_keys:
        raise RuntimeError(
            "No matching RGB/DSM/label tiles found. Check your folders and filenames. "
            "For Potsdam, keep matching files such as top_potsdam_2_10_RGB.tif, "
            "dsm_potsdam_02_10.tif, and top_potsdam_2_10_label.tif in separate folders."
        )

    print(f"Matched {len(shared_keys)} full RGB/DSM/label tiles.")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    saved_count = 0

    for key in shared_keys:
        rgb_path = rgb_map[key]
        dsm_path = dsm_map[key]
        label_path = label_map[key]

        print(f"Processing {key}")
        print(f"  RGB:   {rgb_path.name}")
        print(f"  DSM:   {dsm_path.name}")
        print(f"  Label: {label_path.name}")

        rgb = to_uint8_rgb(tiff.imread(rgb_path))
        dsm = read_dsm(dsm_path)
        mask = label_rgb_to_reduced_mask(tiff.imread(label_path))

        h = min(rgb.shape[0], dsm.shape[0], mask.shape[0])
        w = min(rgb.shape[1], dsm.shape[1], mask.shape[1])
        rgb, dsm, mask = rgb[:h, :w], dsm[:h, :w], mask[:h, :w]

        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                mask_patch = mask[y0:y0 + patch_size, x0:x0 + patch_size]
                if class_count(mask_patch) < min_classes:
                    continue

                rgb_patch = rgb[y0:y0 + patch_size, x0:x0 + patch_size]
                dsm_patch = dsm[y0:y0 + patch_size, x0:x0 + patch_size]

                tile_id = f"{key}_r{y0}_c{x0}"
                row = save_patch(output_dir, tile_id, rgb_patch, dsm_patch, mask_patch)
                row.update(
                    derive_patch_georef(
                        top_path=rgb_path,
                        x0=x0,
                        y0=y0,
                        fallback_index=saved_count,
                        source_crs=source_crs,
                    )
                )
                row.update(
                    {
                        "source_key": key,
                        "source_rgb": str(rgb_path),
                        "source_dsm": str(dsm_path),
                        "source_label": str(label_path),
                        "patch_x": x0,
                        "patch_y": y0,
                        "patch_size": patch_size,
                        "class_mode": "reduced_4_class",
                    }
                )
                metadata_rows.append(row)
                saved_count += 1

                if max_patches is not None and saved_count >= max_patches:
                    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
                    print(f"Saved {saved_count} patches to {output_dir}")
                    return

    pd.DataFrame(metadata_rows).to_csv(output_dir / "metadata.csv", index=False)
    print(f"Saved {saved_count} patches to {output_dir}")
    print("Reduced class mapping:")
    for class_id, name in REDUCED_CLASS_NAMES.items():
        print(f"  {class_id}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real ISPRS RGB/TOP + DSM + label data for this project.")
    parser.add_argument("--rgb-dir", type=Path, required=True, help="Folder containing ISPRS TOP/RGB TIFF files.")
    parser.add_argument("--dsm-dir", type=Path, required=True, help="Folder containing DSM or normalised DSM TIFF files.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Folder containing colour-coded label/ground-truth TIFF files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/isprs_processed"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument(
        "--source-crs",
        type=str,
        default=None,
        help="Source CRS for world-file coordinates, e.g. EPSG:32633 for Potsdam or EPSG:32632 for Vaihingen. If omitted, demo coordinates are used.",
    )
    args = parser.parse_args()

    prepare_dataset(
        rgb_dir=args.rgb_dir,
        dsm_dir=args.dsm_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        max_patches=args.max_patches,
        min_classes=args.min_classes,
        source_crs=args.source_crs,
    )


if __name__ == "__main__":
    main()
