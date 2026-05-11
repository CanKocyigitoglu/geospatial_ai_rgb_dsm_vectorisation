from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.features.object_features import extract_component_features
from src.vectorisation.mask_to_geojson import component_polygons_to_features
from src.utils.geojson_utils import pixel_to_geo, write_geojson


CLASS_COLOURS = {
    0: (0, 0, 0),
    1: (230, 190, 130),
    2: (40, 150, 65),
    3: (120, 120, 120),
}


def colourise_mask(mask):
    colour = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLOURS.items():
        colour[mask == class_id] = rgb
    return colour


def save_preview(rgb, mask, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = cv2.addWeighted(rgb, 0.65, colourise_mask(mask), 0.35, 0)
    Image.fromarray(overlay).save(output_path)


def load_mask(tile_id: str, data_dir: Path, mask_source: Path | None):
    if mask_source is not None:
        for name in [f"{tile_id}_pred_mask.png", f"{tile_id}_mask.png"]:
            path = mask_source / name
            if path.exists():
                return np.array(Image.open(path)).astype(np.uint8)
        raise FileNotFoundError(f"Could not find predicted mask for {tile_id} in {mask_source}")

    return np.array(Image.open(data_dir / "masks" / f"{tile_id}_mask.png")).astype(np.uint8)


def run_pipeline(data_dir: Path, output_dir: Path, mask_source: Path | None = None, min_area: int = 80):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(data_dir / "metadata.csv")

    all_rows = []
    all_features = []

    for idx, meta in metadata.iterrows():
        tile_id = meta["tile_id"]
        rgb = np.array(Image.open(data_dir / "rgb" / f"{tile_id}_rgb.png")).astype(np.uint8)
        dsm = np.load(data_dir / "dsm" / f"{tile_id}_dsm.npy")
        mask = load_mask(tile_id, data_dir, mask_source)

        origin_lon = float(meta["origin_lon"])
        origin_lat = float(meta["origin_lat"])
        pixel_size_lon = float(meta["pixel_size_lon"])
        pixel_size_lat = float(meta["pixel_size_lat"])

        for class_id in [1, 2, 3]:
            rows = extract_component_features(rgb, dsm, mask, class_id=class_id, min_area=min_area)
            for row in rows:
                lon, lat = pixel_to_geo(
                    row["centroid_x"],
                    row["centroid_y"],
                    origin_lon,
                    origin_lat,
                    pixel_size_lon,
                    pixel_size_lat,
                )
                row["tile_id"] = tile_id
                row["centroid_lon"] = lon
                row["centroid_lat"] = lat
                row["source_mask"] = "prediction" if mask_source is not None else "demo_ground_truth"
                all_rows.append(row)

            all_features.extend(
                component_polygons_to_features(
                    mask=mask,
                    class_id=class_id,
                    tile_id=tile_id,
                    origin_lon=origin_lon,
                    origin_lat=origin_lat,
                    pixel_size_lon=pixel_size_lon,
                    pixel_size_lat=pixel_size_lat,
                    min_area=min_area,
                )
            )

        if idx == 0:
            save_preview(rgb, mask, output_dir / "preview_overlay.png")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "feature_table.csv", index=False)
    write_geojson(output_dir / "asset_vectors.geojson", all_features)

    print(f"Saved feature table: {output_dir / 'feature_table.csv'}")
    print(f"Saved vector polygons: {output_dir / 'asset_vectors.geojson'}")
    print(f"Saved preview overlay: {output_dir / 'preview_overlay.png'}")
    print(f"Object count: {len(df)}")
    if not df.empty:
        print(df.groupby("class_name").size())


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction and GIS vectorisation pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--mask-source", type=Path, default=None)
    parser.add_argument("--min-area", type=int, default=80)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, args.mask_source, args.min_area)


if __name__ == "__main__":
    main()
