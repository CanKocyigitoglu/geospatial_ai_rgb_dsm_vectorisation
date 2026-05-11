from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def _draw_rotated_rectangle(mask, image, dsm, center, size, angle, class_id, colour, height):
    rect = (tuple(center), tuple(size), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], class_id)
    cv2.fillPoly(image, [box], colour)
    cv2.fillPoly(dsm, [box], float(height))


def _draw_tree(mask, image, dsm, center, radius):
    cx, cy = center
    colour = (
        int(random.randint(35, 80)),
        int(random.randint(95, 150)),
        int(random.randint(40, 85)),
    )
    cv2.circle(mask, (cx, cy), radius, 2, -1)
    cv2.circle(image, (cx, cy), radius, colour, -1)

    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    canopy = np.clip(1 - dist / max(radius, 1), 0, 1)
    dsm[:] = np.where(mask == 2, np.maximum(dsm, 4 + 9 * canopy), dsm)


def create_tile(tile_id: int, size: int = 512, seed: int = 42):
    random.seed(seed + tile_id)
    np.random.seed(seed + tile_id)

    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.random.normal(120, 10, (size, size)).clip(0, 255)
    rgb[:, :, 1] = np.random.normal(130, 12, (size, size)).clip(0, 255)
    rgb[:, :, 2] = np.random.normal(120, 10, (size, size)).clip(0, 255)

    dsm = np.random.normal(0.4, 0.15, (size, size)).astype(np.float32).clip(0, None)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Roads: linear low-height objects
    for _ in range(random.randint(2, 4)):
        y = random.randint(50, size - 50)
        thickness = random.randint(18, 38)
        angle = random.choice([0, 0, 10, -10, 25, -25])
        center = (size // 2 + random.randint(-40, 40), y)
        road_len = random.randint(size // 2, size + 100)
        _draw_rotated_rectangle(mask, rgb, dsm, center, (road_len, thickness), angle, 3, (95, 95, 95), 0.2)

    # Buildings: compact high objects
    for _ in range(random.randint(6, 11)):
        w = random.randint(35, 95)
        h = random.randint(30, 100)
        cx = random.randint(w // 2 + 5, size - w // 2 - 5)
        cy = random.randint(h // 2 + 5, size - h // 2 - 5)
        angle = random.choice([0, 0, 5, -5, 15, -15])
        roof_colour = (
            random.randint(135, 190),
            random.randint(120, 170),
            random.randint(105, 150),
        )
        height = random.uniform(5.0, 18.0)
        _draw_rotated_rectangle(mask, rgb, dsm, (cx, cy), (w, h), angle, 1, roof_colour, height)

    # Trees: rounded high vegetation
    for _ in range(random.randint(8, 16)):
        radius = random.randint(12, 34)
        cx = random.randint(radius + 5, size - radius - 5)
        cy = random.randint(radius + 5, size - radius - 5)
        _draw_tree(mask, rgb, dsm, (cx, cy), radius)

    noise = np.random.normal(0, 4, rgb.shape).astype(np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return rgb, dsm, mask


def save_tile(output_dir: Path, tile_id: int, rgb, dsm, mask, origin_lon, origin_lat, pixel_size_deg):
    rgb_dir = output_dir / "rgb"
    dsm_dir = output_dir / "dsm"
    mask_dir = output_dir / "masks"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dsm_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    tile_name = f"tile_{tile_id:03d}"
    Image.fromarray(rgb).save(rgb_dir / f"{tile_name}_rgb.png")
    np.save(dsm_dir / f"{tile_name}_dsm.npy", dsm)
    Image.fromarray(mask).save(mask_dir / f"{tile_name}_mask.png")

    return {
        "tile_id": tile_name,
        "rgb_path": str(rgb_dir / f"{tile_name}_rgb.png"),
        "dsm_path": str(dsm_dir / f"{tile_name}_dsm.npy"),
        "mask_path": str(mask_dir / f"{tile_name}_mask.png"),
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "pixel_size_lon": pixel_size_deg,
        "pixel_size_lat": pixel_size_deg,
        "crs": "EPSG:4326_demo",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic RGB + DSM semantic segmentation dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--num-tiles", type=int, default=6)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    base_lon = -1.3500
    base_lat = 51.5200
    pixel_size_deg = 0.000006

    for i in range(args.num_tiles):
        rgb, dsm, mask = create_tile(i, size=args.size, seed=args.seed)
        metadata.append(
            save_tile(
                args.output_dir,
                i,
                rgb,
                dsm,
                mask,
                origin_lon=base_lon + i * 0.005,
                origin_lat=base_lat + i * 0.002,
                pixel_size_deg=pixel_size_deg,
            )
        )

    pd.DataFrame(metadata).to_csv(args.output_dir / "metadata.csv", index=False)
    print(f"Created synthetic dataset at: {args.output_dir}")
    print(f"Tiles: {args.num_tiles}")


if __name__ == "__main__":
    main()
