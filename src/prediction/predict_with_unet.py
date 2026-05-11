from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from src.models.simple_unet import SimpleUNet


def load_input(data_dir: Path, tile_id: str):
    rgb = np.array(Image.open(data_dir / "rgb" / f"{tile_id}_rgb.png")).astype(np.float32) / 255.0
    dsm = np.load(data_dir / "dsm" / f"{tile_id}_dsm.npy").astype(np.float32)
    dsm = (dsm - dsm.mean()) / (dsm.std() + 1e-6)
    x = np.concatenate([rgb, dsm[..., None]], axis=-1)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def predict(data_dir: Path, weights: Path, output_dir: Path):
    metadata = pd.read_csv(data_dir / "metadata.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=4, num_classes=4, base_channels=16).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    for tile_id in metadata["tile_id"].tolist():
        x = load_input(data_dir, tile_id).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(x), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(output_dir / f"{tile_id}_pred_mask.png")

    print(f"Saved predicted masks to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Predict semantic masks using a trained U-Net.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predicted_masks"))
    args = parser.parse_args()
    predict(args.data_dir, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
