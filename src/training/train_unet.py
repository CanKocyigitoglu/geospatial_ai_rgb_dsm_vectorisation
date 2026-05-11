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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.simple_unet import SimpleUNet


class RGBDSMDataset(Dataset):
    def __init__(self, data_dir: Path, tile_ids: list[str]):
        self.data_dir = data_dir
        self.tile_ids = tile_ids

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        rgb = np.array(Image.open(self.data_dir / "rgb" / f"{tile_id}_rgb.png")).astype(np.float32) / 255.0
        dsm = np.load(self.data_dir / "dsm" / f"{tile_id}_dsm.npy").astype(np.float32)
        mask = np.array(Image.open(self.data_dir / "masks" / f"{tile_id}_mask.png")).astype(np.int64)
        dsm = (dsm - dsm.mean()) / (dsm.std() + 1e-6)
        x = np.concatenate([rgb, dsm[..., None]], axis=-1)
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.from_numpy(mask).long()


def pixel_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train(data_dir: Path, output_weights: Path, epochs: int = 5, batch_size: int = 2, lr: float = 1e-3):
    metadata = pd.read_csv(data_dir / "metadata.csv")
    dataset = RGBDSMDataset(data_dir, metadata["tile_id"].tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet(in_channels=4, num_classes=4, base_channels=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device} with {len(dataset)} tiles.")
    for epoch in range(1, epochs + 1):
        losses, accs = [], []
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            accs.append(pixel_accuracy(logits.detach(), y))
        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | pixel_acc={np.mean(accs):.4f}")

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_weights)
    print(f"Saved weights to: {output_weights}")


def main():
    parser = argparse.ArgumentParser(description="Train a small U-Net on RGB + DSM tiles.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-weights", type=Path, default=Path("outputs/unet_synthetic.pt"))
    args = parser.parse_args()
    train(args.data_dir, args.output_weights, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
