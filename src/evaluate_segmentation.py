from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {0: "background", 1: "building", 2: "tree", 3: "road"}


def compute_metrics(y_true, y_pred, num_classes: int = 4):
    rows = []
    for class_id in range(num_classes):
        tp = np.logical_and(y_true == class_id, y_pred == class_id).sum()
        fp = np.logical_and(y_true != class_id, y_pred == class_id).sum()
        fn = np.logical_and(y_true == class_id, y_pred != class_id).sum()
        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted segmentation masks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/predicted_masks"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/segmentation_metrics.csv"))
    args = parser.parse_args()

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    all_metrics = []
    for tile_id in metadata["tile_id"].tolist():
        true_mask = np.array(Image.open(args.data_dir / "masks" / f"{tile_id}_mask.png")).astype(np.uint8)
        pred_mask = np.array(Image.open(args.pred_dir / f"{tile_id}_pred_mask.png")).astype(np.uint8)
        df = compute_metrics(true_mask, pred_mask)
        df["tile_id"] = tile_id
        all_metrics.append(df)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = metrics.groupby(["class_id", "class_name"])[["iou", "precision", "recall", "f1"]].mean().reset_index()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary)
    print(f"Saved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
