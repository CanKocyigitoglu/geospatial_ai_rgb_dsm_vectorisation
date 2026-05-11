from __future__ import annotations

import math
import cv2
import numpy as np


# Reduced ISPRS-compatible classes used in this project.
CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "vegetation",
    3: "road_or_impervious",
}


def _compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float((4 * math.pi * area) / (perimeter ** 2))


def extract_component_features(
    rgb: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    class_id: int,
    min_area: int = 50,
) -> list[dict]:
    """
    Extract object-level features from connected components of a class mask.

    Features include:
    - shape: area, perimeter, bounding box, compactness
    - height: mean, max, std from DSM/nDSM
    - colour: mean RGB/TOP channels
    """
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    rows = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        component_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

        dsm_values = dsm[component_mask == 1]
        rgb_values = rgb[component_mask == 1]

        rows.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "component_id": int(label_id),
                "pixel_area": int(area),
                "perimeter_px": round(perimeter, 3),
                "compactness": round(_compactness(area, perimeter), 4),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_aspect_ratio": round(float(w / max(h, 1)), 4),
                "centroid_x": round(float(cx), 3),
                "centroid_y": round(float(cy), 3),
                "mean_height": round(float(np.mean(dsm_values)), 3),
                "max_height": round(float(np.max(dsm_values)), 3),
                "height_std": round(float(np.std(dsm_values)), 3),
                "mean_r": round(float(np.mean(rgb_values[:, 0])), 3),
                "mean_g": round(float(np.mean(rgb_values[:, 1])), 3),
                "mean_b": round(float(np.mean(rgb_values[:, 2])), 3),
            }
        )

    return rows
