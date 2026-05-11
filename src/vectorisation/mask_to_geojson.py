from __future__ import annotations

import cv2
import numpy as np

from src.utils.geojson_utils import pixel_to_geo


CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "vegetation",
    3: "road_or_impervious",
}


def component_polygons_to_features(
    mask: np.ndarray,
    class_id: int,
    tile_id: str,
    origin_lon: float,
    origin_lat: float,
    pixel_size_lon: float,
    pixel_size_lat: float,
    min_area: int = 50,
) -> list[dict]:
    """
    Convert connected components of a class mask into GeoJSON polygon features.
    """
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    features = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 4:
            continue

        # Larger values simplify more aggressively. Keep modest for building edges.
        epsilon = 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
        coords = []

        for point in approx.reshape(-1, 2):
            x, y = float(point[0]), float(point[1])
            lon, lat = pixel_to_geo(x, y, origin_lon, origin_lat, pixel_size_lon, pixel_size_lat)
            coords.append([lon, lat])

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "tile_id": tile_id,
                    "class_id": int(class_id),
                    "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                    "component_id": int(label_id),
                    "pixel_area": area,
                    "vectorisation_method": "opencv_contour_to_geojson",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        )

    return features
