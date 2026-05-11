from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_geojson(path: Path, features: list[dict[str, Any]], crs_name: str = "EPSG:4326") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def pixel_to_geo(x: float, y: float, origin_lon: float, origin_lat: float, pixel_size_lon: float, pixel_size_lat: float):
    lon = origin_lon + x * pixel_size_lon
    lat = origin_lat - y * pixel_size_lat
    return lon, lat
