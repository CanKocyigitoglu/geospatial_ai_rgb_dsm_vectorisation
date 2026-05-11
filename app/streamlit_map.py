from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


DEFAULT_GEOJSON = Path("outputs/demo/asset_vectors.geojson")
DEFAULT_FEATURES = Path("outputs/demo/feature_table.csv")
DEFAULT_PREVIEW = Path("outputs/demo/preview_overlay.png")

CLASS_COLOURS = {"building": "#d9a441", "tree": "#2e8b57", "road": "#808080"}


def load_geojson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def geojson_center(geojson):
    lons, lats = [], []
    for feature in geojson.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        for lon, lat in coords[0]:
            lons.append(lon)
            lats.append(lat)
    if not lons:
        return [51.520, -1.350]
    return [sum(lats) / len(lats), sum(lons) / len(lons)]


st.set_page_config(page_title="Geospatial AI RGB + DSM Vectorisation", layout="wide")
st.title("Geospatial AI Asset Mapping: RGB + DSM Segmentation to Vector GIS")
st.caption("Raster segmentation masks converted into GIS-ready vector polygons with height and shape features.")

geojson_path = Path(st.sidebar.text_input("GeoJSON path", str(DEFAULT_GEOJSON)))
features_path = Path(st.sidebar.text_input("Feature CSV path", str(DEFAULT_FEATURES)))

if not geojson_path.exists() or not features_path.exists():
    st.warning("Demo outputs not found. Run: bash scripts/run_demo.sh")
    st.stop()

geojson = load_geojson(geojson_path)
features_df = pd.read_csv(features_path)

left, right = st.columns([2, 1])

with left:
    m = folium.Map(location=geojson_center(geojson), zoom_start=15, tiles="OpenStreetMap")

    def style_function(feature):
        class_name = feature.get("properties", {}).get("class_name", "unknown")
        return {
            "fillColor": CLASS_COLOURS.get(class_name, "#3388ff"),
            "color": CLASS_COLOURS.get(class_name, "#3388ff"),
            "weight": 1,
            "fillOpacity": 0.45,
        }

    folium.GeoJson(
        geojson,
        name="Vectorised objects",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["class_name", "tile_id", "pixel_area"],
            aliases=["Class", "Tile", "Area px"],
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=950, height=600)

with right:
    st.subheader("Project outputs")
    st.metric("Vector polygons", len(geojson.get("features", [])))
    st.metric("Object features", len(features_df))
    st.write("Objects by class")
    st.dataframe(features_df["class_name"].value_counts().rename_axis("class").reset_index(name="count"))

    if DEFAULT_PREVIEW.exists():
        st.image(str(DEFAULT_PREVIEW), caption="RGB image with segmentation overlay")

st.subheader("Feature table")
show_cols = [
    "tile_id",
    "class_name",
    "pixel_area",
    "mean_height",
    "max_height",
    "compactness",
    "bbox_aspect_ratio",
    "centroid_lon",
    "centroid_lat",
]
st.dataframe(features_df[[c for c in show_cols if c in features_df.columns]], use_container_width=True)

st.download_button(
    "Download GeoJSON",
    data=geojson_path.read_bytes(),
    file_name="asset_vectors.geojson",
    mime="application/geo+json",
)
