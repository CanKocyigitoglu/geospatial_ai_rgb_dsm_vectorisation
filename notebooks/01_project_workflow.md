# Notebook 01 — Project Workflow

The project follows this workflow:

```text
Aerial RGB tile + DSM height map
        ↓
CNN-based semantic segmentation
        ↓
segmentation mask
        ↓
object-level feature extraction
        ↓
raster-to-vector conversion
        ↓
GeoJSON GIS layer
        ↓
Streamlit/Folium or QGIS review
```

Evidence to collect:

- screenshot of RGB + mask overlay
- feature table
- GeoJSON opened in the map
- U-Net training logs
- IoU / F1 table
