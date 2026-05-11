# Geospatial AI Asset Mapping from RGB Imagery and DSM Height Data

This project demonstrates a small geospatial AI workflow:

- aerial RGB imagery
- LiDAR-derived DSM / height data
- CNN-based semantic segmentation
- feature extraction
- GIS / mapping
- raster-to-vector conversion
- GeoJSON export

The project is designed around the type of data used in the ISPRS Vaihingen/Potsdam 2D Semantic Labeling datasets: aerial orthophotos plus DSM height information and semantic labels.

## What runs immediately

The repository includes a synthetic demo dataset generator, so you can run the full pipeline without downloading external data first.

The demo creates artificial aerial-style RGB tiles, DSM height maps, and semantic masks for:

- buildings
- trees / high vegetation
- roads / impervious linear surfaces

It then extracts height/shape features and converts the raster masks into GIS-ready vector polygons.

## Quick start: lightweight demo

This version does not require PyTorch.

### Git Bash on Windows

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-demo.txt
bash scripts/run_demo.sh
streamlit run app/streamlit_map.py
```

### PowerShell on Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
.\scripts\run_demo_windows.ps1
streamlit run app\streamlit_map.py
```

## Optional: CNN/U-Net training

Install the full environment:

```bash
pip install -r requirements-full.txt
```

Generate demo data:

```bash
python src/generate_synthetic_dataset.py --output-dir data/demo --num-tiles 12 --size 512
```

Train the small U-Net:

```bash
python src/training/train_unet.py --data-dir data/demo --epochs 5 --output-weights outputs/unet_synthetic.pt
```

Predict masks:

```bash
python src/prediction/predict_with_unet.py --data-dir data/demo --weights outputs/unet_synthetic.pt --output-dir outputs/predicted_masks
```

Vectorise the predicted masks:

```bash
python src/pipeline/run_demo_pipeline.py --data-dir data/demo --mask-source outputs/predicted_masks --output-dir outputs/cnn_vectorised
```

Evaluate predictions:

```bash
python src/evaluate_segmentation.py --data-dir data/demo --pred-dir outputs/predicted_masks --output-csv outputs/segmentation_metrics.csv
```

## Main outputs

```text
outputs/demo/asset_vectors.geojson
outputs/demo/feature_table.csv
outputs/demo/preview_overlay.png
```

You can open `asset_vectors.geojson` in QGIS, ArcGIS, geojson.io, or the included Streamlit/Folium map.

## Real ISPRS data workflow

This updated version includes `src/prepare_isprs_dataset.py`, which converts real ISPRS Potsdam/Vaihingen TOP/RGB + DSM + colour-coded ground-truth TIFF files into the same patch format used by the training and vectorisation pipeline.

Install the additional ISPRS reader dependencies:

```bash
python -m pip install -r requirements-isprs.txt
```

Recommended first dataset: **ISPRS Potsdam**, because it provides true RGB images, DSM/nDSM data, labels for training/internal evaluation, and world files for georeferencing.

Prepare folders like this:

```text
data/isprs_raw/potsdam/
├── rgb/
├── dsm/
└── labels/
```

Run conversion:

```bash
python src/prepare_isprs_dataset.py \
  --rgb-dir data/isprs_raw/potsdam/rgb \
  --dsm-dir data/isprs_raw/potsdam/dsm \
  --label-dir data/isprs_raw/potsdam/labels \
  --output-dir data/isprs_processed \
  --patch-size 512 \
  --stride 512 \
  --max-patches 100 \
  --source-crs EPSG:32633
```

Then train/evaluate/vectorise:

```bash
python src/training/train_unet.py --data-dir data/isprs_processed --epochs 10 --batch-size 2 --output-weights outputs/unet_isprs.pt
python src/prediction/predict_with_unet.py --data-dir data/isprs_processed --weights outputs/unet_isprs.pt --output-dir outputs/isprs_predicted_masks
python src/evaluate_segmentation.py --data-dir data/isprs_processed --pred-dir outputs/isprs_predicted_masks --output-csv outputs/isprs_segmentation_metrics.csv
python src/pipeline/run_demo_pipeline.py --data-dir data/isprs_processed --mask-source outputs/isprs_predicted_masks --output-dir outputs/isprs_vectorised
streamlit run app/streamlit_map.py
```

In Streamlit, use:

```text
outputs/isprs_vectorised/asset_vectors.geojson
outputs/isprs_vectorised/feature_table.csv
```

The real-data workflow uses a reduced four-class mapping:

```text
0 = background / ignored
1 = building
2 = vegetation
3 = road_or_impervious
```
