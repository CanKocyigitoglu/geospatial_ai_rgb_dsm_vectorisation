# Notebook 04 — Real ISPRS Data Workflow

This project should use synthetic data only for checking that the code runs. For CV evidence, prepare real ISPRS Potsdam or Vaihingen tiles.

## Recommended first choice

Use **Potsdam** first, because it provides true RGB TIFF files, DSM/nDSM files, same-size 6000 x 6000 tiles, and world files for georeferencing.

## Folder layout

Create this structure manually after downloading/extracting the dataset:

```text
data/isprs_raw/potsdam/
├── rgb/
│   ├── top_potsdam_2_10_RGB.tif
│   └── ...
├── dsm/
│   ├── dsm_potsdam_02_10.tif
│   └── ...
└── labels/
    ├── top_potsdam_2_10_label.tif
    └── ...
```

World files such as `.tfw` can be placed next to the RGB TIFFs if you want real map coordinates in the GeoJSON output.

## Convert real ISPRS tiles into project patches

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

For Vaihingen, use `--source-crs EPSG:32632` if using the world-file coordinates.

## Train the U-Net on real data

```bash
python src/training/train_unet.py \
  --data-dir data/isprs_processed \
  --epochs 10 \
  --batch-size 2 \
  --output-weights outputs/unet_isprs.pt
```

## Predict masks

```bash
python src/prediction/predict_with_unet.py \
  --data-dir data/isprs_processed \
  --weights outputs/unet_isprs.pt \
  --output-dir outputs/isprs_predicted_masks
```

## Evaluate

```bash
python src/evaluate_segmentation.py \
  --data-dir data/isprs_processed \
  --pred-dir outputs/isprs_predicted_masks \
  --output-csv outputs/isprs_segmentation_metrics.csv
```

## Vectorise the CNN predictions

```bash
python src/pipeline/run_demo_pipeline.py \
  --data-dir data/isprs_processed \
  --mask-source outputs/isprs_predicted_masks \
  --output-dir outputs/isprs_vectorised
```

## View results

```bash
streamlit run app/streamlit_map.py
```

Then set the sidebar paths to:

```text
outputs/isprs_vectorised/asset_vectors.geojson
outputs/isprs_vectorised/feature_table.csv
```

## Reduced class mapping

The preparation script uses a reduced four-class setup:

| ISPRS class | Project class |
|---|---|
| Impervious surfaces | road_or_impervious |
| Building | building |
| Low vegetation | vegetation |
| Tree | vegetation |
| Car | background / ignored |
| Clutter/background | background |

This keeps the project focused on asset-mapping style outputs rather than fine-grained benchmark competition performance.
