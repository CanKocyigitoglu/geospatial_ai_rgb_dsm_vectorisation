#!/usr/bin/env bash
set -e

# Edit these folders after downloading/extracting the ISPRS data.
RGB_DIR="data/isprs_raw/potsdam/rgb"
DSM_DIR="data/isprs_raw/potsdam/dsm"
LABEL_DIR="data/isprs_raw/potsdam/labels"

python src/prepare_isprs_dataset.py \
  --rgb-dir "$RGB_DIR" \
  --dsm-dir "$DSM_DIR" \
  --label-dir "$LABEL_DIR" \
  --output-dir data/isprs_processed \
  --patch-size 512 \
  --stride 512 \
  --max-patches 100 \
  --source-crs EPSG:32633

python src/training/train_unet.py \
  --data-dir data/isprs_processed \
  --epochs 10 \
  --batch-size 2 \
  --output-weights outputs/unet_isprs.pt

python src/prediction/predict_with_unet.py \
  --data-dir data/isprs_processed \
  --weights outputs/unet_isprs.pt \
  --output-dir outputs/isprs_predicted_masks

python src/evaluate_segmentation.py \
  --data-dir data/isprs_processed \
  --pred-dir outputs/isprs_predicted_masks \
  --output-csv outputs/isprs_segmentation_metrics.csv

python src/pipeline/run_demo_pipeline.py \
  --data-dir data/isprs_processed \
  --mask-source outputs/isprs_predicted_masks \
  --output-dir outputs/isprs_vectorised

echo "Done. Open Streamlit with:"
echo "streamlit run app/streamlit_map.py"
echo "Then set paths to outputs/isprs_vectorised/asset_vectors.geojson and outputs/isprs_vectorised/feature_table.csv"
