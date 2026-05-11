#!/usr/bin/env bash
set -e

python src/generate_synthetic_dataset.py --output-dir data/demo --num-tiles 6 --size 512
python src/pipeline/run_demo_pipeline.py --data-dir data/demo --output-dir outputs/demo

echo ""
echo "Demo complete."
echo "Open the map with:"
echo "streamlit run app/streamlit_map.py"
