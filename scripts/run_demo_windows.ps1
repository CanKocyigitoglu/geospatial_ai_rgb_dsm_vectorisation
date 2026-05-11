python src/generate_synthetic_dataset.py --output-dir data/demo --num-tiles 6 --size 512
python src/pipeline/run_demo_pipeline.py --data-dir data/demo --output-dir outputs/demo

Write-Host ""
Write-Host "Demo complete."
Write-Host "Open the map with:"
Write-Host "streamlit run app/streamlit_map.py"
