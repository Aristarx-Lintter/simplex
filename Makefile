create_env:
	uv sync

run_training:
	uv run python ./scripts/launcher_cuda_parallel.py --config ./configs/experiment_config_transformer_mess3_bloch_hw_6.yaml

jupyter:
    jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=simplex 