#!/bin/sh
source .venv/bin/activate
pip install -e .
pip install regex safetensors tokenizers wandb dotenv 'accelerate>=0.26.0'

python3 tests/models/bert_dim_pos/test_training.py --model_name "test_bert_dim_pos"