#!/usr/bin/env bash

# ---------------------
# 1) ENV SETUP
# ---------------------

# (Optional) Create and activate a new virtual environment:
#   python3 -m venv venv
#   source venv/bin/activate

# Upgrade pip if needed
python3 -m pip install --upgrade pip

# Install necessary packages
pip install accelerate transformers datasets evaluate matplotlib
# Install mergekit from source
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
cd ..

# ---------------------
# 2) CREATE MERGE CONFIG
# ---------------------

# This YAML uses a linear averaging of the 3 checkpoints
# You can tweak alpha values for weighted merges
cat <<EOT > merge-config.yml
merge_method: linear
models:
  - path: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    alpha: 1
  - path: "Henrychur/MMed-Llama-3-8B"
    alpha: 1
  - path: "TsinghuaC3I/Llama-3-8B-UltraMedical"
    alpha: 1
tokenizer_source: "union"
# If you want a different final dtype, e.g. fp16 or bf16, add:
# dtype: "float16"
EOT

# ---------------------
# 3) RUN MERGE
# ---------------------

# This will create a directory named "merged-model" with the merged weights
mergekit-yaml merge-config.yml ./merged-model --cuda  # or --cpu if you prefer
