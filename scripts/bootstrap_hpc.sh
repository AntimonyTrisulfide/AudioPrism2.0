#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-audioprism2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r requirements-hpc.txt
python -m unittest discover -s tests -v
echo "Environment ${ENV_NAME} is ready. Set PYTHON_BIN=$(which python) when submitting."
