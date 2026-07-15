#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${TRAIN_DIR:?Export TRAIN_DIR=/path/to/preprocessed_train}"
: "${VAL_DIR:?Export VAL_DIR=/path/to/preprocessed_val}"

RUN_NAME="${RUN_NAME:-bsct_16k_base}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml/bin/python}"
NGPUS="${NGPUS:-1}"

qsub -v \
RUN_NAME="${RUN_NAME}",\
CONFIG="${CONFIG}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
TRAIN_DIR="${TRAIN_DIR}",\
VAL_DIR="${VAL_DIR}",\
PYTHON_BIN="${PYTHON_BIN}",\
NGPUS="${NGPUS}" \
"${PROJECT_ROOT}/scripts/train.pbs"
