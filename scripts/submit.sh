#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="${RUN_NAME:-bsct_16k_base}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml/bin/python}"
NGPUS="${NGPUS:-1}"
PREPROCESS_IN_JOB="${PREPROCESS_IN_JOB:-1}"
PREPROCESS_PROJECT="${PREPROCESS_PROJECT:-$HOME/AudioPrism_HPC}"
RAW_DIR="${RAW_DIR:-${PREPROCESS_PROJECT}/data/extracted/V2}"
SPLIT_DIR="${SPLIT_DIR:-${PREPROCESS_PROJECT}/splits}"
TRAIN_DIR="${TRAIN_DIR:-job_scratch_train}"
VAL_DIR="${VAL_DIR:-job_scratch_val}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ "${PREPROCESS_IN_JOB}" == "1" ]]; then
  for required in \
    "${PREPROCESS_PROJECT}/scripts/preprocess_dataset.py" \
    "${RAW_DIR}" \
    "${SPLIT_DIR}/train.txt" \
    "${SPLIT_DIR}/val.txt"; do
    if [[ ! -e "${required}" ]]; then
      echo "Required preprocessing input does not exist: ${required}" >&2
      exit 3
    fi
  done
else
  for metadata in "${TRAIN_DIR}/dataset_metadata.pt" "${VAL_DIR}/dataset_metadata.pt"; do
    if [[ ! -f "${metadata}" ]]; then
      echo "Required dataset metadata does not exist: ${metadata}" >&2
      exit 4
    fi
  done
fi

qsub -v \
RUN_NAME="${RUN_NAME}",\
CONFIG="${CONFIG}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
PYTHON_BIN="${PYTHON_BIN}",\
NGPUS="${NGPUS}",\
PREPROCESS_IN_JOB="${PREPROCESS_IN_JOB}",\
PREPROCESS_PROJECT="${PREPROCESS_PROJECT}",\
RAW_DIR="${RAW_DIR}",\
SPLIT_DIR="${SPLIT_DIR}",\
TRAIN_DIR="${TRAIN_DIR}",\
VAL_DIR="${VAL_DIR}" \
"${PROJECT_ROOT}/scripts/train.pbs"
