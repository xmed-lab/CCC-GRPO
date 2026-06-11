#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: bash scripts/eval_ccc_grpo.sh <agedb|imdb_movie|boneage|imdb_wiki> <model_path> <step>"
  exit 1
fi

DATASET="$1"
MODEL_PATH="$2"
STEP="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/hf_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${MODEL_PATH}")")}"

case "${DATASET}" in
  agedb)
    SUBSET_ROOT="${DATA_ROOT}/agedb"
    DATA_FILE="${SUBSET_ROOT}/test_conversation_from_agedb.json"
    ;;
  imdb_movie)
    SUBSET_ROOT="${DATA_ROOT}/imdb_movie"
    DATA_FILE="${SUBSET_ROOT}/test.json"
    ;;
  boneage)
    SUBSET_ROOT="${DATA_ROOT}/boneage"
    DATA_FILE="${SUBSET_ROOT}/test_conversation_from_boneage.json"
    ;;
  imdb_wiki)
    SUBSET_ROOT="${DATA_ROOT}/imdb_wiki"
    DATA_FILE="${SUBSET_ROOT}/test_conversation_from_imdb_leq100.json"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}"
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
cd "${SUBSET_ROOT}"

python "${REPO_ROOT}/src/eval/run_eval_single_step.py" \
  --step "${STEP}" \
  --run_name "${RUN_NAME}" \
  --dataset "${DATASET}" \
  --data_file "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}"
