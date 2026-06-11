#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/train_ccc_grpo.sh <agedb|imdb_movie|boneage|imdb_wiki> [exp_name]"
  exit 1
fi

DATASET="$1"
EXP_NAME="${2:-ccc-grpo-${DATASET}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/hf_data}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/rl}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/src/open-r1-multimodal/local_scripts/zero2.json}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12558}"

case "${DATASET}" in
  agedb)
    DATA_FILE="${DATA_ROOT}/agedb/agedb_train.jsonl"
    IMAGE_FOLDER="${DATA_ROOT}/agedb"
    REWARD_FUNCS=(format global_rank)
    NUM_EPOCHS="${NUM_EPOCHS:-4}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    ;;
  imdb_movie)
    DATA_FILE="${DATA_ROOT}/imdb_movie/train.jsonl"
    IMAGE_FOLDER="${DATA_ROOT}/imdb_movie"
    REWARD_FUNCS=(format global_rank)
    NUM_EPOCHS="${NUM_EPOCHS:-4}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    ;;
  boneage)
    DATA_FILE="${DATA_ROOT}/boneage/boneage_train.jsonl"
    IMAGE_FOLDER="${DATA_ROOT}/boneage"
    REWARD_FUNCS=(format-boneage global_rank_boneage)
    NUM_EPOCHS="${NUM_EPOCHS:-4}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    ;;
  imdb_wiki)
    DATA_FILE="${DATA_ROOT}/imdb_wiki/imdb_train_peak_compressed_3500_leq100.jsonl"
    IMAGE_FOLDER="${DATA_ROOT}/imdb_wiki"
    REWARD_FUNCS=(format global_rank)
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-200}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}"
    exit 1
    ;;
esac

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/${EXP_NAME}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="0" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
  --use_vllm False \
  --output_dir "${OUTPUT_ROOT}/${EXP_NAME}" \
  --resume_from_checkpoint False \
  --model_name_or_path "${MODEL_PATH}" \
  --data_file_paths "${DATA_FILE}" \
  --image_folders "${IMAGE_FOLDER}" \
  --is_reward_customized_from_vlm_module True \
  --task_type reg \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing True \
  --logging_steps 1 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --bf16 \
  --attn_implementation flash_attention_2 \
  --run_name "${EXP_NAME}" \
  --data_seed 42 \
  --save_steps "${SAVE_STEPS}" \
  --num_generations 4 \
  --max_completion_length 32 \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --beta 0.04 \
  --report_to wandb \
  --dataset-name released_dir_benchmark \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --learning_rate 1e-5 \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_task_type CAUSAL_LM \
  --freeze_vision_modules True
