export REPO_HOME="/ssong/250010214/MLLM_regression/VLM-R1"
echo "REPO_HOME: $REPO_HOME"







data_paths="/ssong/250010214/MLLM_regression/dataset/imdb-wiki-dir/data/imdb_train_peak_compressed_3500_leq100.jsonl"
image_folders="/ssong/250010214/MLLM_regression/dataset/imdb-wiki-dir/data"










model_path="/ssong/250010214/checkpoint/Qwen2.5-VL-7B-Instruct"

is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"











export EXP_NAME="Qwen2.5-VL-7B-Instruct-reg-lora-IMDB-preciserank-samplemean-4generation-bz16-global-ccc-only-1epoch-compression-3500-lep100"





TASK_TYPE="reg"
cd ${REPO_HOME}

export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12559" \
  src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir /ssong/share/sss_weights/vlm-r1/${EXP_NAME} \
    --resume_from_checkpoint False \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 200 \
    --num_generations 4 \
    --max_completion_length 32 \
    --reward_funcs format global_rank \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed /ssong/250010214/MLLM_regression/VLM-R1/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules True

echo "Training completed for ${EXP_NAME}"
