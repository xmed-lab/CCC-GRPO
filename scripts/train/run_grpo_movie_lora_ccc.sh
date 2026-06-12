# PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="/ssong/250010214/MLLM_regression/VLM-R1"
# export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
# data_paths="/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcoco_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocop_train.jsonl:/training/shz/dataset/vlm-r1/rec_jsonsl_train/refcocog_train.jsonl" 
# image_folders="/training/shz/dataset/coco:/training/shz/dataset/coco:/training/shz/dataset/coco"


# ===========================this is AgeDB Dataset==============================
# image_folders="/ssong/250010214/MLLM_regression/dataset/agedb-dir/data/AgeDB"
# data_paths="/ssong/250010214/MLLM_regression/dataset/agedb-dir/data/agedb_train.jsonl"


# ===========================this is Movie Dataset==============================
image_folders="/ssong/250010214/MLLM_regression/dataset/poster_score/poster_downloads"
data_paths="/ssong/250010214/MLLM_regression/dataset/poster_score/train.jsonl"



# ===========================this is IMDB-WIKI Dataset==============================
# ---------------------------------------------using this data jsonl--------------------------------------
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3500_leq100.jsonl"
# image_folders="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data"
# ---------------------------------------------------------------------------------------------------------

# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1over3.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_leq100.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1of10.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_1of5.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_2500_leq100.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_leq100.jsonl"
# data_paths="/home/ydubf/imbalanced-regression/imdb-wiki-dir/data/imdb_train_peak_compressed_3000_linear_leq100.jsonl"


# ===========================this is BoneAge Dataset==============================
# ---------------------------------------------using this data jsonl--------------------------------------
# data_paths="/home/ydubf/imbalanced-regression/BoneAge/boneage_train.jsonl"
# image_folders="/home/ydubf/imbalanced-regression/BoneAge/boneage_resize"
# ---------------------------------------------------------------------------------------------------------

# data_paths="/home/ydubf/imbalanced-regression/BoneAge/boneage_train_peakfilter.jsonl"



# data_paths="/home/ydubf/imbalanced-regression/EchoNet-LVH-Keyframe/train.jsonl"
# image_folders="/home/ydubf/imbalanced-regression/EchoNet-LVH-Keyframe/Key_frames"



# model_path="Qwen/Qwen2.5-VL-3B-Instruct"
model_path="/ssong/250010214/checkpoint/Qwen2.5-VL-7B-Instruct"

is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"


# TODO: change this to your own experiment name

# ===========================this is Movie Dataset totally four setting. CCC, Pure MAE, Pure VisualQuality, and Pure MAE Reweightingt, for other dataset, pls follow the same setting==============================
# try this first, Movie, BoneAge and AgeDB are fast training, IMDB-WIKI has many data, slow.
# For Movie, BoneAge and AgeDB, 4 epoch, 100 step save is ok.   For IMDB-WIKI, 1 epoch with 200 step save.

# export EXP_NAME="Qwen2.5-VL-7B-Instruct-reg-lora-Movie-preciserank-samplemean-4generation-bz16-global-ccc-only-4generation-bz16-4epoch"
export EXP_NAME="Qwen2.5-VL-7B-Instruct-reg-lora-Movie-preciserank-samplemean-4generation-bz24-global-ccc-only-4generation-bz24-6epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-Movie-pureregression-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-Movie-purevisualquality-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-Movie-pureregression-reweighting-4generation-bz16-4epoch"


# ===========================this is BoneAge Dataset==============================
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-BoneAge-preciserank-samplemean-4generation-bz16-global-ccc-only-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-BoneAge-pureregression-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-BoneAge-pureregression-reweighting-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-BoneAge-pureregression-reweighting-small-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-BoneAge-purevisualquality-4generation-bz16-4epoch"


# ===========================this is AgeDB Dataset==============================
# export EXP_NAME="Qwen2.5-VL-7B-Instruct-reg-lora-AgeDB-preciserank-samplemean-4generation-bz16-global-ccc-only-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-AgeDB-pureregression-reweighting-4generation-bz16-4epoch"



# ===========================this is IMDB-WIKI Dataset==============================
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-IMDB-preciserank-samplemean-4generation-bz16-global-ccc-only-1epoch-compression-3500-lep100"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-IMDB-pureregression-reweighting-4generation-bz16-4epoch"



# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-EchoIVS-preciserank-samplemean-4generation-bz16-global-ccc-only-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-EchoIVS-preciserank-samplemean-4generation-bz16-global-ccc-only-withvision-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-EchoIVS-preciserank-samplemean-4generation-bz16-global-ccc-only-with-regression-withvision-4generation-bz16-4epoch"
# export EXP_NAME="Qwen2.5-VL-3B-Instruct-reg-lora-EchoIVS-pureregression-4generation-bz16-4epoch"

# 如果reward 组合没有效果， 可能就是self.vlm_module.post_model_init(self.ref_model, processing_class)这一句comment的原因。

TASK_TYPE="reg"
cd ${REPO_HOME}

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_P2P_DISABLE=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1

# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=3
torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12560" \
  src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir /ssong/share/sss_weights/vlm-r1/${EXP_NAME} \
    --resume_from_checkpoint False \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --num_train_epochs 6 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
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
