import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import torch.distributed as dist
import argparse
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

parser = argparse.ArgumentParser()

parser.add_argument("--step", type=int, required=True)
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./logs")
parser.add_argument("--model_path", type=str, default=None)

args = parser.parse_args()

STEP = args.step
RUN_NAME = args.run_name
DATASET = args.dataset
DATA_FILE = args.data_file
OUTPUT_DIR = args.output_dir
MODEL_PATH_ARG = args.model_path

def setup_distributed():
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        world_size = 1
        rank = 0
    return local_rank, world_size, rank


local_rank, world_size, rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.device_count() > 1 else "cuda:0")
print(f"Process {rank} using {device}")

BSZ = 16
main_rank = 0


def extract_value_from_text(content):
    patterns = [
        r'年龄[:：]?\s*(\d+\.?\d*)',
        r'Age[:：]?\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)'
    ]
    for pat in patterns:
        match = re.search(pat, content)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


def evaluate_regression():
    if rank == 0:
        print(f"Processing {DATASET}...")

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    QUESTION_TEMPLATE = "{Question} Please output the final answer in <answer> </answer> tags."

    if world_size > 1:
        per_rank_data = len(data) // world_size
        start_idx = rank * per_rank_data
        end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
        rank_data = data[start_idx:end_idx]
    else:
        start_idx, end_idx = 0, len(data)
        rank_data = data

    messages = []
    for x in rank_data:
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": x["image"]},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x["problem"])}
                ]
            }
        ])

    rank_outputs = []
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank):
        batch_messages = messages[i:i + BSZ]
        text = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=256,
            do_sample=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        rank_outputs.extend(batch_output_text)

    if world_size > 1:
        all_outputs = [None] * len(data)
        rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, rank_results)
    else:
        all_outputs = rank_outputs

    if rank == main_rank:
        if world_size > 1:
            for results in gathered_results:
                for idx, output in results:
                    all_outputs[idx] = output

        final_output = []
        y_true, y_pred = [], []

        for input_example, model_output in zip(data, all_outputs):
            match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            prediction = extract_value_from_text(match.group(1)) if match else extract_value_from_text(model_output)
            target = float(input_example["solution"])

            if prediction is not None:
                y_true.append(target)
                y_pred.append(prediction)

            final_output.append({
                "image": input_example["image"],
                "question": input_example["problem"],
                "ground_truth": target,
                "model_output": model_output,
                "prediction": prediction,
                "error": abs(prediction - target) if prediction is not None else None
            })

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"[Step {STEP}] MSE={mse:.3f}, MAE={mae:.3f}")

        with open(OUTPUT_PATH, "w") as f:
            json.dump({
                "mse": mse,
                "mae": mae,
                "results": final_output
            }, f, indent=2)

    if world_size > 1:
        dist.barrier()


def run_evaluation(step):
    global model, tokenizer, processor, OUTPUT_PATH

    print(f"\n=== Evaluating checkpoint-{step} ===")

    MODEL_PATH = MODEL_PATH_ARG or f"/ssong/share/sss_weights/vlm-r1/{RUN_NAME}/checkpoint-{step}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(
        OUTPUT_DIR,
        f"predictions_{DATASET}_{RUN_NAME}_{STEP}.json"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank} if torch.cuda.device_count() > 1 else {"": 0},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    evaluate_regression()


if __name__ == "__main__":
    run_evaluation(STEP)
