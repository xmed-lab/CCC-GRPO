from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import numpy as np

from collections import deque
import json
import os

# ===============================
# Global Memory Bank for CCC
# ===============================
CCC_MEM_PRED = []    # list[float]
CCC_MEM_GT   = []    # list[float]
CCC_MEM_LIFE = []    # list[int]   ← 每个 memory 的剩余生命

# ===============================
# Global Memory Bank for CCC (Hard Samples)
# ===============================
CCC_MEM = deque()     # 不设 maxlen，我们自己按 hard score 控制
CCC_STEP = 0               # reward-level global step
# MEM_WARMUP_STEPS = 1    # ⭐ 你要的阈值：多少步之后启用 memory


# MEM_WARMUP_STEPS = 200   # 建议 >= 1 epoch imdb
MEM_WARMUP_STEPS = 1000   # 建议 >= 1 epoch  boenage



# ===============================this parameter is used in Pure MAE reweighting setting only==========================
AGE_BIN_WEIGHTS_PATH_agedb = os.environ.get(
    "AGE_BIN_WEIGHTS_agedb",
    "/ssong/250010214/MLLM_regression/VLM-R1/age_bin_weights.json"
)

with open(AGE_BIN_WEIGHTS_PATH_agedb, "r") as f:
    AGE_BIN_WEIGHTS_agedb = {
        int(k): float(v) for k, v in json.load(f).items()
    }

AGE_BIN_WEIGHTS_PATH_imdb_wiki = os.environ.get(
    "AGE_BIN_WEIGHTS_imdb_wiki",
    # "/ssong/250010214/MLLM_regression/VLM-R1/age_bin_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/movie_bin_weights.json"
    # "/home/ydubf/VLM-R1/boneage_bin_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/boneage_bin_small_weights.json"
    "/ssong/250010214/MLLM_regression/VLM-R1/imdb-wiki-dir_bin_weights.json"
)

with open(AGE_BIN_WEIGHTS_PATH_imdb_wiki, "r") as f:
    AGE_BIN_WEIGHTS_imdb_wiki = {
        int(k): float(v) for k, v in json.load(f).items()
    }


AGE_BIN_WEIGHTS_PATH_movie = os.environ.get(
    "AGE_BIN_WEIGHTS_movie",
    # "/ssong/250010214/MLLM_regression/VLM-R1/age_bin_weights.json"
    "/ssong/250010214/MLLM_regression/VLM-R1/movie_bin_weights.json"
    # "/home/ydubf/VLM-R1/boneage_bin_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/boneage_bin_small_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/imdb-wiki-dir_bin_weights.json"
)

with open(AGE_BIN_WEIGHTS_PATH_movie, "r") as f:
    AGE_BIN_WEIGHTS_movie = {
        int(k): float(v) for k, v in json.load(f).items()
    }



AGE_BIN_WEIGHTS_PATH_boneage = os.environ.get(
    "AGE_BIN_WEIGHTS_boneage",
    # "/ssong/250010214/MLLM_regression/VLM-R1/age_bin_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/movie_bin_weights.json"
    # "/home/ydubf/VLM-R1/boneage_bin_weights.json"
    "/ssong/250010214/MLLM_regression/VLM-R1/boneage_bin_small_weights.json"
    # "/ssong/250010214/MLLM_regression/VLM-R1/imdb-wiki-dir_bin_weights.json"
)


with open(AGE_BIN_WEIGHTS_PATH_boneage, "r") as f:
    AGE_BIN_WEIGHTS_boneage = {
        int(k): float(v) for k, v in json.load(f).items()
    }


class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()


    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "reg":
                # used for BoneAge
                # SYSTEM_PROMPT = (
                #     # "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                #     "You are a professional pediatric radiologist."
                #     # "You are an experienced echocardiography expert specializing in cardiac structure measurement and ventricular wall thickness assessment."
                # )
                # return SYSTEM_PROMPT + '\n' + "{Question} Please output the final answer in <answer> </answer> tags."
                
                # used for the other three tasks.
                # return "{Question} Please output the final answer in <answer> </answer> tags."

                return "{Question}"
            
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards


    @staticmethod
    def format_reward_fundus(completions, **kwargs):
        """
        Check if the output contains a valid <answer> tag
        with an integer label in [0, 4].
        """
        import re
        import os
        from datetime import datetime

        # 合法 label 范围（整数）
        MIN_LABEL = 0
        MAX_LABEL = 4

        # 只接受整数（不接受小数）
        pattern = r"<answer>\s*(-?\d+)\s*</answer>"

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        # Debug
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH")

        if debug_mode and log_path:
            f = open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8")
            f.write(f"\n------------- {current_time} Format reward -------------\n")

        for content in completion_contents:
            match = re.search(pattern, content, re.DOTALL)

            if not match:
                # ❌ 没有 <answer> 或不是整数
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: missing <answer> or non-integer\n")
                continue

            # 解析整数
            try:
                value = int(match.group(1))
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: parse int error\n")
                continue

            # 范围检查
            if MIN_LABEL <= value <= MAX_LABEL:
                # rewards.append(0.5)   # ✅ 合法格式 + 合法 label
                rewards.append(0.1)   # ✅ 合法格式 + 合法 label
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ VALID: label={value}\n")
            else:
                rewards.append(0.0)   # ❌ 越界
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ INVALID: label={value} out of range\n")

        if debug_mode and log_path:
            f.close()

        return rewards

# 单向reward, 只加在anchor sample
    @staticmethod
    def age_reward(completions, solution, **kwargs):
        import re
        import torch
        import numpy as np
        import os
        from datetime import datetime
        import random

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./age_reward_log.txt")

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)

        # ============================================================
        # 1. Parse GT
        # ============================================================
        gt_vals = []
        for sol in solution:
            try:
                s = re.findall(r"<answer>(.*?)</answer>", sol)[-1]
                v = float(re.findall(r"[-+]?\d+", s)[0])
            except Exception:
                print("Meet Error (GT parse) ...")
                v = random.uniform(1, 100)
            gt_vals.append(v)

        gt_vals_grouped = [gt_vals[i:i + n_gen] for i in range(0, len(gt_vals), n_gen)]
        gt_list = [g[0] for g in gt_vals_grouped]

        # ============================================================
        # 2. Parse predictions
        # ============================================================
        # preds = []
        # for comp in completions:
        #     content = comp[0]["content"]
        #     try:
        #         s = re.findall(r"<answer>(.*?)</answer>", content)[-1]
        #         v = float(re.findall(r"[-+]?\d+", s)[0])
        #     except Exception:
        #         print("Meet Error (Pred parse) ...")
        #         v = random.uniform(1, 100)
        #     preds.append(v)

        preds = []
        for comp in completions:
            content = comp[0]["content"]

            try:
                # 尝试从 <answer>...</answer> 里提取文本
                matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
                answer_text = matches[-1].strip() if matches else content.strip()

                # 使用你更鲁棒的方法：extract_first_number()
                pred = extract_first_number(answer_text)

                # 如果没找到数字，给一个默认 fallback
                if pred is None:
                    print("Pred parse warning: no number found, using fallback.")
                    pred = random.uniform(1, 100)

            except Exception as e:
                print("Pred parse unexpected error:", e)
                pred = random.uniform(1, 100)

            preds.append(pred)

        preds_grouped = [preds[i:i + n_gen] for i in range(0, len(preds), n_gen)]

        # ============================================================
        # 3. flatten
        # ============================================================
        pred_all = []
        gt_all = []

        for k, pg in enumerate(preds_grouped):
            pred_all.extend(pg)
            gt_all.extend([gt_list[k]] * n_gen)

        pred_all = torch.tensor(pred_all, device=device)
        gt_all = torch.tensor(gt_all, device=device)

        K = len(pred_all)

        # ============================================================
        # 4. mean prediction per sample
        # ============================================================
        sample_pred_means = []
        for pg in preds_grouped:
            t = torch.tensor(pg, device=device)
            sample_pred_means.append(t.mean())

        sample_pred_means = torch.stack(sample_pred_means)
        sample_gt = torch.tensor(gt_list, device=device)

        # ============================================================
        # 5. Ranking reward
        # ============================================================
        rewards = []
        idx_global = 0

        for samp_idx in range(len(preds_grouped)):

            r_inter_list = []

            for j in range(n_gen):
                idx = idx_global

                pred_i = pred_all[idx]
                gt_i = gt_all[idx]
                sample_i = idx // n_gen

                fidelity_sum = 0
                cnt = 0

                # ====== 与其他 sample 的 mean 对比 ======
                for other_sample in range(len(preds_grouped)):
                    if other_sample == sample_i:
                        continue  # 跳过同一个 sample

                    gt_o = sample_gt[other_sample]
                    pred_o_mean = sample_pred_means[other_sample]

                    # ---- Step 1: 方向一致性检查 ----
                    diff_gt = gt_i - gt_o
                    diff_pred = pred_i - pred_o_mean

                    sign_gt = torch.sign(diff_gt)
                    sign_pred = torch.sign(diff_pred)

                    # 如果方向不一致 reward = 0
                    if sign_gt != 0 and sign_pred != 0 and sign_gt != sign_pred:
                        fidelity_sum += 0.0
                        cnt += 1

                        # ===== Debug log =====
                        if DEBUG:
                            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                            with open(log_path.replace(".txt","_rank.txt"), "a") as f:
                                f.write(
                                    f"[DirWrong] sample {samp_idx} gen {j} vs sample_mean {other_sample} | "
                                    f"GT_i={gt_i.item():.1f}, GT_o={gt_o.item():.1f}, diff_gt={diff_gt.item():.1f}, sign_gt={sign_gt} "
                                    f"Pred_i={pred_i.item():.1f}, Pred_mean_o={pred_o_mean.item():.1f}, diff_pred={diff_pred.item():.1f}, sign_pred={sign_pred} "
                                    f"reward_pair=0.000\n"
                                )
                        continue

                    # ---- Step 2: 幅度一致性 ----
                    diff_gt_abs = torch.abs(diff_gt)
                    diff_pred_abs = torch.abs(diff_pred)

                    rel_error = torch.abs(diff_pred_abs - diff_gt_abs)

                    reward_pair = torch.clamp(1.0 - rel_error * 0.1, min=0.0, max=1.0)

                    # reward_pair = reward_pair * 0.5

                    fidelity_sum += reward_pair
                    cnt += 1


                    # ===== Debug log =====
                    if DEBUG:
                        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                        with open(log_path.replace(".txt","_rank.txt"), "a") as f:
                            f.write(
                                f"[Compare] sample {samp_idx} gen {j} vs sample_mean {other_sample} | "
                                f"GT_i={gt_i.item():.1f}, GT_o={gt_o.item():.1f}, "
                                f"Pred_i={pred_i.item():.1f}, Pred_mean_o={pred_o_mean.item():.1f}, "
                                f"rel_error={rel_error.item():.2f}, reward_pair={reward_pair.item():.3f}\n"
                            )
                with open(log_path.replace(".txt","_rank.txt"), "a") as f:
                    f.write("=================================================\n\n")

                if cnt == 0:
                    r_inter = torch.tensor(0.0, device=device)
                else:
                    r_inter = fidelity_sum / cnt

                if not isinstance(r_inter, torch.Tensor):
                    r_inter = torch.tensor(r_inter, dtype=torch.float32, device=device)

                r_inter_list.append(r_inter)
                idx_global += 1

            r_inter_tensor = torch.stack(r_inter_list)
            r_final = r_inter_tensor
            rewards.extend(r_final.tolist())

            # ============================================================
            # Debug Logging
            # ============================================================
            # if DEBUG:
            #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

            #     with open(log_path.replace(".txt", "_rank.txt"), "a", encoding="utf-8") as f:
            #         for j in range(n_gen):
            #             idx = samp_idx * n_gen + j
            #             pred_i = float(pred_all[idx].item())
            #             gt_i = float(gt_all[idx].item())
            #             sample_i = idx // n_gen

            #             f.write("\n")
            #             f.write(f"================ {current_time} =================\n")
            #             f.write(f"### Sample {samp_idx}, Generation {j}, GlobalIdx={idx}\n")
            #             f.write(f"GT_i = {gt_i}, Pred_i = {pred_i}\n")
            #             f.write(f"Final Reward: {r_final[j].item():.4f}\n")
            #             f.write(f"Fidelity Mean-Rank Avg = {r_inter_list[j].item():.4f}\n")
            #             f.write("---- Pairwise Comparisons (with other sample means) ----\n")

            #             for other_sample in range(len(preds_grouped)):
            #                 if other_sample == sample_i:
            #                     continue

            #                 gt_o = float(sample_gt[other_sample].item())
            #                 pred_o_mean = float(sample_pred_means[other_sample].item())

            #                 diff_gt = gt_i - gt_o
            #                 diff_pred = pred_i - pred_o_mean

            #                 sign_gt = torch.sign(torch.tensor(diff_gt)).item()
            #                 sign_pred = torch.sign(torch.tensor(diff_pred)).item()
            #                 direction_ok = (sign_gt == 0 or sign_pred == 0 or sign_gt == sign_pred)

            #                 diff_gt_abs = abs(diff_gt)
            #                 diff_pred_abs = abs(diff_pred)
            #                 rel_error = abs(diff_pred_abs - diff_gt_abs)
            #                 reward_pair = max(0.0, 1.0 - rel_error * 0.1)

            #                 f.write(
            #                     f"[Compare sample {sample_i} gen {j} vs sample_mean {other_sample}] "
            #                     f"GT_i={gt_i:.1f}, GT_o={gt_o:.1f}, diff_gt={diff_gt:.1f}, sign_gt={sign_gt}; "
            #                     f"Pred_i={pred_i:.1f}, Pred_mean_o={pred_o_mean:.1f}, diff_pred={diff_pred:.1f}, sign_pred={sign_pred}; "
            #                     f"DirOK={direction_ok}; rel_error={rel_error:.2f}; reward_pair={reward_pair:.3f}\n"
            #                 )

            #             f.write("=================================================\n\n")

        return rewards

    @staticmethod
    def age_reward_diff_ccc(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime

        # --------------------------------------------------
        # Load configs
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_diffccc.txt")

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------
        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT (same as CCC)
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        # GT for each sample
        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var = [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)
            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])

        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])

        # --------------------------------------------------
        # Difference-Structure CCC Reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):               # for each sample
            for j in range(n_gen):               # for each generation

                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # Build lists
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                # ----------- NEW: Difference-Structure Vector ----------
                # Using your current pred_list and gt_list_cmp (do NOT change previous logic)

                pred_arr = np.array(pred_list, dtype=np.float32)
                gt_arr   = np.array(gt_list_cmp, dtype=np.float32)

                # Pairwise difference matrices
                # Difference vectors (anchor vs each other sample)
                pred_diff = (pred_arr - pred_i_j).astype(np.float32)
                gt_diff   = (gt_arr  - gt_i).astype(np.float32)

                # 去掉自己减自己的 anchor 点
                pred_diff = pred_diff[1:]
                gt_diff   = gt_diff[1:]


                # Convert to numpy
                x = np.array(pred_diff, dtype=np.float32)
                y = np.array(gt_diff, dtype=np.float32)




                # Means, variances, covariance
                mu_x = x.mean()
                mu_y = y.mean()
                var_x = x.var()
                var_y = y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                # CCC computation
                denom = var_x + var_y + (mu_x - mu_y) ** 2
                if denom == 0:
                    ccc = 0.0
                else:
                    ccc = (2 * cov_xy) / denom

                if np.isnan(ccc) or np.isinf(ccc):
                    ccc = 0.0

                # reward = ccc * 0.5
                reward = ccc

                # print("reward", reward)
                rewards.append(float(reward))

                # Debug logging
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ Diff-Structure CCC =================\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"Pred_ij = {pred_i_j}, GT_i = {gt_i}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list  = {gt_list_cmp}\n")
                        f.write(f"Pred diff list = {pred_diff}\n")
                        f.write(f"GT diff list  = {gt_diff}\n")
                        f.write(f"CCC = {ccc:.4f}\n")
                        f.write("======================================================\n")


        return rewards


# for debug, and re design this module.  =====================
    @staticmethod
    def age_reward_global(completions, solution, **kwargs):
        import re
        import torch
        import numpy as np
        import os
        from datetime import datetime
        import random
        from scipy.stats import spearmanr

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        # DEBUG = (os.getenv("DEBUG_MODE") == "true")
        # log_path = os.getenv("LOG_PATH", "./age_reward_global_log.txt")
        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_spearman.txt")


        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 5)

        def fidelity_reward(pred1, pred2, var1, var2, gt, device):
            esp = 1e-6
            try:
                normal_dist = torch.distributions.Normal(0, 1)
                _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
                p = normal_dist.cdf(_cur)
            except:
                print("Meet Error ...")
                p = torch.tensor(0.5, dtype=torch.float32, device=device)
            
            reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
            return reward

        # ============================================================
        # 1. Parse GT
        # ============================================================
        # gt_vals = []
        # for sol in solution:
        #     try:
        #         s = re.findall(r"<answer>(.*?)</answer>", sol)[-1]
        #         v = float(re.findall(r"[-+]?\d+", s)[0])
        #     except Exception:
        #         print("Meet Error (GT parse) ...")
        #         v = random.uniform(1, 100)
        #     gt_vals.append(v)

        # gt_vals_grouped = [gt_vals[i:i + n_gen] for i in range(0, len(gt_vals), n_gen)]
        # gt_list = [g[0] for g in gt_vals_grouped]

        # print("gt_vals_grouped", gt_vals_grouped)
        # print("gt_list", gt_list)


        # extract solution
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                _cur = reshaped_solution[i][j]
                sol_match = re.search(r'<answer>(.*?)</answer>', _cur)
                ground_truth = sol_match.group(1).strip() if sol_match else sol_match.strip()
                reshaped_solution[i][j] = float(ground_truth)
        
        # print("reshaped_solution", reshaped_solution)



        # ============================================================
        # 2. Parse predictions
        # ============================================================

        # extract content
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        # print("contents",contents)
        # print("reshaped_content", reshaped_content)

        batch_mean, batch_var, batch_pred = [], [], []
        for i in range(len(reshaped_content)): # batch
            cur_pred_list = []
            for j in range(len(reshaped_content[i])): # num generations
                try:
                    content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                    student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                    pred = extract_first_number(student_answer)
                except:
                    print("Meet Error ...")
                    pred = random.uniform(1, 100)
                cur_pred_list.append(pred)
            
            batch_pred.append(cur_pred_list)
            p = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            p_mean = torch.mean(p)
            p_var = torch.var(p)
            batch_mean.append([p_mean])
            batch_var.append([p_var])
        
        # print("batch_mean", batch_mean)
        # print("batch_var", batch_var)
        # print("batch_pred", batch_pred)



        # reshaped_solution [[75.0, 75.0, 75.0, 75.0], [42.0, 42.0, 42.0, 42.0], [62.0, 62.0, 62.0, 62.0], [35.0, 35.0, 35.0, 35.0], [43.0, 43.0, 43.0, 43.0], [60.0, 60.0, 60.0, 60.0]]
        # batch_mean [[tensor(88.)], [tensor(64.7500)], [tensor(71.7500)], [tensor(40.7500)], [tensor(61.5000)], [tensor(53.7500)]]
        # batch_var [[tensor(8.6667)], [tensor(192.2500)], [tensor(297.5833)], [tensor(18.9167)], [tensor(891.)], [tensor(78.9167)]]
        # batch_pred [[85.0, 92.0, 87.0, 88.0], [69.0, 61.0, 48.0, 81.0], [81.0, 82.0, 78.0, 46.0], [43.0, 35.0, 45.0, 40.0], [36.0, 72.0, 39.0, 99.0], [62.0, 60.0, 43.0, 50.0]]


        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])
        rewards = []

        # GT for each sample (only one GT per sample)
        gt_list = [sol[0] for sol in reshaped_solution]  # shape [batch]

        for i in range(batch_size):                      # loop each sample
            for j in range(n_gen):                       # loop each generation

                pred_i_j = batch_pred[i][j]              # 当前 generation prediction
                gt_i = gt_list[i]

                # --- 构建 Pred_list & GT_list ---
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                # --- 排序并计算 Spearman rank ---
                rho, _ = spearmanr(pred_list, gt_list_cmp)

                # Spearman 可能出现 nan（例如 list 全相等），处理掉
                if rho != rho:   # nan
                    rho = 0.0

                # reward = (rho + 1) / 2     # map to 0~1
                reward = rho

                # reward = reward * 0.5

                rewards.append(float(reward))

                # -----------------------------
                # Debug Logging
                # -----------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ Spearman Pair =================\n")
                        f.write(f"Sample i = {i}, Generation j = {j}\n")
                        f.write(f"GT_i = {gt_i}\n")
                        f.write(f"pred_i_j = {pred_i_j}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list   = {gt_list_cmp}\n")
                        f.write(f"Spearman rho = {rho:.4f}\n")
                        f.write("===============================================\n")
        # print(rewards)

        # return rewards

        # rewards = []
        # for i in range(len(batch_pred)):
        #     for j in range(len(batch_pred[i])):
        #         _reward_sum, _count_idx = 0, 0
        #         for z in range(len(batch_mean)):
        #             if z != i:
                        
        #                 input_pred1 = batch_pred[i][j]
        #                 input_pred2 = batch_mean[z][0]
        #                 input_var1 = batch_var[i][0]
        #                 input_var2 = batch_var[z][0]




        #                 if reshaped_solution[i][j] > reshaped_solution[z][0]:
        #                     input_gt = torch.tensor(1.0, dtype=torch.float32, device=device)
        #                 elif reshaped_solution[i][j] < reshaped_solution[z][0]:
        #                     input_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
        #                 else:
        #                     input_gt = torch.tensor(0.5, dtype=torch.float32, device=device)

        #                 _reward = fidelity_reward(
        #                     pred1=input_pred1, pred2=input_pred2, var1=input_var1, 
        #                     var2=input_var2, gt=input_gt, device=device
        #                 )

        #                 _reward_sum = _reward_sum + _reward
        #                 _count_idx = _count_idx + 1

        #         _cur_reward = _reward_sum / _count_idx
        #         rewards.append(_cur_reward)
        
        # print("rewards_rank", rewards)

        # preds = []
        # for comp in completions:
        #     content = comp[0]["content"]
        #     try:
        #         s = re.findall(r"<answer>(.*?)</answer>", content)[-1]
        #         v = float(re.findall(r"[-+]?\d+", s)[0])
        #     except Exception:
        #         print("Meet Error (Pred parse) ...")
        #         v = random.uniform(1, 100)
        #     preds.append(v)

        # preds_grouped = [preds[i:i + n_gen] for i in range(0, len(preds), n_gen)]

        # print("preds_grouped", preds_grouped)

        # # ============================================================
        # # 3. flatten
        # # ============================================================
        # pred_all = []
        # gt_all = []

        # for k, pg in enumerate(preds_grouped):
        #     pred_all.extend(pg)
        #     gt_all.extend([gt_list[k]] * n_gen)

        # pred_all = torch.tensor(pred_all, device=device)
        # gt_all = torch.tensor(gt_all, device=device)

        # K = len(pred_all)

        # print("pred_all", pred_all)
        # print("gt_all", gt_all)
        # print("K", K)

        # ============================================================
        # 4. mean prediction per sample
        # ============================================================
        # sample_pred_means = []
        # for pg in preds_grouped:
        #     t = torch.tensor(pg, device=device)
        #     sample_pred_means.append(t.mean())

        # sample_pred_means = torch.stack(sample_pred_means)
        # sample_gt = torch.tensor(gt_list, device=device)


        # print("sample_pred_means", sample_pred_means)
        # print("sample_gt", sample_gt)


        # ============================================================
        # 5. Ranking reward
        # ============================================================
        # rewards = []
        # idx_global = 0

        # for samp_idx in range(len(preds_grouped)):

        #     r_inter_list = []

        #     for j in range(n_gen):
        #         idx = idx_global

        #         pred_i = pred_all[idx]
        #         gt_i = gt_all[idx]
        #         sample_i = idx // n_gen

        #         fidelity_sum = 0
        #         cnt = 0

        #         # ====== 与其他 sample 的 mean 对比 ======
        #         for other_sample in range(len(preds_grouped)):
        #             if other_sample == sample_i:
        #                 continue  # 跳过同一个 sample

        #             gt_o = sample_gt[other_sample]
        #             pred_o_mean = sample_pred_means[other_sample]

        #             # ---- Step 1: 方向一致性检查 ----
        #             diff_gt = gt_i - gt_o
        #             diff_pred = pred_i - pred_o_mean

        #             sign_gt = torch.sign(diff_gt)
        #             sign_pred = torch.sign(diff_pred)

        #             # 如果方向不一致 reward = 0
        #             if sign_gt != 0 and sign_pred != 0 and sign_gt != sign_pred:
        #                 fidelity_sum += 0.0
        #                 cnt += 1

        #             # if sign_gt != sign_pred:
        #             #     fidelity_sum += 0.0
        #             #     cnt += 1

        #                 # ===== Debug log =====
        #                 if DEBUG:
        #                     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        #                     with open(log_path.replace(".txt","_rank.txt"), "a") as f:
        #                         f.write(
        #                             f"[DirWrong] sample {samp_idx} gen {j} vs sample_mean {other_sample} | "
        #                             f"GT_i={gt_i.item():.1f}, GT_o={gt_o.item():.1f}, diff_gt={diff_gt.item():.1f}, sign_gt={sign_gt} "
        #                             f"Pred_i={pred_i.item():.1f}, Pred_mean_o={pred_o_mean.item():.1f}, diff_pred={diff_pred.item():.1f}, sign_pred={sign_pred} "
        #                             f"reward_pair=0.000\n"
        #                         )
        #                 continue

        #             # ---- Step 2: 幅度一致性 ----
        #             diff_gt_abs = torch.abs(diff_gt)
        #             diff_pred_abs = torch.abs(diff_pred)

        #             rel_error = torch.abs(diff_pred_abs - diff_gt_abs)

        #             reward_pair = torch.clamp(1.0 - rel_error * 0.1, min=0.0, max=1.0)

        #             fidelity_sum += reward_pair
        #             cnt += 1


        #             # ===== Debug log =====
        #             if DEBUG:
        #                 current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        #                 with open(log_path.replace(".txt","_rank.txt"), "a") as f:
        #                     f.write(
        #                         f"[Compare] sample {samp_idx} gen {j} vs sample_mean {other_sample} | "
        #                         f"GT_i={gt_i.item():.1f}, GT_o={gt_o.item():.1f}, "
        #                         f"Pred_i={pred_i.item():.1f}, Pred_mean_o={pred_o_mean.item():.1f}, "
        #                         f"rel_error={rel_error.item():.2f}, reward_pair={reward_pair.item():.3f}\n"
        #                     )
        #         with open(log_path.replace(".txt","_rank.txt"), "a") as f:
        #             f.write("=================================================\n\n")

        #         if cnt == 0:
        #             r_inter = torch.tensor(0.0, device=device)
        #         else:
        #             r_inter = fidelity_sum / cnt

        #         if not isinstance(r_inter, torch.Tensor):
        #             r_inter = torch.tensor(r_inter, dtype=torch.float32, device=device)

        #         r_inter_list.append(r_inter)
        #         idx_global += 1

        #     r_inter_tensor = torch.stack(r_inter_list)
        #     r_final = r_inter_tensor
        #     rewards.extend(r_final.tolist())

            # ============================================================
            # Debug Logging
            # ============================================================
            # if DEBUG:
            #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

            #     with open(log_path.replace(".txt", "_rank.txt"), "a", encoding="utf-8") as f:
            #         for j in range(n_gen):
            #             idx = samp_idx * n_gen + j
            #             pred_i = float(pred_all[idx].item())
            #             gt_i = float(gt_all[idx].item())
            #             sample_i = idx // n_gen

            #             f.write("\n")
            #             f.write(f"================ {current_time} =================\n")
            #             f.write(f"### Sample {samp_idx}, Generation {j}, GlobalIdx={idx}\n")
            #             f.write(f"GT_i = {gt_i}, Pred_i = {pred_i}\n")
            #             f.write(f"Final Reward: {r_final[j].item():.4f}\n")
            #             f.write(f"Fidelity Mean-Rank Avg = {r_inter_list[j].item():.4f}\n")
            #             f.write("---- Pairwise Comparisons (with other sample means) ----\n")

            #             for other_sample in range(len(preds_grouped)):
            #                 if other_sample == sample_i:
            #                     continue

            #                 gt_o = float(sample_gt[other_sample].item())
            #                 pred_o_mean = float(sample_pred_means[other_sample].item())

            #                 diff_gt = gt_i - gt_o
            #                 diff_pred = pred_i - pred_o_mean

            #                 sign_gt = torch.sign(torch.tensor(diff_gt)).item()
            #                 sign_pred = torch.sign(torch.tensor(diff_pred)).item()
            #                 direction_ok = (sign_gt == 0 or sign_pred == 0 or sign_gt == sign_pred)

            #                 diff_gt_abs = abs(diff_gt)
            #                 diff_pred_abs = abs(diff_pred)
            #                 rel_error = abs(diff_pred_abs - diff_gt_abs)
            #                 reward_pair = max(0.0, 1.0 - rel_error * 0.1)

            #                 f.write(
            #                     f"[Compare sample {sample_i} gen {j} vs sample_mean {other_sample}] "
            #                     f"GT_i={gt_i:.1f}, GT_o={gt_o:.1f}, diff_gt={diff_gt:.1f}, sign_gt={sign_gt}; "
            #                     f"Pred_i={pred_i:.1f}, Pred_mean_o={pred_o_mean:.1f}, diff_pred={diff_pred:.1f}, sign_pred={sign_pred}; "
            #                     f"DirOK={direction_ok}; rel_error={rel_error:.2f}; reward_pair={reward_pair:.3f}\n"
            #                 )

            #             f.write("=================================================\n\n")

        return rewards


    @staticmethod
    def age_reward_global_ccc_with_memory(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        import torch

        # -------------------------------
        # Global step & memory
        # -------------------------------
        global CCC_MEM, CCC_STEP
        CCC_STEP += 1

        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)
        device = kwargs.get("device")

        use_memory = CCC_STEP >= MEM_WARMUP_STEPS

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # -------------------------------
        # Helper
        # -------------------------------
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group()), False
            else:
                # return random.uniform(0, 100), True
                return random.uniform(1, 228), True

        # -------------------------------
        # Parse GT
        # -------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        gt_list = [s[0] for s in reshaped_solution]   # one GT per sample

        # -------------------------------
        # Parse predictions
        # -------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var, batch_has_random = [], [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            has_random = False
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred, is_rand = extract_first_number(student_answer)
                cur_pred_list.append(pred)
                if is_rand:
                    has_random = True

            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            # batch_mean.append([t.mean()])
            # batch_var.append([t.var()])
            batch_mean.append(float(t.mean().item()))
            batch_var.append(float(t.var().item()))
            batch_has_random.append(has_random)

        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])
        # -------------------------------
        # Snapshot memory
        # -------------------------------
        mem_snapshot = list(CCC_MEM) if use_memory else []

        # -------------------------------
        # Compute CCC reward
        # -------------------------------
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]

            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]

                # ===== Build comparison set =====
                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # batch anchors
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")

                # memory anchors (tail only)
                if use_memory:
                    for m in mem_snapshot:
                        pred_list.append(m["pred"])
                        gt_list_cmp.append(m["gt"])
                        src_list.append("MEM-TAIL")

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ===== CCC computation =====
                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else float(ccc)

                rewards.append(ccc)

                # ===== DEBUG =====
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n===== AGE CCC + MEMORY DEBUG =====\n")
                        f.write(
                            f"STEP={CCC_STEP}\n"
                            f"Sample i={i}, Gen j={j}\n"
                            f"GT={gt_i:.2f}, pred={pred_i_j:.2f}\n"
                        )
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<10} "
                                f"pred={pred_list[k]:.2f}, gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(
                            f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}, "
                            f"var_x={var_x:.3f}, var_y={var_y:.3f}, "
                            f"CCC={ccc:.4f}\n"
                        )
                        f.write("=================================\n")

        # -------------------------------
        # Update memory (TAIL ONLY)
        # -------------------------------
        if use_memory:
            for i in range(batch_size):
                if batch_has_random[i]:
                    continue

                gt_i = gt_list[i]
                pred_mean_i = batch_mean[i]

                # 合法性
                # if not (0.0 <= gt_i <= 100.0):
                #     continue
                # if not (0.0 <= pred_mean_i <= 100.0):
                #     continue

                # # ⭐ 只存 tail（IMDB 关键）
                # if 20.0 <= gt_i <= 60.0:
                #     continue

                # # 防止极端噪声
                # if abs(pred_mean_i - gt_i) > 10:
                #     continue

                # 合法性
                if not (0.0 <= gt_i <= 228.0):
                    continue
                if not (0.0 <= pred_mean_i <= 228.0):
                    continue

                # ⭐ 只存 tail（IMDB 关键）
                if 50.0 <= gt_i <= 200.0:
                    continue

                # 防止极端噪声
                if abs(pred_mean_i - gt_i) > 15:
                    continue

                CCC_MEM.append({
                    "pred": pred_mean_i,
                    "gt": gt_i
                })

            # FIFO
            CCC_MEM = deque(list(CCC_MEM)[-MAX_MEM_SIZE:])

        return rewards


# version3  ccc generation wij
    @staticmethod
    def age_reward_global_ccc_weighted(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        import torch

        # --------------------------------------------------
        # Config
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen  = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # hyper-params
        ALPHA = kwargs.get("alpha", 8.0)
        TAU   = kwargs.get("tau", 5.0)

        CENTER  = 35.0
        LEFT_L  = 20.0
        RIGHT_R = 80.0

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group())
            return random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT
        # --------------------------------------------------
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]

        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = m.group(1).strip() if m else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean = [], []
        for i in range(len(reshaped_content)):
            preds = []
            for j in range(len(reshaped_content[i])):
                matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                txt = matches[-1].strip() if matches else reshaped_content[i][j].strip()
                preds.append(extract_first_number(txt))
            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))

        batch_size = len(batch_pred)

        # --------------------------------------------------
        # Compute CCC (gen-aware)
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]

            # ---------- sample-level norm_d ----------
            if gt_i < LEFT_L:
                norm_d = 1.0
            elif gt_i > RIGHT_R:
                norm_d = 1.0
            elif gt_i < CENTER:
                norm_d = (CENTER - gt_i) / (CENTER - LEFT_L)
            elif gt_i > CENTER:
                norm_d = (gt_i - CENTER) / (RIGHT_R - CENTER)
            else:
                norm_d = 0.0

            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]

                # ---------- comparison pool ----------
                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]

                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ---------- gen-aware factor g_ij ----------
                e = abs(pred_i_j - gt_i)
                # g_ij = e / (e + TAU)          # ∈ (0,1)
                g_ij = TAU / (e + TAU)    # ∈ (0,1]，误差越小越接近 1

                # ---------- weight vector (only index 0) ----------
                w = np.ones_like(x, dtype=np.float32)
                w[0] = 1.0 + ALPHA * norm_d * g_ij

                w_sum = w.sum() + 1e-8

                # ---------- weighted CCC ----------
                mu_x = np.sum(w * x) / w_sum
                mu_y = np.sum(w * y) / w_sum

                var_x = np.sum(w * (x - mu_x) ** 2) / w_sum
                var_y = np.sum(w * (y - mu_y) ** 2) / w_sum
                cov_xy = np.sum(w * (x - mu_x) * (y - mu_y)) / w_sum

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                if np.isnan(ccc):
                    ccc = 0.0

                rewards.append(float(ccc))

                # ---------- Debug ----------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n====== GEN-AWARE CCC ======\n")
                        f.write(f"sample={i}, gen={j}\n")
                        f.write(f"gt={gt_i:.2f}, pred={pred_i_j:.2f}\n")
                        f.write(f"norm_d={norm_d:.3f}, g_ij={g_ij:.3f}\n")
                        f.write(f"w_ij={w[0]:.3f}\n")
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("===========================\n")

        return rewards


# 引入了memory mean, 和 memory gt的概念。
    @staticmethod
    def age_reward_global_ccc_memory_weighted(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np

        # ===============================
        # Global Memory Bank
        # ===============================
        global CCC_MEM_PRED, CCC_MEM_GT
        mem_pred = CCC_MEM_PRED
        mem_gt   = CCC_MEM_GT

        n_gen = kwargs.get("num_generations", 4)
        device = kwargs.get("device")
        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------
        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        # GT per sample
        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var = [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)
            # t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            # batch_mean.append(float(t.mean()))
            batch_mean.append(float(np.mean(cur_pred_list)))
            # batch_var.append([t.var()])

        batch_size = len(batch_pred)

        # ==================================================
        # Step 1️⃣ 冻结 memory（防止 self-leak）
        # ==================================================
        mem_pred_snapshot = list(mem_pred)
        mem_gt_snapshot   = list(mem_gt)

        # ==================================================
        # Step 2️⃣ Compute CCC reward (with current-sample weighting)
        # ==================================================
        rewards = []

        # extreme config（只影响当前 sample）
        ALPHA = kwargs.get("alpha", 1.0)   # 建议 0.5 ~ 1.5
        LOW_T = 30.0
        HIGH_T = 70.0

        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # ---------- 构建比较池 ----------
                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]

                # ① batch 内 other samples
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])

                # ② memory（全局尺度）
                if len(mem_pred_snapshot) > 0:
                    pred_list.extend(mem_pred_snapshot)
                    gt_list_cmp.extend(mem_gt_snapshot)
                    mode = "BATCH+MEMORY"
                else:
                    mode = "BATCH_ONLY"

                # ---------- 转 numpy ----------
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ---------- 构造权重（只对当前 sample） ----------
                w = np.ones_like(x, dtype=np.float32)
                if gt_i < LOW_T or gt_i > HIGH_T:
                    w[0] = 1.0 + ALPHA   # 只放大当前 sample

                w_sum = w.sum() + 1e-8

                # ---------- Weighted CCC ----------
                mu_x = np.sum(w * x) / w_sum
                mu_y = np.sum(w * y) / w_sum

                var_x = np.sum(w * (x - mu_x) ** 2) / w_sum
                var_y = np.sum(w * (y - mu_y) ** 2) / w_sum
                cov_xy = np.sum(w * (x - mu_x) * (y - mu_y)) / w_sum

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                # ---------- Debug ----------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ CCC DEBUG (Weighted) ================\n")
                        f.write(f"Sample i={i}, Generation j={j}\n")
                        f.write(f"Mode        : {mode}\n")
                        f.write(f"gt_i        : {gt_i:.3f}\n")
                        f.write(f"weight_i    : {w[0]:.2f}\n")
                        f.write(f"pred_i_j    : {pred_i_j:.3f}\n")
                        f.write(f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n")
                        f.write(f"var_x={var_x:.3f}, var_y={var_y:.3f}\n")
                        f.write(f"cov_xy={cov_xy:.3f}\n")
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("======================================================\n")
        return rewards


# 引入了memory mean, 和 memory gt的概念。 在memory 中加入extreme bank
    @staticmethod
    def age_reward_global_ccc_memory_weighted(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np

        # ===============================
        # Global Memory Bank
        # ===============================
        global CCC_MEM_PRED, CCC_MEM_GT, CCC_MEM_LIFE
        mem_pred = CCC_MEM_PRED
        mem_gt   = CCC_MEM_GT
        mem_life = CCC_MEM_LIFE

        n_gen = kwargs.get("num_generations", 4)
        device = kwargs.get("device")
        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------
        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean = [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)
            batch_mean.append(float(np.mean(cur_pred_list)))

        batch_size = len(batch_pred)

        # ==================================================
        # Step 1️⃣ 冻结 memory（防止 self-leak）
        # ==================================================
        mem_pred_snapshot = list(mem_pred)
        mem_gt_snapshot   = list(mem_gt)

        # ==================================================
        # Step 2️⃣ Compute CCC reward (current-sample weighted)
        # ==================================================
        rewards = []

        ALPHA = kwargs.get("alpha", 1.0)   # 只放大当前 sample
        LOW_T = 30.0
        HIGH_T = 70.0

        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # ---------- 构建比较池 ----------
                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]

                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])

                if len(mem_pred_snapshot) > 0:
                    pred_list.extend(mem_pred_snapshot)
                    gt_list_cmp.extend(mem_gt_snapshot)
                    mode = "BATCH+MEMORY"
                else:
                    mode = "BATCH_ONLY"

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ---------- 只对当前 sample 加权 ----------
                w = np.ones_like(x, dtype=np.float32)
                if gt_i < LOW_T or gt_i > HIGH_T:
                    w[0] = 1.0 + ALPHA

                w_sum = w.sum() + 1e-8

                mu_x = np.sum(w * x) / w_sum
                mu_y = np.sum(w * y) / w_sum
                var_x = np.sum(w * (x - mu_x) ** 2) / w_sum
                var_y = np.sum(w * (y - mu_y) ** 2) / w_sum
                cov_xy = np.sum(w * (x - mu_x) * (y - mu_y)) / w_sum

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ CCC DEBUG (Weighted+Memory) ================\n")
                        f.write(f"Sample i={i}, Gen j={j}, Mode={mode}\n")
                        f.write(f"gt_i={gt_i:.2f}, weight_i={w[0]:.2f}, pred={pred_i_j:.2f}\n")
                        f.write(f"Pred list   : {pred_list}\n")
                        f.write(f"GT list     : {gt_list_cmp}\n")
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("============================================================\n")

        # ==================================================
        # Step 3️⃣ Update memory with lifetime control
        # ==================================================
        # EXTREME_LIFE = kwargs.get("extreme_life", 20)   # 推荐 30~60
        EXTREME_LIFE = kwargs.get("extreme_life", 5)   # 推荐 30~60
        NORMAL_LIFE  = kwargs.get("normal_life", 1)    # 推荐 5~15

        # ① 所有已有 memory 的 life -= 1
        new_pred, new_gt, new_life = [], [], []
        for p, g, l in zip(mem_pred, mem_gt, mem_life):
            if l > 1:
                new_pred.append(p)
                new_gt.append(g)
                new_life.append(l - 1)

        mem_pred[:] = new_pred
        mem_gt[:]   = new_gt
        mem_life[:] = new_life

        # ② 插入当前 batch（extreme 活得更久）
        for i in range(batch_size):
            gt_i = gt_list[i]
            pred_i = batch_mean[i]

            if gt_i < LOW_T or gt_i > HIGH_T:
                life = EXTREME_LIFE
            else:
                life = NORMAL_LIFE

            mem_pred.append(pred_i)
            mem_gt.append(gt_i)
            mem_life.append(life)

        return rewards


# 这两个纯hard 版本会引入 非常大的预测量，导致结果不稳定。
    @staticmethod
    def age_reward_global_ccc_memory_hard(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        from collections import deque

        # ===============================
        # Global Memory Bank (Hard samples)
        # ===============================
        global CCC_MEM
        if "CCC_MEM" not in globals():
            CCC_MEM = deque()

        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)

        # 合法 pred 范围（Age 任务先验）
        PRED_MIN = 0.0
        PRED_MAX = 100.0

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            return float(m.group()) if m else random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                reshaped_solution[i][j] = float(m.group(1)) if m else float(reshaped_solution[i][j])

        gt_list = [s[0] for s in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean, batch_std = [], [], []

        for i in range(len(reshaped_content)):
            preds = []
            for j in range(len(reshaped_content[i])):
                m = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                text = m[-1] if m else reshaped_content[i][j]
                preds.append(extract_first_number(text))

            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))
            batch_std.append(float(np.std(preds)))

        batch_size = len(batch_pred)

        # --------------------------------------------------
        # Snapshot memory (NO self-leak)
        # --------------------------------------------------
        mem_snapshot = list(CCC_MEM)

        # --------------------------------------------------
        # Compute CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # batch samples
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")

                # memory samples（此处默认已合法）
                for m in mem_snapshot:
                    pred_list.append(m["pred"])
                    gt_list_cmp.append(m["gt"])
                    src_list.append("MEM")

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                # --------------------------------------------------
                # CCC DEBUG
                # --------------------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n========== CCC DEBUG ==========\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"GT_i={gt_i:.2f}, pred_i_j={pred_i_j:.2f}\n")
                        f.write("Participants:\n")
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<8} "
                                f"pred={pred_list[k]:.2f}, "
                                f"gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(
                            f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}, "
                            f"var_x={var_x:.3f}, var_y={var_y:.3f}, "
                            f"cov_xy={cov_xy:.3f}\n"
                        )
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("================================\n")

        # --------------------------------------------------
        # Update Memory with HARD samples (合法性过滤)
        # --------------------------------------------------
        for i in range(batch_size):
            pred_mean_i = batch_mean[i]
            gt_i = gt_list[i]
            err_i = abs(pred_mean_i - gt_i)
            std_i = batch_std[i]
            hard_i = err_i + std_i   # 选择标准保持不变

            # -------- pred 合法性过滤 --------
            if not (PRED_MIN <= pred_mean_i <= PRED_MAX):
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write(
                            f"[MEM-SKIP] sample {i}: "
                            f"gt={gt_i:.2f}, "
                            f"pred_mean={pred_mean_i:.2f} (OUT OF RANGE)\n"
                        )
                continue

            CCC_MEM.append({
                "pred": pred_mean_i,
                "gt": gt_i,
                "err": err_i,
                "std": std_i,
                "hard": hard_i
            })

            if DEBUG:
                with open(log_path_rank, "a", encoding="utf-8") as f:
                    f.write(
                        f"[MEM-ADD] sample {i}: "
                        f"gt={gt_i:.2f}, "
                        f"pred_mean={pred_mean_i:.2f}, "
                        f"err={err_i:.2f}, "
                        f"std={std_i:.2f}, "
                        f"hard={hard_i:.2f}\n"
                    )

        # keep hardest K（仍然按 hard 排序）
        CCC_MEM = deque(
            sorted(CCC_MEM, key=lambda x: x["hard"], reverse=True)[:MAX_MEM_SIZE]
        )

        # --------------------------------------------------
        # Memory DEBUG
        # --------------------------------------------------
        if DEBUG:
            with open(log_path_rank, "a", encoding="utf-8") as f:
                f.write("\n====== HARD MEMORY STATUS ======\n")
                for k, m in enumerate(CCC_MEM):
                    f.write(
                        f"[{k}] gt={m['gt']:.2f}, "
                        f"pred={m['pred']:.2f}, "
                        f"err={m['err']:.2f}, "
                        f"std={m['std']:.2f}, "
                        f"hard={m['hard']:.2f}\n"
                    )
                f.write("================================\n")

        return rewards



    @staticmethod
    def age_reward_global_ccc_memory_hard_(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        from collections import deque

        # ===============================
        # Global states
        # ===============================
        global CCC_MEM, CCC_STEP, MEM_WARMUP_STEPS

        # if "CCC_MEM" not in globals():
        #     CCC_MEM = deque()
        # if "CCC_STEP" not in globals():
        #     CCC_STEP = 0
        # if "MEM_WARMUP_STEPS" not in globals():
        #     MEM_WARMUP_STEPS = 2000   # ⭐ 你可以在这里统一控制 warmup

        CCC_STEP += 1

        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)

        # memory warmup
        use_memory = CCC_STEP >= MEM_WARMUP_STEPS

        # 合法 pred 范围（Age 任务先验）
        PRED_MIN = 0.0
        PRED_MAX = 100.0

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper (return value + is_random flag)
        # --------------------------------------------------
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group()), False
            else:
                return random.uniform(1, 100), True

        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                reshaped_solution[i][j] = float(m.group(1)) if m else float(reshaped_solution[i][j])

        gt_list = [s[0] for s in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean, batch_std, batch_has_random = [], [], [], []

        for i in range(len(reshaped_content)):
            preds = []
            has_random = False
            for j in range(len(reshaped_content[i])):
                m = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                text = m[-1] if m else reshaped_content[i][j]
                val, is_rand = extract_first_number(text)
                preds.append(val)
                if is_rand:
                    has_random = True

            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))
            batch_std.append(float(np.std(preds)))
            batch_has_random.append(has_random)

        batch_size = len(batch_pred)

        # --------------------------------------------------
        # Snapshot memory (NO self-leak)
        # --------------------------------------------------
        mem_snapshot = list(CCC_MEM) if use_memory else []

        # --------------------------------------------------
        # Compute CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # batch samples
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")

                # memory samples（warmup 后才用）
                if use_memory:
                    for m in mem_snapshot:
                        pred_list.append(m["pred"])
                        gt_list_cmp.append(m["gt"])
                        src_list.append("MEM")

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                # --------------------------------------------------
                # CCC DEBUG
                # --------------------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n========== CCC DEBUG ==========\n")
                        f.write(
                            f"STEP={CCC_STEP}, use_memory={use_memory}\n"
                            f"Sample i={i}, Gen j={j}\n"
                            f"GT_i={gt_i:.2f}, pred_i_j={pred_i_j:.2f}\n"
                        )
                        f.write("Participants:\n")
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<8} "
                                f"pred={pred_list[k]:.2f}, "
                                f"gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(
                            f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}, "
                            f"var_x={var_x:.3f}, var_y={var_y:.3f}, "
                            f"cov_xy={cov_xy:.3f}\n"
                        )
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("================================\n")

        # --------------------------------------------------
        # Update Memory with HARD samples (after warmup)
        # --------------------------------------------------
        if use_memory:
            for i in range(batch_size):

                # ❌ 跳过 random fallback 样本
                if batch_has_random[i]:
                    if DEBUG:
                        with open(log_path_rank, "a", encoding="utf-8") as f:
                            f.write(
                                f"[MEM-SKIP-RANDOM] gt={gt_list[i]:.2f}, "
                                f"pred_mean={batch_mean[i]:.2f}\n"
                            )
                    continue

                pred_mean_i = batch_mean[i]
                gt_i = gt_list[i]
                err_i = abs(pred_mean_i - gt_i)
                std_i = batch_std[i]
                hard_i = err_i + std_i

                # pred 合法性过滤
                if not (PRED_MIN <= pred_mean_i <= PRED_MAX):
                    if DEBUG:
                        with open(log_path_rank, "a", encoding="utf-8") as f:
                            f.write(
                                f"[MEM-SKIP-RANGE] gt={gt_i:.2f}, "
                                f"pred_mean={pred_mean_i:.2f}\n"
                            )
                    continue

                CCC_MEM.append({
                    "pred": pred_mean_i,
                    "gt": gt_i,
                    "err": err_i,
                    "std": std_i,
                    "hard": hard_i
                })

                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write(
                            f"[MEM-ADD] gt={gt_i:.2f}, "
                            f"pred_mean={pred_mean_i:.2f}, "
                            f"err={err_i:.2f}, std={std_i:.2f}, "
                            f"hard={hard_i:.2f}\n"
                        )

            # keep hardest K
            CCC_MEM = deque(
                sorted(CCC_MEM, key=lambda x: x["hard"], reverse=True)[:MAX_MEM_SIZE]
            )

            # Memory debug
            if DEBUG:
                with open(log_path_rank, "a", encoding="utf-8") as f:
                    f.write("\n====== HARD MEMORY STATUS ======\n")
                    for k, m in enumerate(CCC_MEM):
                        f.write(
                            f"[{k}] gt={m['gt']:.2f}, "
                            f"pred={m['pred']:.2f}, "
                            f"err={m['err']:.2f}, "
                            f"std={m['std']:.2f}, "
                            f"hard={m['hard']:.2f}\n"
                        )
                    f.write("================================\n")

        return rewards


  
# 全局都做，新加入了一个裁剪  memory 参数。
    @staticmethod
    def age_reward_global_ccc_memory_counter_direction(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        from collections import deque

        # ===============================
        # Global states
        # ===============================
        global CCC_MEM, CCC_STEP, MEM_WARMUP_STEPS
        CCC_STEP += 1

        # ===============================
        # Config
        # ===============================
        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)

        USE_REG_LOW = 30.0
        USE_REG_HIGH = 50.0
        REG_SCALE = 5.0   # regression reward 的尺度，避免 std 过大

        ERR_TRIGGER = 7.0

        # ⭐ 方案 1：counter-memory 幅度上限
        DELTA_MAX = 30.0

        NORMAL_LOW = 20.0
        NORMAL_HIGH = 70.0

        use_memory = CCC_STEP >= MEM_WARMUP_STEPS

        PRED_MIN = 0.0
        PRED_MAX = 100.0

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # ===============================
        # Helper
        # ===============================
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group()), False
            else:
                return random.uniform(1, 100), True


        # def get_memory_prob_by_gt(gt):
        #     GT_BIN_COUNTS = [
        #         (0, 20, 231),
        #         (20, 30, 1901),
        #         (30, 40, 3012),
        #         (40, 50, 2564),
        #         (50, 60, 2045),
        #         (60, 70, 1451),
        #         (70, 80, 793),
        #         (80, 100, 210),
        #     ]

        #     ANCHOR = 210        # 最小 bin 作为 anchor
        #     P_MIN = 0.05        # 防止中段完全不参与（可调）
        #     P_MAX = 1.0         # 极端 bin 几乎必用

        #     for low, high, count in GT_BIN_COUNTS:
        #         if low <= gt < high:
        #             p = ANCHOR / count
        #             p = max(P_MIN, min(P_MAX, p))
        #             return p

        #     return P_MIN  # fallback

        def memory_prob_from_count(count, min_count, alpha=0.3):
            """
            Temperature-scaled inverse frequency (alpha=0.5)

            p = (min_count / count) ** alpha
            """
            if count <= 0:
                return 1.0
            return (min_count / count) ** alpha

        def get_memory_prob_by_gt(gt):
            GT_BIN_COUNTS = [
                (0, 20, 231),
                (20, 30, 1901),
                (30, 40, 3012),
                (40, 50, 2564),
                (50, 60, 2045),
                (60, 70, 1451),
                (70, 80, 793),
                (80, 100, 210),
            ]

            c_min = 210  # anchor: 最小 bin

            for low, high, count in GT_BIN_COUNTS:
                if low <= gt < high:
                    return memory_prob_from_count(
                        count=count,
                        min_count=c_min,
                        # alpha=0.5
                        alpha=0.3
                    )

            return 0.3  # fallback

        # ===============================
        # Parse GT
        # ===============================
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                reshaped_solution[i][j] = float(m.group(1)) if m else float(reshaped_solution[i][j])

        gt_list = [s[0] for s in reshaped_solution]

        # ===============================
        # Parse predictions
        # ===============================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean, batch_has_random = [], [], []

        for i in range(len(reshaped_content)):
            preds = []
            has_random = False
            for j in range(len(reshaped_content[i])):
                m = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                text = m[-1] if m else reshaped_content[i][j]
                val, is_rand = extract_first_number(text)
                preds.append(val)
                if is_rand:
                    has_random = True

            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))
            batch_has_random.append(has_random)

        batch_size = len(batch_pred)

        # ===============================
        # Snapshot memory
        # ===============================
        mem_snapshot = list(CCC_MEM) if use_memory else []

        # ===============================
        # Compute CCC reward
        # ===============================
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]


            # 是否使用 regression reward（样本级）
            # use_regression_reward = (USE_REG_LOW <= gt_i <= USE_REG_HIGH)



            # ===============================
            # Sample-level memory gate
            # ===============================
            # is_extreme_sample = (gt_i < NORMAL_LOW or gt_i > NORMAL_HIGH)

            # allow_memory_for_sample  = (gt_i < NORMAL_LOW or gt_i > NORMAL_HIGH)

            # if is_extreme_sample:
            #     allow_memory_for_sample = True
            # else:
            #     # allow_memory_for_sample = (random.random() < 0.5)
            #     allow_memory_for_sample = (random.random() < (1.0 / 3.0))


            # p_mem = get_memory_prob_by_gt(gt_i)
            # allow_memory_for_sample = (
            #     (random.random() < p_mem)
            # )

            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                err_ij = abs(pred_i_j - gt_i)



                # ==================================================
                # 🟢 CASE 1: 30–50 区间 → regression reward
                # ==================================================
                # if use_regression_reward:
                #     # reg_reward = -err_ij / REG_SCALE
                #     reg_reward = max(0.0, 1 - err_ij * 0.1)
                #     rewards.append(float(reg_reward))

                #     if DEBUG:
                #         with open(log_path_rank, "a", encoding="utf-8") as f:
                #             f.write("\n===== REGRESSION REWARD DEBUG =====\n")
                #             f.write(
                #                 f"STEP={CCC_STEP}\n"
                #                 f"Sample i={i}, Gen j={j}\n"
                #                 f"GT={gt_i:.2f}, pred={pred_i_j:.2f}, err={err_ij:.2f}\n"
                #                 f"reg_reward={reg_reward:.4f}\n"
                #                 f"(GT in [30,50], CCC skipped)\n"
                #                 "===================================\n"
                #             )
                #     continue




                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # ---- batch samples ----
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")




                # # -------------------------------
                # # Build comparison set (batch)
                # # -------------------------------
                # other_indices = [z for z in range(batch_size) if z != i]

                # # ⭐ 在 [50–70] 区间，仅保留 2 个 batch 对照（CURRENT + 2 = 3）
                # # if 50.0 <= gt_i <= 70.0 and len(other_indices) > 2:
                # if (
                #     (20.0 <= gt_i <= 30.0 or 50.0 <= gt_i <= 70.0)
                #     and len(other_indices) > 2
                # ):
                #     # 按 GT 距离排序（远的更不相关）
                #     other_indices = sorted(
                #         other_indices,
                #         key=lambda z: abs(gt_list[z] - gt_i)
                #     )
                #     # 只保留距离最近的 2 个
                #     other_indices = other_indices[:2]

                # # 初始化 comparison list
                # pred_list = [pred_i_j]
                # gt_list_cmp = [gt_i]
                # src_list = ["CURRENT"]

                # # 加入 batch 对照
                # for z in other_indices:
                #     pred_list.append(batch_mean[z])
                #     gt_list_cmp.append(gt_list[z])
                #     src_list.append(f"BATCH[{z}]")



                # ---- counter-direction memory (scale-only) ----
                # if use_memory and allow_memory_for_sample and err_ij > ERR_TRIGGER:
                if use_memory and err_ij > ERR_TRIGGER:
                    for m in mem_snapshot:
                        # 方向相反
                        if (pred_i_j - gt_i) * (m["pred"] - m["gt"]) < 0:
                            delta = m["pred"] - m["gt"]
                            delta_clipped = np.clip(delta, -DELTA_MAX, DELTA_MAX)
                            pred_counter = m["gt"] + delta_clipped

                            pred_list.append(pred_counter)
                            gt_list_cmp.append(m["gt"])
                            src_list.append("MEM-COUNTER-CLIPPED")

                # ---- CCC ----
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                # ===============================
                # DEBUG
                # ===============================
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n===== CCC DEBUG (COUNTER-SCALE-ONLY) =====\n")
                        f.write(
                            f"STEP={CCC_STEP}, err_ij={err_ij:.2f}\n"
                            f"Sample i={i}, Gen j={j}\n"
                            f"GT={gt_i:.2f}, pred={pred_i_j:.2f}\n"
                        )
                        f.write("Participants:\n")
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<20} "
                                f"pred={pred_list[k]:.2f}, gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("=========================================\n")

        # ===============================
        # Update Memory (hard samples)
        # ===============================
        if use_memory:
            for i in range(batch_size):
                if batch_has_random[i]:
                    continue

                pred_mean_i = batch_mean[i]
                gt_i = gt_list[i]

                if not (PRED_MIN <= pred_mean_i <= PRED_MAX):
                    continue

                err_i = abs(pred_mean_i - gt_i)

                CCC_MEM.append({
                    "pred": pred_mean_i,
                    "gt": gt_i,
                    "err": err_i
                })

            CCC_MEM = deque(
                sorted(CCC_MEM, key=lambda x: x["err"], reverse=True)[:MAX_MEM_SIZE]
            )

        return rewards



    @staticmethod
    def fundus_reward_global_ccc_with_memory(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        from collections import deque
        from scipy.stats import spearmanr

        # ===============================
        # Global states (must be defined outside)
        # ===============================
        global CCC_MEM, CCC_STEP, MEM_WARMUP_STEPS
        CCC_STEP += 1

        # ===============================
        # Config
        # ===============================
        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)

        use_memory = CCC_STEP >= MEM_WARMUP_STEPS

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # ===============================
        # Helper
        # ===============================
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group()), False
            else:
                # fundus fallback：随机 0–4
                # return random.uniform(0, 4), True
                return random.randint(0, 4), True

        # ===============================
        # Parse GT
        # ===============================
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]

        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                reshaped_solution[i][j] = float(m.group(1)) if m else float(reshaped_solution[i][j])

        gt_list = [s[0] for s in reshaped_solution]   # GT per sample

        # ===============================
        # Parse predictions
        # ===============================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean, batch_has_random = [], [], []

        for i in range(len(reshaped_content)):
            preds = []
            has_random = False
            for j in range(len(reshaped_content[i])):
                m = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                text = m[-1] if m else reshaped_content[i][j]
                val, is_rand = extract_first_number(text)
                preds.append(val)
                if is_rand:
                    has_random = True

            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))
            batch_has_random.append(has_random)

        batch_size = len(batch_pred)

        # ===============================
        # Snapshot memory
        # ===============================
        mem_snapshot = list(CCC_MEM) if use_memory else []

        # ===============================
        # Compute CCC reward
        # ===============================
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]

            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]

                # -------------------------------
                # Build comparison set
                # -------------------------------
                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # ---- batch anchors ----
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")

                # ---- memory anchors (non-zero only) ----
                if use_memory:
                    for m in mem_snapshot:
                        if m["gt"] == 0:
                            continue
                        pred_list.append(m["pred"])
                        gt_list_cmp.append(m["gt"])
                        src_list.append("MEM-NONZERO")

                # -------------------------------
                # CCC computation (0–4 space)
                # -------------------------------
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ------ Spearman rank correlation ------
                # rho, _ = spearmanr(x, y)
                # if np.isnan(rho):
                #     rho = 0.0
                # rewards.append(rho)

                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else float(ccc)


                # if gt_i != 0 and pred_i_j == 0:
                #     ccc = ccc - 0.05   # 轻微惩罚全0
                # elif gt_i != 0 and pred_i_j != 0:
                #     ccc = ccc + 0.02   # 轻微奖励非0（可选）

                rewards.append(ccc)

                # -------------------------------
                # DEBUG
                # -------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n===== FUNDUS CCC DEBUG =====\n")
                        f.write(
                            f"STEP={CCC_STEP}\n"
                            f"Sample i={i}, Gen j={j}\n"
                            f"GT={gt_i:.2f}, pred={pred_i_j:.2f}\n"
                        )
                        f.write("Participants:\n")
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<15} "
                                f"pred={pred_list[k]:.2f}, gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(
                            f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}, "
                            f"var_x={var_x:.3f}, var_y={var_y:.3f}, "
                            f"CCC={ccc:.4f}\n"
                            # f"Spearman rho = {rho:.4f}\n"
                        )
                        f.write("================================\n")


        # ===============================
        # Update Memory (GT ≠ 0 AND pred ≠ 0)
        # ===============================
        if use_memory:
            for i in range(batch_size):
                if batch_has_random[i]:
                    continue

                gt_i = gt_list[i]
                pred_mean_i = batch_mean[i]

                # ⭐ 核心过滤条件
                if gt_i == 0:
                    continue
                if pred_mean_i == 0:
                    continue

                if not (0.0 <= pred_mean_i <= 4.0):
                    continue

                CCC_MEM.append({
                    "pred": pred_mean_i,
                    "gt": gt_i
                })

            # FIFO 截断（不按 error）
            CCC_MEM = deque(list(CCC_MEM)[-MAX_MEM_SIZE:])

 

        return rewards


    @staticmethod
    def fundus_reward_global_ccc_interval(completions, solution, **kwargs):
        import re
        import numpy as np
        import random
        import os

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)
        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper: extract number
        # --------------------------------------------------
        def extract_label(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group())
            else:
                return random.randint(0, 4)

        # --------------------------------------------------
        # Helper: interval mapping (核心)
        # --------------------------------------------------
        def map_to_interval(v, noise=True):
            """
            v in [0,4] → expanded continuous space
            """
            base = v * 20.0   # 0→0, 1→20, ..., 4→80
            if noise:
                return base + random.uniform(0.0, 4.0)
            else:
                return base + 5.0

        # --------------------------------------------------
        # Parse GT
        # --------------------------------------------------
        reshaped_solution = [
            solution[i:i + n_gen]
            for i in range(0, len(solution), n_gen)
        ]

        gt_list = []
        for sol in reshaped_solution:
            m = re.search(r'<answer>(.*?)</answer>', sol[0])
            gt = float(m.group(1)) if m else float(sol[0])
            gt_list.append(gt)

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen]
            for i in range(0, len(contents), n_gen)
        ]

        batch_pred = []
        batch_mean = []

        for preds in reshaped_content:
            cur = []
            for p in preds:
                m = re.findall(r'<answer>(.*?)</answer>', p, re.DOTALL)
                text = m[-1] if m else p
                cur.append(extract_label(text))
            batch_pred.append(cur)
            batch_mean.append(float(np.mean(cur)))

        batch_size = len(batch_pred)

        # --------------------------------------------------
        # CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]

            for j in range(n_gen):
                pred_ij = batch_pred[i][j]

                # ---- build comparison set ----
                pred_list = [pred_ij]
                gt_list_cmp = [gt_i]

                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])

                # ---- interval mapping ----
                x = np.array(
                    [map_to_interval(p) for p in pred_list],
                    dtype=np.float32
                )
                y = np.array(
                    [map_to_interval(g) for g in gt_list_cmp],
                    dtype=np.float32
                )

                # ---- CCC ----
                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else float(ccc)

                rewards.append(ccc)

                # ---------------------------------------
                # Debug Log
                # ---------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ CCC Pair =================\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"GT_i = {gt_i}\n")
                        f.write(f"pred_i_j = {pred_ij}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list   = {gt_list_cmp}\n")
                        f.write(f"Pred list = {x}\n")
                        f.write(f"GT list   = {y}\n")
                        f.write(f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n")
                        f.write(f"var_x={var_x:.3f}, var_y={var_y:.3f}\n")
                        f.write(f"cov_xy={cov_xy:.3f}\n")
                        f.write(f"CCC = {ccc:.4f}, Reward = {ccc:.4f}\n")
                        # f.write(f"abs_err = {abs_err:.2f}\n")
                        # f.write(f"reg_gate = {reg_gate:.4f}\n")
                        # f.write(f"zone={'REG' if REG_LOW <= gt_i <= REG_HIGH else 'CCC'}\n")
                        f.write(f"final_reward = {ccc:.4f}\n")
                        f.write("===========================================\n")

        return rewards

        return rewards

# 这里对于极端样本加入 单一generation的选择过滤，虽然结果平滑，但是  效果 几乎没有。
    @staticmethod
    def age_reward_global_ccc_memory_counter_direction_v2(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np
        from collections import deque

        # ===============================
        # Global states
        # ===============================
        global CCC_MEM, CCC_STEP, MEM_WARMUP_STEPS
        CCC_STEP += 1

        # ===============================
        # Config
        # ===============================
        MAX_MEM_SIZE = kwargs.get("max_mem_size", 4)
        n_gen = kwargs.get("num_generations", 4)

        ERR_TRIGGER = 5.0          # 单个 generation 的“很差”
        GOOD_GEN_THRESHOLD = 5.0  # 至少要有一个“不差”的 generation

        use_memory = CCC_STEP >= MEM_WARMUP_STEPS

        PRED_MIN = 0.0
        PRED_MAX = 100.0

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # ===============================
        # Helper
        # ===============================
        def extract_first_number(text):
            m = re.search(r'-?\d+(\.\d+)?', text)
            if m:
                return float(m.group()), False
            else:
                return random.uniform(1, 100), True

        # ===============================
        # Parse GT
        # ===============================
        reshaped_solution = [
            solution[i:i + n_gen] for i in range(0, len(solution), n_gen)
        ]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                m = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                reshaped_solution[i][j] = float(m.group(1)) if m else float(reshaped_solution[i][j])

        gt_list = [s[0] for s in reshaped_solution]

        # ===============================
        # Parse predictions
        # ===============================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [
            contents[i:i + n_gen] for i in range(0, len(contents), n_gen)
        ]

        batch_pred, batch_mean, batch_has_random = [], [], []

        for i in range(len(reshaped_content)):
            preds = []
            has_random = False
            for j in range(len(reshaped_content[i])):
                m = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                text = m[-1] if m else reshaped_content[i][j]
                val, is_rand = extract_first_number(text)
                preds.append(val)
                if is_rand:
                    has_random = True

            batch_pred.append(preds)
            batch_mean.append(float(np.mean(preds)))
            batch_has_random.append(has_random)

        batch_size = len(batch_pred)

        # ===============================
        # Snapshot memory
        # ===============================
        mem_snapshot = list(CCC_MEM) if use_memory else []

        # ===============================
        # Compute rewards
        # ===============================
        rewards = []

        for i in range(batch_size):
            gt_i = gt_list[i]

            # ---------- generation-level gate ----------
            gen_errors = [abs(batch_pred[i][jj] - gt_i) for jj in range(n_gen)]
            has_good_gen = min(gen_errors) < GOOD_GEN_THRESHOLD

            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]

                # # 🚨 hard guard: current pred 必须合法
                # if not (PRED_MIN <= pred_i_j <= PRED_MAX):
                #     rewards.append(-1.0)
                #     continue

                err_ij = abs(pred_i_j - gt_i)

                pred_list = [pred_i_j]
                gt_list_cmp = [gt_i]
                src_list = ["CURRENT"]

                # ---- batch samples ----
                for z in range(batch_size):
                    if z != i:
                        pred_list.append(batch_mean[z])
                        gt_list_cmp.append(gt_list[z])
                        src_list.append(f"BATCH[{z}]")

                # ---- counter-direction memory（严格 gate）----
                if (
                    use_memory and
                    has_good_gen and
                    err_ij > ERR_TRIGGER
                ):
                    for m in mem_snapshot:
                        if (pred_i_j - gt_i) * (m["pred"] - m["gt"]) < 0:
                            pred_list.append(m["pred"])
                            gt_list_cmp.append(m["gt"])
                            src_list.append("MEM-COUNTER")

                # ---- CCC ----
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                mu_x, mu_y = x.mean(), y.mean()
                var_x, var_y = x.var(), y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2
                ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
                ccc = 0.0 if np.isnan(ccc) else ccc

                rewards.append(float(ccc))

                # ===============================
                # DEBUG
                # ===============================
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n===== CCC DEBUG (COUNTER v2) =====\n")
                        f.write(
                            f"STEP={CCC_STEP}, use_memory={use_memory}, "
                            f"has_good_gen={has_good_gen}, "
                            f"min_gen_err={min(gen_errors):.2f}\n"
                            f"Sample i={i}, Gen j={j}\n"
                            f"GT={gt_i:.2f}, pred={pred_i_j:.2f}, err_ij={err_ij:.2f}\n"
                        )
                        f.write("Participants:\n")
                        for k in range(len(pred_list)):
                            f.write(
                                f"  [{k}] src={src_list[k]:<14} "
                                f"pred={pred_list[k]:.2f}, gt={gt_list_cmp[k]:.2f}\n"
                            )
                        f.write(f"CCC={ccc:.4f}\n")
                        f.write("=================================\n")

        # ===============================
        # Update memory (hard samples)
        # ===============================
        if use_memory:
            for i in range(batch_size):

                if batch_has_random[i]:
                    continue

                pred_mean_i = batch_mean[i]
                gt_i = gt_list[i]

                if not (PRED_MIN <= pred_mean_i <= PRED_MAX):
                    continue

                err_i = abs(pred_mean_i - gt_i)

                CCC_MEM.append({
                    "pred": pred_mean_i,
                    "gt": gt_i,
                    "err": err_i
                })

            CCC_MEM = deque(
                sorted(CCC_MEM, key=lambda x: x["err"], reverse=True)[:MAX_MEM_SIZE]
            )

        return rewards


    @staticmethod
    def age_reward_global_ccc_spear(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime
        from scipy.stats import spearmanr

        # --------------------------------------------------
        # Load configs
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_cccspear.txt")

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------
        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.uniform(1, 100)
                return random.randint(0, 4)

        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        # GT per sample
        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var = [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(
                    r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL
                )
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])

        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])

        # --------------------------------------------------
        # Compute Hybrid Reward: 0.5 * Spearman + 0.5 * CCC
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):               # for each sample
            for j in range(n_gen):               # for each generation

                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # Build lists
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                # Convert to numpy
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # ------ Spearman rank correlation ------
                rho, _ = spearmanr(x, y)
                if np.isnan(rho):
                    rho = 0.0

                # # ------ CCC calculation ------
                # mu_x = x.mean()
                # mu_y = y.mean()
                # var_x = x.var()
                # var_y = y.var()
                # cov_xy = np.mean((x - mu_x) * (y - mu_y))

                # denom = var_x + var_y + (mu_x - mu_y) ** 2
                # if denom == 0:
                #     ccc = 0.0
                # else:
                #     ccc = (2 * cov_xy) / denom

                # if np.isnan(ccc):
                #     ccc = 0.0

                # ------ Final Hybrid Reward ------
                # Option A: keep raw [-1,1] scale
                # reward = 0.5 * rho + 0.5 * ccc
                reward = rho

                # Option B: map both to [0,1] then average
                # reward = 0.5 * ((rho + 1)/2) + 0.5 * ((ccc + 1)/2)

                rewards.append(float(reward))

                # ---------------------------------------
                # Debug Log
                # ---------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ Hybrid CCC+Spearman =================\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"GT_i = {gt_i}\n")
                        f.write(f"pred_i_j = {pred_i_j}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list   = {gt_list_cmp}\n")
                        f.write(f"Spearman rho = {rho:.4f}\n")
                        # f.write(f"CCC = {ccc:.4f}\n")
                        f.write(f"Final Reward = {reward:.4f}\n")
                        f.write("=====================================================\n")

        return rewards


    @staticmethod
    def gaze_reward_global_ccc_w_angle(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            def to_vec(pitch, yaw):
                pitch = np.radians(pitch)
                yaw = np.radians(yaw)

                x = -np.cos(pitch) * np.sin(yaw)
                y = -np.sin(pitch)
                z = -np.cos(pitch) * np.cos(yaw)
                return np.array([x, y, z], dtype=np.float32)

            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                # -------- CCC (yaw)
                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                # 建议做平均，避免量纲过大
                ccc = 0.5 * (ccc_p + ccc_y)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                # 默认“两个reward都给1份权重”
                reward = r_ang + ccc

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_avg={ccc:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards

    @staticmethod
    def gaze_reward_global_ccc_wo_angle(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            def to_vec(pitch, yaw):
                pitch = np.radians(pitch)
                yaw = np.radians(yaw)

                x = -np.cos(pitch) * np.sin(yaw)
                y = -np.sin(pitch)
                z = -np.cos(pitch) * np.cos(yaw)
                return np.array([x, y, z], dtype=np.float32)

            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                # -------- CCC (yaw)
                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                # 建议做平均，避免量纲过大

                ccc = (ccc_p + ccc_y)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                # 默认“两个reward都给1份权重”
                reward = ccc
                

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_avg={ccc:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards


    @staticmethod
    def gaze_reward_global_angular_w_ccc_vector(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        # reward mode:
        # "angular"
        # "ccc_xy"
        # "angular+ccc_xy"
        # "ccc_vec"
        # "angular+ccc_vec"
        # reward_mode = kwargs.get("reward_mode", "angular+ccc_vec")
        reward_mode = kwargs.get("reward_mode", "angular+ccc_vec")

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        def to_vec(pitch, yaw):
            """
            pitch / yaw 单位：degree
            输出 3D gaze direction vector
            """
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)

            x = -np.cos(pitch) * np.sin(yaw)
            y = -np.sin(pitch)
            z = -np.cos(pitch) * np.cos(yaw)
            return np.array([x, y, z], dtype=np.float32)

        def compute_vector_ccc(pred_vecs, gt_vecs):
            """
            pred_vecs: list/array of shape [B, 3]
            gt_vecs:   list/array of shape [B, 3]
            做法：分别对 x/y/z 三维算 CCC，再取平均
            """
            pred = np.array(pred_vecs, dtype=np.float32)  # [B, 3]
            gt = np.array(gt_vecs, dtype=np.float32)      # [B, 3]

            cccs = []
            for d in range(3):
                cccs.append(compute_ccc(pred[:, d], gt[:, d]))

            return float(np.mean(cccs))

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch/yaw scalar space)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                ccc_xy = 0.5 * (ccc_p + ccc_y)

                # -------- Vector CCC (3D gaze direction space)
                pred_vecs = []
                gt_vecs = []

                for z in range(batch_size):
                    if z == i:
                        pred_vecs.append(to_vec(p_pred, y_pred))
                        gt_vecs.append(to_vec(p_gt, y_gt))
                    else:
                        pred_vecs.append(to_vec(batch_mean_pitch[z], batch_mean_yaw[z]))
                        gt_vecs.append(to_vec(gt_pitch[z], gt_yaw[z]))

                ccc_vec = compute_vector_ccc(pred_vecs, gt_vecs)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                if reward_mode == "angular":
                    reward = r_ang

                elif reward_mode == "ccc_xy":
                    reward = ccc_xy

                elif reward_mode == "angular+ccc_xy":
                    reward = r_ang + ccc_xy

                elif reward_mode == "ccc_vec":
                    reward = ccc_vec

                elif reward_mode == "angular+ccc_vec":
                    reward = r_ang + ccc_vec

                else:
                    raise ValueError(f"Unknown reward_mode: {reward_mode}")

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"reward_mode = {reward_mode}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_xy={ccc_xy:.4f}\n")
                        f.write(f"CCC_vec={ccc_vec:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards



    @staticmethod
    def gaze_reward_global_ccc_vector(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        # reward mode:
        # "angular"
        # "ccc_xy"
        # "angular+ccc_xy"
        # "ccc_vec"
        # "angular+ccc_vec"
        # reward_mode = kwargs.get("reward_mode", "angular+ccc_vec")
        reward_mode = kwargs.get("reward_mode", "ccc_vec")

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        def to_vec(pitch, yaw):
            """
            pitch / yaw 单位：degree
            输出 3D gaze direction vector
            """
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)

            x = -np.cos(pitch) * np.sin(yaw)
            y = -np.sin(pitch)
            z = -np.cos(pitch) * np.cos(yaw)
            return np.array([x, y, z], dtype=np.float32)

        def compute_vector_ccc(pred_vecs, gt_vecs):
            """
            pred_vecs: list/array of shape [B, 3]
            gt_vecs:   list/array of shape [B, 3]
            做法：分别对 x/y/z 三维算 CCC，再取平均
            """
            pred = np.array(pred_vecs, dtype=np.float32)  # [B, 3]
            gt = np.array(gt_vecs, dtype=np.float32)      # [B, 3]

            cccs = []
            for d in range(3):
                cccs.append(compute_ccc(pred[:, d], gt[:, d]))

            return float(np.mean(cccs))

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch/yaw scalar space)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                ccc_xy = 0.5 * (ccc_p + ccc_y)

                # -------- Vector CCC (3D gaze direction space)
                pred_vecs = []
                gt_vecs = []

                for z in range(batch_size):
                    if z == i:
                        pred_vecs.append(to_vec(p_pred, y_pred))
                        gt_vecs.append(to_vec(p_gt, y_gt))
                    else:
                        pred_vecs.append(to_vec(batch_mean_pitch[z], batch_mean_yaw[z]))
                        gt_vecs.append(to_vec(gt_pitch[z], gt_yaw[z]))

                ccc_vec = compute_vector_ccc(pred_vecs, gt_vecs)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                if reward_mode == "angular":
                    reward = r_ang

                elif reward_mode == "ccc_xy":
                    reward = ccc_xy

                elif reward_mode == "angular+ccc_xy":
                    reward = r_ang + ccc_xy

                elif reward_mode == "ccc_vec":
                    reward = ccc_vec

                elif reward_mode == "angular+ccc_vec":
                    reward = r_ang + ccc_vec

                else:
                    raise ValueError(f"Unknown reward_mode: {reward_mode}")

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"reward_mode = {reward_mode}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_xy={ccc_xy:.4f}\n")
                        f.write(f"CCC_vec={ccc_vec:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards


    @staticmethod
    def gaze_reward_global_ccc_vector_xy(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        # reward mode:
        # "angular"
        # "ccc_xy"
        # "angular+ccc_xy"
        # "ccc_vec"
        # "angular+ccc_vec"
        # reward_mode = kwargs.get("reward_mode", "angular+ccc_vec")
        reward_mode = kwargs.get("reward_mode", "ccc_xy+ccc_vec")

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        def to_vec(pitch, yaw):
            """
            pitch / yaw 单位：degree
            输出 3D gaze direction vector
            """
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)

            x = -np.cos(pitch) * np.sin(yaw)
            y = -np.sin(pitch)
            z = -np.cos(pitch) * np.cos(yaw)
            return np.array([x, y, z], dtype=np.float32)

        def compute_vector_ccc(pred_vecs, gt_vecs):
            """
            pred_vecs: list/array of shape [B, 3]
            gt_vecs:   list/array of shape [B, 3]
            做法：分别对 x/y/z 三维算 CCC，再取平均
            """
            pred = np.array(pred_vecs, dtype=np.float32)  # [B, 3]
            gt = np.array(gt_vecs, dtype=np.float32)      # [B, 3]

            cccs = []
            for d in range(3):
                cccs.append(compute_ccc(pred[:, d], gt[:, d]))

            return float(np.mean(cccs))

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch/yaw scalar space)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                ccc_xy = 0.5 * (ccc_p + ccc_y)

                # -------- Vector CCC (3D gaze direction space)
                pred_vecs = []
                gt_vecs = []

                for z in range(batch_size):
                    if z == i:
                        pred_vecs.append(to_vec(p_pred, y_pred))
                        gt_vecs.append(to_vec(p_gt, y_gt))
                    else:
                        pred_vecs.append(to_vec(batch_mean_pitch[z], batch_mean_yaw[z]))
                        gt_vecs.append(to_vec(gt_pitch[z], gt_yaw[z]))

                ccc_vec = compute_vector_ccc(pred_vecs, gt_vecs)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                if reward_mode == "angular":
                    reward = r_ang

                elif reward_mode == "ccc_xy":
                    reward = ccc_xy

                elif reward_mode == "angular+ccc_xy":
                    reward = r_ang + ccc_xy

                elif reward_mode == "ccc_vec":
                    reward = ccc_vec

                elif reward_mode == "angular+ccc_vec":
                    reward = r_ang + ccc_vec
                    
                elif reward_mode == "ccc_xy+ccc_vec":
                    reward = ccc_xy + ccc_vec
                else:
                    raise ValueError(f"Unknown reward_mode: {reward_mode}")

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"reward_mode = {reward_mode}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_xy={ccc_xy:.4f}\n")
                        f.write(f"CCC_vec={ccc_vec:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards


    @staticmethod
    def regression_reward_fundus(completions, solution, **kwargs):
        """
        Strong baseline regression reward for fundus (0–4):
        reward = 1.0 if pred == gt else 0.0
        """
        import re
        import os
        import random
        from datetime import datetime


        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.uniform(1, 100)
                return random.randint(0, 4)

        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                # -------- prediction --------
                content = completion[0]["content"]
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content
                pred = extract_first_number(pred_text)

                # -------- ground truth --------
                gt_matches = re.findall(r'<answer>(.*?)</answer>', sol)
                gt = int(round(float(gt_matches[-1])))

                # -------- reward --------
                if pred is None:
                    reward = 0.0
                else:
                    reward = 1.0 if pred == gt else 0.0

            except Exception:
                reward = 0.0
                pred = "ERROR"
                gt = "ERROR"

            rewards.append(float(reward))

            # -------- DEBUG --------
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path.replace(".txt", "_fundus_regression.txt"),
                        "a", encoding="utf-8") as f:
                    f.write(
                        f"------------- {current_time} Fundus Regression Reward -------------\n"
                        f"Prediction: {pred}\n"
                        f"Ground Truth: {gt}\n"
                        f"Reward: {reward}\n"
                    )

        return rewards


    @staticmethod
    def diversity_reward(completions, solution, **kwargs):
        """
        给每个 sample 的 n_gen generations 一个 diversity reward。
        奖励 generation 的标准差接近 target_std（默认=5）。
        """
        import re
        import torch
        import numpy as np
        import os
        from datetime import datetime

        device = kwargs.get("device", "cuda")
        n_gen = kwargs.get("num_generations", 8)

        # hyperparameters
        target_std = kwargs.get("target_std", 5.0)
        sigma = kwargs.get("sigma_std", 2.0)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./diversity_reward_log.txt")
        log_path_dbg = log_path.replace(".txt", "_div.txt")

        # ------------------------------------------------
        # 1. Parse predictions only
        # ------------------------------------------------
        preds = []
        for comp in completions:
            content = comp[0]["content"]
            try:
                s = re.findall(r"<answer>(.*?)</answer>", content)[-1]
                v = float(re.findall(r"[-+]?\d+", s)[0])
            except:
                v = 0.0
            preds.append(v)

        # group by sample
        preds_grouped = [preds[i:i+n_gen] for i in range(0, len(preds), n_gen)]

        rewards = []

        # ------------------------------------------------
        # 2. Compute std reward for each sample
        # ------------------------------------------------
        for idx, pg in enumerate(preds_grouped):

            t = torch.tensor(pg, dtype=torch.float32, device=device)
            std_pred = t.std(unbiased=False)

            # Gaussian reward for diversity
            # diversity_score = torch.exp(-((std_pred - target_std) ** 2) / (2 * sigma ** 2))
            # linear diversity reward
            diversity_score = torch.clamp(std_pred / target_std, min=0.0, max=1.0)

            # 赋值给每个 generation
            for _ in range(n_gen):
                rewards.append(float(diversity_score))

            # ------------------------- DEBUG -------------------------
            if DEBUG:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path_dbg, "a", encoding="utf-8") as f:
                    f.write(f"====== {current_time} Diversity Sample {idx} ======\n")
                    f.write(f"preds = {pg}\n")
                    f.write(f"std_pred = {std_pred.item():.4f}\n")
                    f.write(f"target_std = {target_std}\n")
                    f.write(f"sigma = {sigma}\n")
                    f.write(f"diversity_reward = {diversity_score.item():.6f}\n")
                    f.write("====================================================\n\n")

        return rewards



    @staticmethod
    def format_reward_age_multi(completions, **kwargs):
        """
        Validates structure: <answer> -> JSON -> {"ranking": [...], "ages": [...]}
        DEBUG: Logs detailed failure reasons to file.
        """
        import json
        import re
        import os
        from datetime import datetime

        rewards = []
        num_images = kwargs.get("num_images", 4)
        
        # Debug 配置
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

        def extract_json(s):
            # 1. 找 <answer> 标签
            match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
            if not match: return None, "No <answer> tag"
            # 2. 找 JSON 大括号
            m = re.search(r"\{[\s\S]*\}", match.group(1))
            if not m: return None, "No JSON in tag"
            try: 
                return json.loads(m.group(0)), "Success"
            except: 
                return None, "JSON Decode Error"

        def log_debug(status, reason, content, parsed=None):
            if not debug_mode: return
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    t = datetime.now().strftime("%H:%M:%S.%f")
                    f.write(f"\n[{t}] [FORMAT] {status} | Reason: {reason}\n")
                    if parsed: f.write(f"Parsed: {json.dumps(parsed)}\n")
                    f.write(f"Raw: {content[:100]}...\n") # 只记录前100字符避免刷屏
            except: pass

        for completion in completions:
            content = completion[0]["content"]
            obj, msg = extract_json(content)

            # 格式解析失败
            if obj is None or not isinstance(obj, dict):
                rewards.append(0.0)
                log_debug("FAIL", msg, content)
                continue

            # 关键字段检查
            if "ranking" not in obj or "ages" not in obj:
                rewards.append(0.0)
                log_debug("FAIL", f"Missing keys. Got: {list(obj.keys())}", content, obj)
                continue
            
            # 长度与类型简单检查
            r_list = obj["ranking"]
            a_list = obj["ages"]
            if not (isinstance(r_list, list) and len(r_list) == num_images):
                rewards.append(0.0)
                log_debug("FAIL", f"Ranking format error. Len: {len(r_list) if isinstance(r_list, list) else 'Not List'}", content, obj)
                continue
            if not (isinstance(a_list, list) and len(a_list) == num_images):
                rewards.append(0.0)
                log_debug("FAIL", f"Ages format error. Len: {len(a_list) if isinstance(a_list, list) else 'Not List'}", content, obj)
                continue

            # 索引合法性检查 (0 到 N-1)
            try:
                if set(r_list) != set(range(num_images)):
                    rewards.append(0.0)
                    log_debug("FAIL", f"Invalid Indices: {r_list}", content, obj)
                    continue
            except:
                rewards.append(0.0)
                log_debug("FAIL", "Index Type Error", content, obj)
                continue

            # 成功
            rewards.append(0.5) # 注意：你这里给的是 0.5
            log_debug("SUCCESS", "Perfect Format", content, obj)

        return rewards


    @staticmethod
    def regression_reward_age_multi(completions, solution, **kwargs):
        """
        Computes Age Error (MAE).
        DEBUG: Logs MAE and predicted vs GT ages.
        """
        import json, numpy as np, re, os
        from datetime import datetime

        rewards = []
        num_images = kwargs.get("num_images", 4)
        scale = 5.0  # 调节宽松度: 差5岁拿 ~0.36分, 差0岁拿 1.0分
        
        # Debug 配置
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

        def extract_json(s):
            match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
            if not match: return None
            m = re.search(r"\{[\s\S]*\}", match.group(1))
            if not m: return None
            try: return json.loads(m.group(0))
            except: return None

        def log_debug(status, mae, reward, p_ages, g_ages):
            if not debug_mode: return
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    t = datetime.now().strftime("%H:%M:%S.%f")
                    f.write(f"\n[{t}] [REGRESS] {status} | MAE: {mae:.2f} | Rew: {reward:.4f}\n")
                    f.write(f"Pred Ages: {p_ages}\n")
                    f.write(f"GT   Ages: {g_ages}\n")
            except: pass

        for completion, sol in zip(completions, solution):
            try:
                pred = extract_json(completion[0]["content"])
                
                # 处理 Solution (GT)
                if isinstance(sol, str):
                    m_gt = re.search(r"\{[\s\S]*\}", sol)
                    gt = json.loads(m_gt.group(0)) if m_gt else None
                else:
                    gt = sol
                
                if pred is None or gt is None:
                    rewards.append(0.0)
                    continue

                # 提取 Ages 数组
                pred_ages = pred.get("ages", [])
                gt_ages = gt.get("ages", [])

                if len(pred_ages) != num_images or len(gt_ages) != num_images:
                    rewards.append(0.0)
                    continue

                # 转 numpy 计算差异
                p_arr = np.array(pred_ages, dtype=np.float32)
                g_arr = np.array(gt_ages, dtype=np.float32)

                mae = np.mean(np.abs(p_arr - g_arr))

                # 你的逻辑：线性截断
                reward = max(0.0, 1.0 - 0.05 * mae)
                # 指数衰减奖励
                # reward = np.exp(-mae / scale)
                rewards.append(float(reward))
                
                # 记录成功日志
                log_debug("CALC", mae, reward, pred_ages, gt_ages)

            except Exception as e:
                rewards.append(0.0)
                # 记录异常日志
                if debug_mode:
                    with open(log_path, "a") as f:
                        f.write(f"[REGRESS] ERROR: {str(e)}\n")

        return rewards


    @staticmethod
    def rank_reward_mse(completions, solution, **kwargs):
        """
        Computes Ranking MSE.
        DEBUG: Logs MSE and ranking comparison.
        """
        import json, numpy as np, re, os
        from datetime import datetime

        rewards = []
        num_images = kwargs.get("num_images", 4)
        
        # Debug 配置
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

        def extract_json(s):
            match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
            if not match: return None
            m = re.search(r"\{[\s\S]*\}", match.group(1))
            if not m: return None
            try: return json.loads(m.group(0))
            except: return None

        def log_debug(status, mse, reward, p_rank, g_rank):
            if not debug_mode: return
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    t = datetime.now().strftime("%H:%M:%S.%f")
                    f.write(f"\n[{t}] [RANK] {status} | MSE: {mse:.2f} | Rew: {reward:.4f}\n")
                    f.write(f"Pred Rank: {p_rank}\n")
                    f.write(f"GT   Rank: {g_rank}\n")
            except: pass

        for completion, sol in zip(completions, solution):
            try:
                pred = extract_json(completion[0]["content"])
                
                if isinstance(sol, str):
                    m_gt = re.search(r"\{[\s\S]*\}", sol)
                    gt = json.loads(m_gt.group(0)) if m_gt else None
                else:
                    gt = sol

                if pred is None or gt is None:
                    rewards.append(0.0)
                    continue

                pred_rank = pred.get("ranking", [])
                gt_rank = gt.get("ranking", [])

                # 基础合法性检查
                if len(pred_rank) != num_images or len(gt_rank) != num_images:
                    rewards.append(0.0)
                    continue
                
                # 必须是有效的索引排列
                if set(pred_rank) != set(range(num_images)):
                    rewards.append(0.0)
                    continue

                # ！！！关键修复：必须转为 np.array 才能相减，列表相减会报错！！！
                p_arr = np.array(pred_rank, dtype=np.float32)
                g_arr = np.array(gt_rank, dtype=np.float32)

                mse = np.mean((p_arr - g_arr) ** 2)

                # 负 MSE，比 exp 稳定得多
                tau = kwargs.get("rank_tau", 1.0)
                # reward = max(0.0, 1.0 - 0.1 * mse)
                reward = float(np.exp(-mse / tau))
                rewards.append(float(reward))
                
                # 记录成功日志
                log_debug("CALC", mse, reward, pred_rank, gt_rank)

            except Exception as e:
                rewards.append(0.0)
                if debug_mode:
                    with open(log_path, "a") as f:
                        f.write(f"[RANK] ERROR: {str(e)}\n")

        return rewards
    



    @staticmethod
    def format_reward_age(completions, **kwargs):
        """Check if the output contains a valid <answer> tag with a number in [1, 100]."""
        import re
        import os
        from datetime import datetime

        # 合法年龄范围
        # for AgeDB, IMDB-WIKI and Movie
        MIN_AGE = 0.0
        MAX_AGE = 100.0

        # for BoneAge
        # MIN_AGE = 1.0
        # MAX_AGE = 228.0


        # 先抓 answer 里的数字
        pattern = r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>"

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        # Debug
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH")

        if debug_mode:
            f = open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8")
            f.write(f"\n------------- {current_time} Format reward -------------\n")

        for content in completion_contents:
            match = re.search(pattern, content, re.DOTALL)

            if not match:
                # ❌ 没有 <answer> 或格式不对
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: format error\n")
                continue

            # 解析数值
            try:
                value = float(match.group(1))
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: parse error\n")
                continue

            # 数值范围检查
            if MIN_AGE <= value <= MAX_AGE:
                rewards.append(0.5)   # ✅ 合法格式 + 合法数值
                # rewards.append(0.1)   # ✅ 合法格式 + 合法数值
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ VALID: value={value}\n")
            else:
                rewards.append(0.0)   # ❌ 数值越界
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ INVALID: value={value} out of range\n")

        if debug_mode:
            f.close()

        return rewards

    @staticmethod
    def format_reward_age_noformat(completions, **kwargs):
        """
        Format reward WITHOUT <answer> tag:
        - 只要文本中包含一个合法数字（在范围内）即可
        - 可选：鼓励“单数字输出”
        """

        import re
        import os
        from datetime import datetime

        # ---------------------------
        # Range
        # ---------------------------
        MIN_AGE = 0.0
        MAX_AGE = 100.0

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        # Debug
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH")

        if debug_mode:
            f = open(log_path.replace(".txt", "_format_no_tag.txt"), "a", encoding="utf-8")
            f.write(f"\n------------- {current_time} Format reward (no tag) -------------\n")

        for content in completion_contents:

            # 提取所有数字
            nums = re.findall(r'-?\d+(?:\.\d+)?', content)

            if len(nums) == 0:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: no number found\n")
                continue

            # 取最后一个（更接近最终答案）
            try:
                value = float(nums[-1])
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: parse error\n")
                continue

            # ---------------------------
            # 核心逻辑
            # ---------------------------
            if MIN_AGE <= value <= MAX_AGE:

                # ✅ 基础奖励
                reward = 0.5

                # ⭐ 可选增强：鼓励“只输出一个数字”
                #if len(nums) == 1:
                #    reward += 0.2   # 更干净的输出

                rewards.append(reward)

                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ VALID: value={value}, nums={nums}, reward={reward}\n")

            else:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ INVALID: value={value} out of range\n")

        if debug_mode:
            f.close()

        return rewards

    @staticmethod
    def format_reward_boneage(completions, **kwargs):
        """Check if the output contains a valid <answer> tag with a number in [1, 100]."""
        import re
        import os
        from datetime import datetime

        # 合法年龄范围
        # for AgeDB, IMDB-WIKI and Movie
        # MIN_AGE = 0.0
        # MAX_AGE = 100.0

        # for BoneAge
        MIN_AGE = 1.0
        MAX_AGE = 228.0


        # 先抓 answer 里的数字
        pattern = r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>"

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []

        # Debug
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH")

        if debug_mode:
            f = open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8")
            f.write(f"\n------------- {current_time} Format reward -------------\n")

        for content in completion_contents:
            match = re.search(pattern, content, re.DOTALL)

            if not match:
                # ❌ 没有 <answer> 或格式不对
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: format error\n")
                continue

            # 解析数值
            try:
                value = float(match.group(1))
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write("→ INVALID: parse error\n")
                continue

            # 数值范围检查
            if MIN_AGE <= value <= MAX_AGE:
                rewards.append(0.5)   # ✅ 合法格式 + 合法数值
                # rewards.append(0.1)   # ✅ 合法格式 + 合法数值
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ VALID: value={value}\n")
            else:
                rewards.append(0.0)   # ❌ 数值越界
                if debug_mode:
                    f.write(f"Content: {content}\n")
                    f.write(f"→ INVALID: value={value} out of range\n")

        if debug_mode:
            f.close()

        return rewards


    @staticmethod
    def format_reward_gaze(completions, **kwargs):
        """
        Format reward for gaze estimation:
        Require: <answer>pitch, yaw</answer>
        """

        import re
        import os
        from datetime import datetime

        # ===== range（用你的 empirical）=====
        MIN_PITCH, MAX_PITCH = -19.0, 1.0
        MIN_YAW, MAX_YAW     = -17.0, 16.0

        completion_contents = [c[0]["content"] for c in completions]
        rewards = []

        # ===== debug =====
        debug_mode = os.getenv("DEBUG_MODE") == "true"
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_format_gaze.txt")

        if debug_mode:
            f = open(log_path, "a", encoding="utf-8")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            f.write(f"\n===== {current_time} GAZE FORMAT =====\n")

        # ===== main =====
        for content in completion_contents:

            # 1️⃣ strict answer match
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

            if not match:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"\nContent: {content}\n→ INVALID: no <answer>\n")
                continue

            answer_text = match.group(1).strip()

            # 2️⃣ extract numbers
            nums = re.findall(r'-?\d+(?:\.\d+)?', answer_text)

            if len(nums) < 2:
                # 只有一个数 → 弱奖励
                rewards.append(0.2)
                if debug_mode:
                    f.write(f"\nContent: {content}\n→ PARTIAL: only one number\n")
                continue

            try:
                pitch = float(nums[0])
                yaw   = float(nums[1])
            except:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f"\nContent: {content}\n→ INVALID: parse error\n")
                continue

            # 3️⃣ check range
            in_pitch = MIN_PITCH <= pitch <= MAX_PITCH
            in_yaw   = MIN_YAW <= yaw <= MAX_YAW

            if in_pitch and in_yaw:
                rewards.append(1.0)   # ✅ 完全正确
                if debug_mode:
                    f.write(f"\nContent: {content}\n→ VALID: ({pitch:.2f}, {yaw:.2f})\n")

            else:
                rewards.append(0.5)   # ⚠️ 格式对但越界
                if debug_mode:
                    f.write(f"\nContent: {content}\n→ OUT-OF-RANGE: ({pitch:.2f}, {yaw:.2f})\n")

        if debug_mode:
            f.close()

        return rewards


    @staticmethod
    def gaze_reward_global_ccc_vector(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        # reward mode:
        # "angular"
        # "ccc_xy"
        # "angular+ccc_xy"
        # "ccc_vec"
        # "angular+ccc_vec"
        # reward_mode = kwargs.get("reward_mode", "angular+ccc_vec")
        reward_mode = kwargs.get("reward_mode", "ccc_vec")

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            """
            从文本中提取两个浮点数。
            优先假设 text 已经是 <answer> 内部内容；若不是，也可直接抽数字。
            """
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            # fallback，避免训练直接炸
            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def compute_ccc(x, y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            mu_x = x.mean()
            mu_y = y.mean()
            var_x = x.var()
            var_y = y.var()
            cov_xy = np.mean((x - mu_x) * (y - mu_y))

            denom = var_x + var_y + (mu_x - mu_y) ** 2
            if denom == 0:
                return 0.0

            ccc = (2 * cov_xy) / denom
            if np.isnan(ccc):
                return 0.0

            return float(ccc)

        def to_vec(pitch, yaw):
            """
            pitch / yaw 单位：degree
            输出 3D gaze direction vector
            """
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)

            x = -np.cos(pitch) * np.sin(yaw)
            y = -np.sin(pitch)
            z = -np.cos(pitch) * np.cos(yaw)
            return np.array([x, y, z], dtype=np.float32)

        def compute_vector_ccc(pred_vecs, gt_vecs):
            """
            pred_vecs: list/array of shape [B, 3]
            gt_vecs:   list/array of shape [B, 3]
            做法：分别对 x/y/z 三维算 CCC，再取平均
            """
            pred = np.array(pred_vecs, dtype=np.float32)  # [B, 3]
            gt = np.array(gt_vecs, dtype=np.float32)      # [B, 3]

            cccs = []
            for d in range(3):
                cccs.append(compute_ccc(pred[:, d], gt[:, d]))

            return float(np.mean(cccs))

        # =========================
        # angular reward
        # =========================
        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            """
            pitch / yaw 单位：degree
            返回 reward in (0,1]
            """
            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []
        batch_mean_pitch, batch_mean_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

            t_p = torch.tensor(cur_pitch, dtype=torch.float32, device=device)
            t_y = torch.tensor(cur_yaw, dtype=torch.float32, device=device)

            batch_mean_pitch.append(t_p.mean().item())
            batch_mean_yaw.append(t_y.mean().item())

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                # -------- CCC (pitch/yaw scalar space)
                pred_list_p = [p_pred] + [batch_mean_pitch[z] for z in range(batch_size) if z != i]
                gt_list_p = [p_gt] + [gt_pitch[z] for z in range(batch_size) if z != i]
                ccc_p = compute_ccc(pred_list_p, gt_list_p)

                pred_list_y = [y_pred] + [batch_mean_yaw[z] for z in range(batch_size) if z != i]
                gt_list_y = [y_gt] + [gt_yaw[z] for z in range(batch_size) if z != i]
                ccc_y = compute_ccc(pred_list_y, gt_list_y)

                ccc_xy = 0.5 * (ccc_p + ccc_y)

                # -------- Vector CCC (3D gaze direction space)
                pred_vecs = []
                gt_vecs = []

                for z in range(batch_size):
                    if z == i:
                        pred_vecs.append(to_vec(p_pred, y_pred))
                        gt_vecs.append(to_vec(p_gt, y_gt))
                    else:
                        pred_vecs.append(to_vec(batch_mean_pitch[z], batch_mean_yaw[z]))
                        gt_vecs.append(to_vec(gt_pitch[z], gt_yaw[z]))

                ccc_vec = compute_vector_ccc(pred_vecs, gt_vecs)

                # -------- angular
                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt)

                # -------- combine
                if reward_mode == "angular":
                    reward = r_ang

                elif reward_mode == "ccc_xy":
                    reward = ccc_xy

                elif reward_mode == "angular+ccc_xy":
                    reward = r_ang + ccc_xy

                elif reward_mode == "ccc_vec":
                    reward = ccc_vec

                elif reward_mode == "angular+ccc_vec":
                    reward = r_ang + ccc_vec

                else:
                    raise ValueError(f"Unknown reward_mode: {reward_mode}")

                rewards.append(float(reward))

                # debug
                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"reward_mode = {reward_mode}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"CCC_p={ccc_p:.4f}, CCC_y={ccc_y:.4f}, CCC_xy={ccc_xy:.4f}\n")
                        f.write(f"CCC_vec={ccc_vec:.4f}\n")
                        f.write(f"Angular={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards


        @staticmethod
        def regression_reward_fundus(completions, solution, **kwargs):
            """
            Strong baseline regression reward for fundus (0–4):
            reward = 1.0 if pred == gt else 0.0
            """
            import re
            import os
            import random
            from datetime import datetime


            def extract_first_number(model_answer):
                match = re.search(r'-?\d+(\.\d+)?', model_answer)
                if match:
                    return float(match.group())
                else:
                    # return random.uniform(1, 100)
                    return random.randint(0, 4)

            rewards = []

            for completion, sol in zip(completions, solution):
                try:
                    # -------- prediction --------
                    content = completion[0]["content"]
                    pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    pred_text = pred_matches[-1].strip() if pred_matches else content
                    pred = extract_first_number(pred_text)

                    # -------- ground truth --------
                    gt_matches = re.findall(r'<answer>(.*?)</answer>', sol)
                    gt = int(round(float(gt_matches[-1])))

                    # -------- reward --------
                    if pred is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if pred == gt else 0.0

                except Exception:
                    reward = 0.0
                    pred = "ERROR"
                    gt = "ERROR"

                rewards.append(float(reward))

                # -------- DEBUG --------
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path.replace(".txt", "_fundus_regression.txt"),
                            "a", encoding="utf-8") as f:
                        f.write(
                            f"------------- {current_time} Fundus Regression Reward -------------\n"
                            f"Prediction: {pred}\n"
                            f"Ground Truth: {gt}\n"
                            f"Reward: {reward}\n"
                        )

            return rewards


        @staticmethod
        def diversity_reward(completions, solution, **kwargs):
            """
            给每个 sample 的 n_gen generations 一个 diversity reward。
            奖励 generation 的标准差接近 target_std（默认=5）。
            """
            import re
            import torch
            import numpy as np
            import os
            from datetime import datetime

            device = kwargs.get("device", "cuda")
            n_gen = kwargs.get("num_generations", 8)

            # hyperparameters
            target_std = kwargs.get("target_std", 5.0)
            sigma = kwargs.get("sigma_std", 2.0)

            DEBUG = (os.getenv("DEBUG_MODE") == "true")
            log_path = os.getenv("LOG_PATH", "./diversity_reward_log.txt")
            log_path_dbg = log_path.replace(".txt", "_div.txt")

            # ------------------------------------------------
            # 1. Parse predictions only
            # ------------------------------------------------
            preds = []
            for comp in completions:
                content = comp[0]["content"]
                try:
                    s = re.findall(r"<answer>(.*?)</answer>", content)[-1]
                    v = float(re.findall(r"[-+]?\d+", s)[0])
                except:
                    v = 0.0
                preds.append(v)

            # group by sample
            preds_grouped = [preds[i:i+n_gen] for i in range(0, len(preds), n_gen)]

            rewards = []

            # ------------------------------------------------
            # 2. Compute std reward for each sample
            # ------------------------------------------------
            for idx, pg in enumerate(preds_grouped):

                t = torch.tensor(pg, dtype=torch.float32, device=device)
                std_pred = t.std(unbiased=False)

                # Gaussian reward for diversity
                # diversity_score = torch.exp(-((std_pred - target_std) ** 2) / (2 * sigma ** 2))
                # linear diversity reward
                diversity_score = torch.clamp(std_pred / target_std, min=0.0, max=1.0)

                # 赋值给每个 generation
                for _ in range(n_gen):
                    rewards.append(float(diversity_score))

                # ------------------------- DEBUG -------------------------
                if DEBUG:
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    with open(log_path_dbg, "a", encoding="utf-8") as f:
                        f.write(f"====== {current_time} Diversity Sample {idx} ======\n")
                        f.write(f"preds = {pg}\n")
                        f.write(f"std_pred = {std_pred.item():.4f}\n")
                        f.write(f"target_std = {target_std}\n")
                        f.write(f"sigma = {sigma}\n")
                        f.write(f"diversity_reward = {diversity_score.item():.6f}\n")
                        f.write("====================================================\n\n")

            return rewards



        @staticmethod
        def format_reward_age_multi(completions, **kwargs):
            """
            Validates structure: <answer> -> JSON -> {"ranking": [...], "ages": [...]}
            DEBUG: Logs detailed failure reasons to file.
            """
            import json
            import re
            import os
            from datetime import datetime

            rewards = []
            num_images = kwargs.get("num_images", 4)
            
            # Debug 配置
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

            def extract_json(s):
                # 1. 找 <answer> 标签
                match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
                if not match: return None, "No <answer> tag"
                # 2. 找 JSON 大括号
                m = re.search(r"\{[\s\S]*\}", match.group(1))
                if not m: return None, "No JSON in tag"
                try: 
                    return json.loads(m.group(0)), "Success"
                except: 
                    return None, "JSON Decode Error"

            def log_debug(status, reason, content, parsed=None):
                if not debug_mode: return
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        t = datetime.now().strftime("%H:%M:%S.%f")
                        f.write(f"\n[{t}] [FORMAT] {status} | Reason: {reason}\n")
                        if parsed: f.write(f"Parsed: {json.dumps(parsed)}\n")
                        f.write(f"Raw: {content[:100]}...\n") # 只记录前100字符避免刷屏
                except: pass

            for completion in completions:
                content = completion[0]["content"]
                obj, msg = extract_json(content)

                # 格式解析失败
                if obj is None or not isinstance(obj, dict):
                    rewards.append(0.0)
                    log_debug("FAIL", msg, content)
                    continue

                # 关键字段检查
                if "ranking" not in obj or "ages" not in obj:
                    rewards.append(0.0)
                    log_debug("FAIL", f"Missing keys. Got: {list(obj.keys())}", content, obj)
                    continue
                
                # 长度与类型简单检查
                r_list = obj["ranking"]
                a_list = obj["ages"]
                if not (isinstance(r_list, list) and len(r_list) == num_images):
                    rewards.append(0.0)
                    log_debug("FAIL", f"Ranking format error. Len: {len(r_list) if isinstance(r_list, list) else 'Not List'}", content, obj)
                    continue
                if not (isinstance(a_list, list) and len(a_list) == num_images):
                    rewards.append(0.0)
                    log_debug("FAIL", f"Ages format error. Len: {len(a_list) if isinstance(a_list, list) else 'Not List'}", content, obj)
                    continue

                # 索引合法性检查 (0 到 N-1)
                try:
                    if set(r_list) != set(range(num_images)):
                        rewards.append(0.0)
                        log_debug("FAIL", f"Invalid Indices: {r_list}", content, obj)
                        continue
                except:
                    rewards.append(0.0)
                    log_debug("FAIL", "Index Type Error", content, obj)
                    continue

                # 成功
                rewards.append(0.5) # 注意：你这里给的是 0.5
                log_debug("SUCCESS", "Perfect Format", content, obj)

            return rewards


        @staticmethod
        def regression_reward_age_multi(completions, solution, **kwargs):
            """
            Computes Age Error (MAE).
            DEBUG: Logs MAE and predicted vs GT ages.
            """
            import json, numpy as np, re, os
            from datetime import datetime

            rewards = []
            num_images = kwargs.get("num_images", 4)
            scale = 5.0  # 调节宽松度: 差5岁拿 ~0.36分, 差0岁拿 1.0分
            
            # Debug 配置
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

            def extract_json(s):
                match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
                if not match: return None
                m = re.search(r"\{[\s\S]*\}", match.group(1))
                if not m: return None
                try: return json.loads(m.group(0))
                except: return None

            def log_debug(status, mae, reward, p_ages, g_ages):
                if not debug_mode: return
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        t = datetime.now().strftime("%H:%M:%S.%f")
                        f.write(f"\n[{t}] [REGRESS] {status} | MAE: {mae:.2f} | Rew: {reward:.4f}\n")
                        f.write(f"Pred Ages: {p_ages}\n")
                        f.write(f"GT   Ages: {g_ages}\n")
                except: pass

            for completion, sol in zip(completions, solution):
                try:
                    pred = extract_json(completion[0]["content"])
                    
                    # 处理 Solution (GT)
                    if isinstance(sol, str):
                        m_gt = re.search(r"\{[\s\S]*\}", sol)
                        gt = json.loads(m_gt.group(0)) if m_gt else None
                    else:
                        gt = sol
                    
                    if pred is None or gt is None:
                        rewards.append(0.0)
                        continue

                    # 提取 Ages 数组
                    pred_ages = pred.get("ages", [])
                    gt_ages = gt.get("ages", [])

                    if len(pred_ages) != num_images or len(gt_ages) != num_images:
                        rewards.append(0.0)
                        continue

                    # 转 numpy 计算差异
                    p_arr = np.array(pred_ages, dtype=np.float32)
                    g_arr = np.array(gt_ages, dtype=np.float32)

                    mae = np.mean(np.abs(p_arr - g_arr))

                    # 你的逻辑：线性截断
                    reward = max(0.0, 1.0 - 0.05 * mae)
                    # 指数衰减奖励
                    # reward = np.exp(-mae / scale)
                    rewards.append(float(reward))
                    
                    # 记录成功日志
                    log_debug("CALC", mae, reward, pred_ages, gt_ages)

                except Exception as e:
                    rewards.append(0.0)
                    # 记录异常日志
                    if debug_mode:
                        with open(log_path, "a") as f:
                            f.write(f"[REGRESS] ERROR: {str(e)}\n")

            return rewards


        @staticmethod
        def rank_reward_mse(completions, solution, **kwargs):
            """
            Computes Ranking MSE.
            DEBUG: Logs MSE and ranking comparison.
            """
            import json, numpy as np, re, os
            from datetime import datetime

            rewards = []
            num_images = kwargs.get("num_images", 4)
            
            # Debug 配置
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH", "debug_rewards.txt")

            def extract_json(s):
                match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL)
                if not match: return None
                m = re.search(r"\{[\s\S]*\}", match.group(1))
                if not m: return None
                try: return json.loads(m.group(0))
                except: return None

            def log_debug(status, mse, reward, p_rank, g_rank):
                if not debug_mode: return
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        t = datetime.now().strftime("%H:%M:%S.%f")
                        f.write(f"\n[{t}] [RANK] {status} | MSE: {mse:.2f} | Rew: {reward:.4f}\n")
                        f.write(f"Pred Rank: {p_rank}\n")
                        f.write(f"GT   Rank: {g_rank}\n")
                except: pass

            for completion, sol in zip(completions, solution):
                try:
                    pred = extract_json(completion[0]["content"])
                    
                    if isinstance(sol, str):
                        m_gt = re.search(r"\{[\s\S]*\}", sol)
                        gt = json.loads(m_gt.group(0)) if m_gt else None
                    else:
                        gt = sol

                    if pred is None or gt is None:
                        rewards.append(0.0)
                        continue

                    pred_rank = pred.get("ranking", [])
                    gt_rank = gt.get("ranking", [])

                    # 基础合法性检查
                    if len(pred_rank) != num_images or len(gt_rank) != num_images:
                        rewards.append(0.0)
                        continue
                    
                    # 必须是有效的索引排列
                    if set(pred_rank) != set(range(num_images)):
                        rewards.append(0.0)
                        continue

                    # ！！！关键修复：必须转为 np.array 才能相减，列表相减会报错！！！
                    p_arr = np.array(pred_rank, dtype=np.float32)
                    g_arr = np.array(gt_rank, dtype=np.float32)

                    mse = np.mean((p_arr - g_arr) ** 2)

                    # 负 MSE，比 exp 稳定得多
                    tau = kwargs.get("rank_tau", 1.0)
                    # reward = max(0.0, 1.0 - 0.1 * mse)
                    reward = float(np.exp(-mse / tau))
                    rewards.append(float(reward))
                    
                    # 记录成功日志
                    log_debug("CALC", mse, reward, pred_rank, gt_rank)

                except Exception as e:
                    rewards.append(0.0)
                    if debug_mode:
                        with open(log_path, "a") as f:
                            f.write(f"[RANK] ERROR: {str(e)}\n")

            return rewards
        



        @staticmethod
        def format_reward_age(completions, **kwargs):
            """Check if the output contains a valid <answer> tag with a number in [1, 100]."""
            import re
            import os
            from datetime import datetime

            # 合法年龄范围
            # for AgeDB, IMDB-WIKI and Movie
            MIN_AGE = 0.0
            MAX_AGE = 100.0

            # for BoneAge
            # MIN_AGE = 1.0
            # MAX_AGE = 228.0


            # 先抓 answer 里的数字
            pattern = r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>"

            completion_contents = [completion[0]["content"] for completion in completions]
            rewards = []

            # Debug
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH")

            if debug_mode:
                f = open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8")
                f.write(f"\n------------- {current_time} Format reward -------------\n")

            for content in completion_contents:
                match = re.search(pattern, content, re.DOTALL)

                if not match:
                    # ❌ 没有 <answer> 或格式不对
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write("→ INVALID: format error\n")
                    continue

                # 解析数值
                try:
                    value = float(match.group(1))
                except Exception:
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write("→ INVALID: parse error\n")
                    continue

                # 数值范围检查
                if MIN_AGE <= value <= MAX_AGE:
                    rewards.append(0.5)   # ✅ 合法格式 + 合法数值
                    # rewards.append(0.1)   # ✅ 合法格式 + 合法数值
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write(f"→ VALID: value={value}\n")
                else:
                    rewards.append(0.0)   # ❌ 数值越界
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write(f"→ INVALID: value={value} out of range\n")

            if debug_mode:
                f.close()

            return rewards


        @staticmethod
        def format_reward_boneage(completions, **kwargs):
            """Check if the output contains a valid <answer> tag with a number in [1, 100]."""
            import re
            import os
            from datetime import datetime

            # 合法年龄范围
            # for AgeDB, IMDB-WIKI and Movie
            # MIN_AGE = 0.0
            # MAX_AGE = 100.0

            # for BoneAge
            MIN_AGE = 1.0
            MAX_AGE = 228.0


            # 先抓 answer 里的数字
            pattern = r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>"

            completion_contents = [completion[0]["content"] for completion in completions]
            rewards = []

            # Debug
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH")

            if debug_mode:
                f = open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8")
                f.write(f"\n------------- {current_time} Format reward -------------\n")

            for content in completion_contents:
                match = re.search(pattern, content, re.DOTALL)

                if not match:
                    # ❌ 没有 <answer> 或格式不对
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write("→ INVALID: format error\n")
                    continue

                # 解析数值
                try:
                    value = float(match.group(1))
                except Exception:
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write("→ INVALID: parse error\n")
                    continue

                # 数值范围检查
                if MIN_AGE <= value <= MAX_AGE:
                    rewards.append(0.5)   # ✅ 合法格式 + 合法数值
                    # rewards.append(0.1)   # ✅ 合法格式 + 合法数值
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write(f"→ VALID: value={value}\n")
                else:
                    rewards.append(0.0)   # ❌ 数值越界
                    if debug_mode:
                        f.write(f"Content: {content}\n")
                        f.write(f"→ INVALID: value={value} out of range\n")

            if debug_mode:
                f.close()

            return rewards


        @staticmethod
        def format_reward_gaze(completions, **kwargs):
            """
            Format reward for gaze estimation:
            Require: <answer>pitch, yaw</answer>
            """

            import re
            import os
            from datetime import datetime

            # ===== range（用你的 empirical）=====
            MIN_PITCH, MAX_PITCH = -19.0, 1.0
            MIN_YAW, MAX_YAW     = -17.0, 16.0

            completion_contents = [c[0]["content"] for c in completions]
            rewards = []

            # ===== debug =====
            debug_mode = os.getenv("DEBUG_MODE") == "true"
            log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_format_gaze.txt")

            if debug_mode:
                f = open(log_path, "a", encoding="utf-8")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f.write(f"\n===== {current_time} GAZE FORMAT =====\n")

            # ===== main =====
            for content in completion_contents:

                # 1️⃣ strict answer match
                match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

                if not match:
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"\nContent: {content}\n→ INVALID: no <answer>\n")
                    continue

                answer_text = match.group(1).strip()

                # 2️⃣ extract numbers
                nums = re.findall(r'-?\d+(?:\.\d+)?', answer_text)

                if len(nums) < 2:
                    # 只有一个数 → 弱奖励
                    rewards.append(0.2)
                    if debug_mode:
                        f.write(f"\nContent: {content}\n→ PARTIAL: only one number\n")
                    continue

                try:
                    pitch = float(nums[0])
                    yaw   = float(nums[1])
                except:
                    rewards.append(0.0)
                    if debug_mode:
                        f.write(f"\nContent: {content}\n→ INVALID: parse error\n")
                    continue

                # 3️⃣ check range
                in_pitch = MIN_PITCH <= pitch <= MAX_PITCH
                in_yaw   = MIN_YAW <= yaw <= MAX_YAW

                if in_pitch and in_yaw:
                    rewards.append(1.0)   # ✅ 完全正确
                    if debug_mode:
                        f.write(f"\nContent: {content}\n→ VALID: ({pitch:.2f}, {yaw:.2f})\n")

                else:
                    rewards.append(0.5)   # ⚠️ 格式对但越界
                    if debug_mode:
                        f.write(f"\nContent: {content}\n→ OUT-OF-RANGE: ({pitch:.2f}, {yaw:.2f})\n")

            if debug_mode:
                f.close()

            return rewards


# ====================================================pure visualquality reward=========================================  only need to change the data range similar above
    @staticmethod
    def visualquality_reward(completions, solution, **kwargs):
        """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)
        import re
        import torch
        import numpy as np
        import os
        from datetime import datetime
        from torch.distributions import Normal
        import random

        # print("n_gen", n_gen)

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)
                # return random.uniform(1, 228)


        def fidelity_reward(pred1, pred2, var1, var2, gt, device):
            esp = 1e-6
            try:
                normal_dist = torch.distributions.Normal(0, 1)
                _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
                p = normal_dist.cdf(_cur)
            except:
                print("Meet Error ...")
                p = torch.tensor(0.5, dtype=torch.float32, device=device)
            
            reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
            return reward
        
        
        # extract solution
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                _cur = reshaped_solution[i][j]
                sol_match = re.search(r'<answer>(.*?)</answer>', _cur)
                ground_truth = sol_match.group(1).strip() if sol_match else sol_match.strip()
                reshaped_solution[i][j] = float(ground_truth)

        # extract content
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        # print("len(reshaped_content)",len(reshaped_content))

        batch_mean, batch_var, batch_pred = [], [], []
        for i in range(len(reshaped_content)): # batch
            cur_pred_list = []
            for j in range(len(reshaped_content[i])): # num generations
                try:
                    content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                    student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                    pred = extract_first_number(student_answer)
                except:
                    print("Meet Error ...")
                    pred = random.uniform(1, 100)
                    # pred = random.uniform(1, 228)
                cur_pred_list.append(pred)
            
            batch_pred.append(cur_pred_list)
            p = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            p_mean = torch.mean(p)
            p_var = torch.var(p)
            batch_mean.append([p_mean])
            batch_var.append([p_var])
        
        rewards = []
        for i in range(len(batch_pred)):
            for j in range(len(batch_pred[i])):
                _reward_sum, _count_idx = 0, 0
                for z in range(len(batch_mean)):
                    if z != i:
                        
                        input_pred1 = batch_pred[i][j]
                        input_pred2 = batch_mean[z][0]
                        input_var1 = batch_var[i][0]
                        input_var2 = batch_var[z][0]

                        if reshaped_solution[i][j] > reshaped_solution[z][0]:
                            input_gt = torch.tensor(1.0, dtype=torch.float32, device=device)
                        elif reshaped_solution[i][j] < reshaped_solution[z][0]:
                            input_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
                        else:
                            input_gt = torch.tensor(0.5, dtype=torch.float32, device=device)

                        _reward = fidelity_reward(
                            pred1=input_pred1, pred2=input_pred2, var1=input_var1, 
                            var2=input_var2, gt=input_gt, device=device
                        )

                        _reward_sum = _reward_sum + _reward
                        _count_idx = _count_idx + 1

                _cur_reward = _reward_sum / _count_idx
                # _cur_reward = _cur_reward / 2
                # _cur_reward = _cur_reward * 0.25
                rewards.append(_cur_reward)

                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    image_path = kwargs.get("image_path") if "image_path" in kwargs else None
                    problem = kwargs.get("problem")[0]
                    image_path = [image_path[i:i + n_gen] for i in range(0, len(image_path), n_gen)]

                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {_cur_reward} -------------\n")
                        f.write(f"accu_reward_method: {_cur_reward}\n")
                        f.write(f"image_path: {image_path[i][j]}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {reshaped_content[i][j]}\n")
                        f.write(f"Solution: {reshaped_solution[i][j]}\n") 

        return rewards


    @staticmethod
    def visualquality_reward_boneage(completions, solution, **kwargs):
        """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)
        import re
        import torch
        import numpy as np
        import os
        from datetime import datetime
        from torch.distributions import Normal
        import random

        # print("n_gen", n_gen)

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)


        def fidelity_reward(pred1, pred2, var1, var2, gt, device):
            esp = 1e-6
            try:
                normal_dist = torch.distributions.Normal(0, 1)
                _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
                p = normal_dist.cdf(_cur)
            except:
                print("Meet Error ...")
                p = torch.tensor(0.5, dtype=torch.float32, device=device)
            
            reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
            return reward
        
        
        # extract solution
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                _cur = reshaped_solution[i][j]
                sol_match = re.search(r'<answer>(.*?)</answer>', _cur)
                ground_truth = sol_match.group(1).strip() if sol_match else sol_match.strip()
                reshaped_solution[i][j] = float(ground_truth)

        # extract content
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        # print("len(reshaped_content)",len(reshaped_content))

        batch_mean, batch_var, batch_pred = [], [], []
        for i in range(len(reshaped_content)): # batch
            cur_pred_list = []
            for j in range(len(reshaped_content[i])): # num generations
                try:
                    content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                    student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                    pred = extract_first_number(student_answer)
                except:
                    print("Meet Error ...")
                    # pred = random.uniform(1, 100)
                    pred = random.uniform(1, 228)
                cur_pred_list.append(pred)
            
            batch_pred.append(cur_pred_list)
            p = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            p_mean = torch.mean(p)
            p_var = torch.var(p)
            batch_mean.append([p_mean])
            batch_var.append([p_var])
        
        rewards = []
        for i in range(len(batch_pred)):
            for j in range(len(batch_pred[i])):
                _reward_sum, _count_idx = 0, 0
                for z in range(len(batch_mean)):
                    if z != i:
                        
                        input_pred1 = batch_pred[i][j]
                        input_pred2 = batch_mean[z][0]
                        input_var1 = batch_var[i][0]
                        input_var2 = batch_var[z][0]

                        if reshaped_solution[i][j] > reshaped_solution[z][0]:
                            input_gt = torch.tensor(1.0, dtype=torch.float32, device=device)
                        elif reshaped_solution[i][j] < reshaped_solution[z][0]:
                            input_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
                        else:
                            input_gt = torch.tensor(0.5, dtype=torch.float32, device=device)

                        _reward = fidelity_reward(
                            pred1=input_pred1, pred2=input_pred2, var1=input_var1, 
                            var2=input_var2, gt=input_gt, device=device
                        )

                        _reward_sum = _reward_sum + _reward
                        _count_idx = _count_idx + 1

                _cur_reward = _reward_sum / _count_idx
                # _cur_reward = _cur_reward / 2
                # _cur_reward = _cur_reward * 0.25
                rewards.append(_cur_reward)

                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    image_path = kwargs.get("image_path") if "image_path" in kwargs else None
                    problem = kwargs.get("problem")[0]
                    image_path = [image_path[i:i + n_gen] for i in range(0, len(image_path), n_gen)]

                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {_cur_reward} -------------\n")
                        f.write(f"accu_reward_method: {_cur_reward}\n")
                        f.write(f"image_path: {image_path[i][j]}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {reshaped_content[i][j]}\n")
                        f.write(f"Solution: {reshaped_solution[i][j]}\n") 

        return rewards



# 引入了  regression-scale 参数， 固定阈值
    @staticmethod
    def age_reward_global_ccc(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime
        from scipy.stats import spearmanr

        # --------------------------------------------------
        # Load configs
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # for AgeDB, IMDB-WIKI and Movbie
                return random.uniform(1, 100)

                # for BoneAge
                # return random.uniform(1, 228)


        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        # GT per sample
        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var = [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])

        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])

        # --------------------------------------------------
        # Compute CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):               # for each sample
            
            for j in range(n_gen):               # for each generation

                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # Build lists
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                # Convert to numpy
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # Compute means and variances
                mu_x = x.mean()
                mu_y = y.mean()
                var_x = x.var()
                var_y = y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                # CCC formula
                denom = var_x + var_y + (mu_x - mu_y) ** 2
                if denom == 0:
                    ccc = 0.0
                else:
                    ccc = (2 * cov_xy) / denom

                # Avoid nan
                if np.isnan(ccc):
                    ccc = 0.0

                reward = ccc
                rewards.append(float(reward))

                # ---------------------------------------
                # Debug Log
                # ---------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ CCC Pair =================\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"GT_i = {gt_i}\n")
                        f.write(f"pred_i_j = {pred_i_j}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list   = {gt_list_cmp}\n")
                        f.write(f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n")
                        f.write(f"var_x={var_x:.3f}, var_y={var_y:.3f}\n")
                        f.write(f"cov_xy={cov_xy:.3f}\n")
                        f.write(f"CCC = {ccc:.4f}, Reward = {reward:.4f}\n")
                        # f.write(f"abs_err = {abs_err:.2f}\n")
                        # f.write(f"reg_gate = {reg_gate:.4f}\n")
                        # f.write(f"zone={'REG' if REG_LOW <= gt_i <= REG_HIGH else 'CCC'}\n")
                        # f.write(f"Base reward: {base_reward}\n")
                        # f.write(f"weight: {w}\n")
                        # f.write(f"weight_eff: {w_eff}\n")
                        f.write(f"final_reward = {reward:.4f}\n")
                        f.write("===========================================\n")

        return rewards

    @staticmethod
    def age_reward_global_ccc_noformat(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np

        # --------------------------------------------------
        # Load configs
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc_noformat.txt")

        # --------------------------------------------------
        # Helper: extract first number (NO <answer>)
        # --------------------------------------------------
        def extract_first_number(text):
            """
            直接从整个输出中抓第一个数字
            """
            match = re.search(r'-?\d+(?:\.\d+)?', text)
            if match:
                return float(match.group())
            else:
                # fallback（避免训练崩）
                return random.uniform(1, 100)

        # --------------------------------------------------
        # Parse GT（同样去掉 <answer> 依赖）
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_list = []
        for i in range(len(reshaped_solution)):
            # GT 只取第一个
            raw = reshaped_solution[i][0]
            gt = extract_first_number(raw)
            gt_list.append(gt)

        # --------------------------------------------------
        # Parse predictions（完全无 tag）
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean = [], []

        for i in range(len(reshaped_content)):
            cur_pred_list = []

            for j in range(len(reshaped_content[i])):
                text = reshaped_content[i][j]
                pred = extract_first_number(text)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)

            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append(t.mean().item())

        batch_size = len(batch_pred)

        # --------------------------------------------------
        # Compute CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):

                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # leave-one-out style
                pred_list = [pred_i_j] + [batch_mean[z] for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                mu_x = x.mean()
                mu_y = y.mean()
                var_x = x.var()
                var_y = y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                denom = var_x + var_y + (mu_x - mu_y) ** 2

                if denom == 0:
                    ccc = 0.0
                else:
                    ccc = (2 * cov_xy) / denom

                if np.isnan(ccc):
                    ccc = 0.0

                rewards.append(float(ccc))

                # Debug
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write(f"\n[i={i}, j={j}] pred={pred_i_j:.2f}, gt={gt_i:.2f}, ccc={ccc:.4f}")

        return rewards


# 引入了  regression-scale 参数， 固定阈值
    @staticmethod
    def age_reward_global_ccc_boneage(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime
        from scipy.stats import spearmanr

        # --------------------------------------------------
        # Load configs
        # --------------------------------------------------
        device = kwargs.get("device")
        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path_rank = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # for AgeDB, IMDB-WIKI and Movbie
                # return random.uniform(1, 100)

                # for BoneAge
                return random.uniform(1, 228)


        # --------------------------------------------------
        # Parse GT (reshape into [batch, n_gen])
        # --------------------------------------------------
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search(r'<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)

        # GT per sample
        gt_list = [sol[0] for sol in reshaped_solution]

        # --------------------------------------------------
        # Parse predictions
        # --------------------------------------------------
        contents = [completion[0]["content"] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pred, batch_mean, batch_var = [], [], []
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)

            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])

        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])

        # --------------------------------------------------
        # Compute CCC reward
        # --------------------------------------------------
        rewards = []

        for i in range(batch_size):               # for each sample
            
            for j in range(n_gen):               # for each generation

                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]

                # Build lists
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
                gt_list_cmp = [gt_i] + [gt_list[z] for z in range(batch_size) if z != i]

                # Convert to numpy
                x = np.array(pred_list, dtype=np.float32)
                y = np.array(gt_list_cmp, dtype=np.float32)

                # Compute means and variances
                mu_x = x.mean()
                mu_y = y.mean()
                var_x = x.var()
                var_y = y.var()
                cov_xy = np.mean((x - mu_x) * (y - mu_y))

                # CCC formula
                denom = var_x + var_y + (mu_x - mu_y) ** 2
                if denom == 0:
                    ccc = 0.0
                else:
                    ccc = (2 * cov_xy) / denom

                # Avoid nan
                if np.isnan(ccc):
                    ccc = 0.0

                reward = ccc
                rewards.append(float(reward))

                # ---------------------------------------
                # Debug Log
                # ---------------------------------------
                if DEBUG:
                    with open(log_path_rank, "a", encoding="utf-8") as f:
                        f.write("\n================ CCC Pair =================\n")
                        f.write(f"Sample i={i}, Gen j={j}\n")
                        f.write(f"GT_i = {gt_i}\n")
                        f.write(f"pred_i_j = {pred_i_j}\n")
                        f.write(f"Pred list = {pred_list}\n")
                        f.write(f"GT list   = {gt_list_cmp}\n")
                        f.write(f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n")
                        f.write(f"var_x={var_x:.3f}, var_y={var_y:.3f}\n")
                        f.write(f"cov_xy={cov_xy:.3f}\n")
                        f.write(f"CCC = {ccc:.4f}, Reward = {reward:.4f}\n")
                        # f.write(f"abs_err = {abs_err:.2f}\n")
                        # f.write(f"reg_gate = {reg_gate:.4f}\n")
                        # f.write(f"zone={'REG' if REG_LOW <= gt_i <= REG_HIGH else 'CCC'}\n")
                        # f.write(f"Base reward: {base_reward}\n")
                        # f.write(f"weight: {w}\n")
                        # f.write(f"weight_eff: {w_eff}\n")
                        f.write(f"final_reward = {reward:.4f}\n")
                        f.write("===========================================\n")

        return rewards



    @staticmethod
    def regression_reward(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)
                # return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    pred_str = random.uniform(1, 100)  # fallback
                    # pred_str = random.uniform(1, 228)  # fallback


                
                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                rel_error = abs(pred - gt)
                reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # abs_err = abs(pred - gt)
                # reward = max(0.0, 1.0 - abs_err / MAX_RANGE)



                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards


    @staticmethod
    def regression_reward_boneage(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    # pred_str = random.uniform(1, 100)  # fallback
                    pred_str = random.uniform(1, 228)  # fallback


                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                MAX_RANGE = 228.0   # BoneAge 最大月数
                abs_err = abs(pred - gt)
                reward = max(0.0, 1.0 - abs_err / MAX_RANGE)

                


                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards


    @staticmethod
    def regression_reward_boneage_weight2(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    # pred_str = random.uniform(1, 100)  # fallback
                    pred_str = random.uniform(1, 228)  # fallback


                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                MAX_RANGE = 228.0   # BoneAge 最大月数
                abs_err = abs(pred - gt)
                reward = max(0.0, 1.0 - abs_err / MAX_RANGE)
                
                reward = reward * 2.0

                


                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards

    @staticmethod
    def regression_reward_boneage_weight0_5(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    # pred_str = random.uniform(1, 100)  # fallback
                    pred_str = random.uniform(1, 228)  # fallback


                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                MAX_RANGE = 228.0   # BoneAge 最大月数
                abs_err = abs(pred - gt)
                reward = max(0.0, 1.0 - abs_err / MAX_RANGE)
                
                
                reward = reward * 0.5

                


                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards


    @staticmethod
    def regression_reward_boneage_weight0_7(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    # pred_str = random.uniform(1, 100)  # fallback
                    pred_str = random.uniform(1, 228)  # fallback


                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                MAX_RANGE = 228.0   # BoneAge 最大月数
                abs_err = abs(pred - gt)
                reward = max(0.0, 1.0 - abs_err / MAX_RANGE)
                
                
                reward = reward * 0.7

                


                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards


    @staticmethod
    def regression_reward_boneage_weight0_3(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                # return random.randint(1, 100)
                return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    # pred_str = random.uniform(1, 100)  # fallback
                    pred_str = random.uniform(1, 228)  # fallback


                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                MAX_RANGE = 228.0   # BoneAge 最大月数
                abs_err = abs(pred - gt)
                reward = max(0.0, 1.0 - abs_err / MAX_RANGE)
                
                
                reward = reward * 0.3

                


                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                # age = int(round(gt))          # or int(gt)
                # # age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                # w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                # rel_error = abs(pred - gt)
                # # base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                # reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    # f.write(f"Base reward: {base_reward}\n")
                    # f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards



    @staticmethod
    def regression_reward_agedb(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)
                # return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    pred_str = random.uniform(1, 100)  # fallback
                    # pred_str = random.uniform(1, 228)  # fallback


                
                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # abs_err = abs(pred - gt)
                # reward = max(0.0, 1.0 - abs_err / MAX_RANGE)



                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                age = int(round(gt))          # or int(gt)
                age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                w = AGE_BIN_WEIGHTS_agedb.get(age, 1.0)

                rel_error = abs(pred - gt)
                base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    f.write(f"Base reward: {base_reward}\n")
                    f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards

    @staticmethod
    def regression_reward_imdb_wiki(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)
                # return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    pred_str = random.uniform(1, 100)  # fallback
                    # pred_str = random.uniform(1, 228)  # fallback


                
                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # abs_err = abs(pred - gt)
                # reward = max(0.0, 1.0 - abs_err / MAX_RANGE)



                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                age = int(round(gt))          # or int(gt)
                age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                w = AGE_BIN_WEIGHTS_imdb_wiki.get(age, 1.0)

                rel_error = abs(pred - gt)
                base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    f.write(f"Base reward: {base_reward}\n")
                    f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards

    @staticmethod
    def regression_reward_movie(completions, solution, **kwargs):
        import re
        import numpy as np
        import os
        from datetime import datetime
        import json
        import random

        def extract_first_number(model_answer):
            match = re.search(r'-?\d+(\.\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.randint(1, 100)
                # return random.uniform(1, 228)

            
        rewards = []
        for completion, sol in zip(completions, solution):
            try:
                content = completion[0]["content"]
                # pred_str = re.findall(r'<answer>(.*?)</answer>', content)[-1].strip()



                # 提取 <answer>...</answer>
                pred_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                pred_text = pred_matches[-1].strip() if pred_matches else content.strip()

                pred_str = extract_first_number(pred_text)
                if pred_str is None:
                    pred_str = random.uniform(1, 100)  # fallback
                    # pred_str = random.uniform(1, 228)  # fallback


                
                sol_str = re.findall(r'<answer>(.*?)</answer>', sol)[-1].strip()

                pred = float(pred_str)
                gt = float(sol_str)


                # =========================this is for AgeDB, IMDB-WIKI and Movie=====================
                # rel_error = abs(pred - gt)
                # reward = max(0.0, 1 - rel_error * 0.1)


                # =========================this is for BoneAge only=====================
                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # abs_err = abs(pred - gt)
                # reward = max(0.0, 1.0 - abs_err / MAX_RANGE)



                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                age = int(round(gt))          # or int(gt)
                age = max(0, min(99, age))    # safety
                # age = max(1, min(228, age))    # safety

                w = AGE_BIN_WEIGHTS_movie.get(age, 1.0)

                rel_error = abs(pred - gt)
                base_reward = max(0.0, 1 - rel_error * 0.1)

                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

                reward = base_reward * w


            except Exception:
                reward = 0.0
                pred_str = "ERROR"
                sol_str = "ERROR"
                # print("error", "error")

            rewards.append(reward)
        
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                with open(log_path.replace(".txt", "_regression.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Regression reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Prediction: {pred_str}\n")
                    f.write(f"Ground Truth: {sol_str}\n")
                    f.write(f"Base reward: {base_reward}\n")
                    f.write(f"weight: {w}\n")
                    f.write(f"Final reward: {reward}\n")
        # print("reward_regression",rewards)
        return rewards


    @staticmethod
    def gaze_reward_regression(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np

        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_reg.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_reg_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def regression_reward(pred, gt, tau=5.0):
            err = abs(pred - gt)
            return float(np.exp(-err / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                r_pitch = regression_reward(p_pred, p_gt, tau=5.0)
                r_yaw = regression_reward(y_pred, y_gt, tau=5.0)

                reward = r_pitch + r_yaw

                rewards.append(float(reward))

                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"Pitch reward={r_pitch:.4f}, Yaw reward={r_yaw:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards
    

    @staticmethod
    def gaze_reward_regression_angle(completions, solution, **kwargs):
        import re
        import os
        import random
        import numpy as np

        n_gen = kwargs.get("num_generations", 4)

        DEBUG = (os.getenv("DEBUG_MODE") == "true")
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_reg_ang.txt")
        parse_log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_gaze_reg_ang_parse.txt")

        # =========================
        # helper
        # =========================
        def extract_two_numbers(text):
            nums = re.findall(r'-?\d+(?:\.\d+)?', text)

            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])

            if DEBUG:
                with open(parse_log_path, "a", encoding="utf-8") as f:
                    f.write("\n[PARSE FAIL]\n")
                    f.write(f"Raw text: {text}\n")
                    f.write(f"Extracted nums: {nums}\n")

            return random.uniform(-10, 0), random.uniform(-10, 10)

        def regression_reward(pred, gt, tau=5.0):
            err = abs(pred - gt)
            return float(np.exp(-err / tau))

        def angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1):
            def to_vec(pitch, yaw):
                pitch = np.radians(pitch)
                yaw = np.radians(yaw)

                x = -np.cos(pitch) * np.sin(yaw)
                y = -np.sin(pitch)
                z = -np.cos(pitch) * np.cos(yaw)
                return np.array([x, y, z], dtype=np.float32)

            v1 = to_vec(p_pred, y_pred)
            v2 = to_vec(p_gt, y_gt)

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angle = np.arccos(cos_sim)  # rad
            return float(np.exp(-angle / tau))

        # =========================
        # parse GT
        # =========================
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]

        gt_pitch, gt_yaw = [], []
        for i in range(len(reshaped_solution)):
            sol = reshaped_solution[i][0]
            match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            text = match.group(1).strip() if match else sol.strip()

            p, y = extract_two_numbers(text)
            gt_pitch.append(p)
            gt_yaw.append(y)

        # =========================
        # parse predictions
        # =========================
        contents = [c[0]["content"] for c in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

        batch_pitch, batch_yaw = [], []

        for i in range(len(reshaped_content)):
            cur_pitch, cur_yaw = [], []

            for j in range(n_gen):
                content = reshaped_content[i][j]
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                ans = match.group(1).strip() if match else content.strip()

                p, y = extract_two_numbers(ans)
                cur_pitch.append(p)
                cur_yaw.append(y)

            batch_pitch.append(cur_pitch)
            batch_yaw.append(cur_yaw)

        batch_size = len(batch_pitch)

        # =========================
        # compute reward
        # =========================
        rewards = []

        for i in range(batch_size):
            for j in range(n_gen):
                p_pred = batch_pitch[i][j]
                y_pred = batch_yaw[i][j]

                p_gt = gt_pitch[i]
                y_gt = gt_yaw[i]

                r_pitch = regression_reward(p_pred, p_gt, tau=5.0)
                r_yaw = regression_reward(y_pred, y_gt, tau=5.0)
                r_reg = r_pitch + r_yaw

                r_ang = angular_reward(p_pred, y_pred, p_gt, y_gt, tau=0.1)

                reward = r_reg + r_ang

                rewards.append(float(reward))

                if DEBUG:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\nSample {i}, Gen {j}\n")
                        f.write(f"GT: ({p_gt:.2f}, {y_gt:.2f})\n")
                        f.write(f"Pred: ({p_pred:.2f}, {y_pred:.2f})\n")
                        f.write(f"Pitch reward={r_pitch:.4f}, Yaw reward={r_yaw:.4f}\n")
                        f.write(f"Reg reward={r_reg:.4f}\n")
                        f.write(f"Angular reward={r_ang:.4f}\n")
                        f.write(f"Final reward={reward:.4f}\n")

        return rewards



    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "rank":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "reg":
                    # return Qwen2VLModule.age_reward
                    return Qwen2VLModule.rank_reward_mse
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "reg":
                    return Qwen2VLModule.format_reward_age
                    # return Qwen2VLModule.format_reward_fundus
                    # return Qwen2VLModule.format_reward_age_multi
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format-noformat":
            match task_type:
                case "reg":
                    return Qwen2VLModule.format_reward_age_noformat
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format-boneage":
            match task_type:
                case "reg":
                    return Qwen2VLModule.format_reward_boneage
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format-gaze":
            match task_type:
                case "reg":
                    return Qwen2VLModule.format_reward_gaze
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward
                    # return Qwen2VLModule.regression_reward_fundus
                    # return Qwen2VLModule.regression_reward_age_multi
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage_weihgt2":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_boneage_weight2
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage_weihgt0_5":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_boneage_weight0_5
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage_weihgt0_3":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_boneage_weight0_3
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage_weihgt0_7":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_boneage_weight0_7
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-agedb":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_agedb
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-imdb-wiki":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_imdb_wiki
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-movie":
            match task_type:
                case "reg":
                    return Qwen2VLModule.regression_reward_movie
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-gaze":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_regression
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-gaze-angle":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_regression_angle
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "visualquality":
            match task_type:
                case "reg":
                    return Qwen2VLModule.visualquality_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "visualquality-boneage":
            match task_type:
                case "reg":
                    return Qwen2VLModule.visualquality_reward_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "diversity":
            match task_type:
                case "reg":
                    return Qwen2VLModule.diversity_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank":
            match task_type:
                case "reg":
                    # return Qwen2VLModule.age_reward_global_ccc_memory_counter_direction
                    # return Qwen2VLModule.age_reward_global_ccc_with_memory
                    return Qwen2VLModule.age_reward_global_ccc
                    # return Qwen2VLModule.fundus_reward_global_ccc_with_memory
                    # return Qwen2VLModule.fundus_reward_global_ccc_interval
                    # return Qwen2VLModule.age_reward_global_ccc_spear
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_noformat":
            match task_type:
                case "reg":
                    return Qwen2VLModule.age_reward_global_ccc_noformat
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_boneage":
            match task_type:
                case "reg":
                    return Qwen2VLModule.age_reward_global_ccc_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_gaze_w_angle":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_global_ccc_w_angle
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_gaze_wo_angle":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_global_ccc_wo_angle
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_gaze_vector_w_angle":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_global_angular_w_ccc_vector
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_gaze_vector_only":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_global_ccc_vector
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_gaze_vector_xy":
            match task_type:
                case "reg":
                    return Qwen2VLModule.gaze_reward_global_ccc_vector_xy
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_diff_rank":
            match task_type:
                case "reg":
                    return Qwen2VLModule.age_reward_diff_ccc
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
