import os
import random
import re
from functools import partial
from typing import Any, Union

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from trl.data_utils import maybe_apply_chat_template

from open_r1.vlm_modules.vlm_module import VLMBaseModule


# These ranges match the reward implementations used by the released training
# scripts. The source prompts remain in the dataset annotations.
DATASET_RANGES = {
    "agedb": (0.0, 100.0),
    "imdb_wiki": (0.0, 100.0),
    "imdb_movie": (0.0, 100.0),
    "boneage": (1.0, 228.0),
}


def _completion_texts(completions):
    return [completion[0]["content"] for completion in completions]


def _extract_number(text, low, high):
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else random.uniform(low, high)


def format_reward(completions, value_range, **kwargs):
    low, high = value_range
    pattern = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>", re.DOTALL)
    rewards = []

    for content in _completion_texts(completions):
        match = pattern.search(content)
        if not match:
            rewards.append(0.0)
            continue
        value = float(match.group(1))
        rewards.append(0.5 if low <= value <= high else 0.0)

    return rewards


def ccc_reward(completions, solution, value_range, **kwargs):
    """Batch-level leave-one-out CCC reward used by CCC-GRPO."""
    low, high = value_range
    num_generations = int(kwargs.get("num_generations", 4))
    if len(completions) % num_generations:
        raise ValueError("Completion count must be divisible by num_generations")

    grouped_solutions = [
        solution[i : i + num_generations]
        for i in range(0, len(solution), num_generations)
    ]
    targets = []
    for group in grouped_solutions:
        match = re.search(r"<answer>(.*?)</answer>", str(group[0]), re.DOTALL)
        targets.append(float(match.group(1).strip() if match else group[0]))

    contents = _completion_texts(completions)
    grouped_contents = [
        contents[i : i + num_generations]
        for i in range(0, len(contents), num_generations)
    ]
    predictions = []
    for group in grouped_contents:
        group_predictions = []
        for content in group:
            answers = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            answer = answers[-1].strip() if answers else content
            group_predictions.append(_extract_number(answer, low, high))
        predictions.append(group_predictions)

    group_means = [float(np.mean(group)) for group in predictions]
    rewards = []
    for sample_index, group in enumerate(predictions):
        for prediction in group:
            compared_predictions = [
                prediction if index == sample_index else group_means[index]
                for index in range(len(group_means))
            ]
            x = np.asarray(compared_predictions, dtype=np.float32)
            y = np.asarray(targets, dtype=np.float32)
            denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
            ccc = 0.0 if denominator == 0 else 2 * np.mean((x - x.mean()) * (y - y.mean())) / denominator
            rewards.append(0.0 if np.isnan(ccc) else float(ccc))

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "./debug_log.txt").replace(".txt", "_ccc.txt")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(f"{reward:.6f}" for reward in rewards) + "\n")

    return rewards


class Qwen2VLModule(VLMBaseModule):
    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2.5-VL" in model_id:
            return Qwen2_5_VLForConditionalGeneration
        if "Qwen2-VL" in model_id:
            return Qwen2VLForConditionalGeneration
        raise ValueError(f"Unsupported model: {model_id}")

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):
        return ["visual"]

    def get_custom_multimodal_keywords(self):
        return ["pixel_values", "image_grid_thw"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [("image_processor", "max_pixels"), ("image_processor", "min_pixels")]

    def prepare_prompt(
        self,
        processing_class,
        inputs: dict[str, Union[torch.Tensor, Any]],
    ):
        return [
            maybe_apply_chat_template(example, processing_class)["prompt"]
            for example in inputs
        ]

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ):
        kwargs = {
            "text": prompts_text,
            "return_tensors": return_tensors,
            "padding": padding,
            "padding_side": padding_side,
            "add_special_tokens": add_special_tokens,
        }
        if images:
            kwargs["images"] = images
        prompt_inputs = processing_class(**kwargs)
        additional_output = None
        if images:
            additional_output = [
                {"image_grid_thw": image_grid_thw}
                for image_grid_thw in prompt_inputs["image_grid_thw"]
            ]
        return prompt_inputs, additional_output

    @staticmethod
    def get_question_template(task_type: str):
        if task_type not in DATASET_RANGES:
            raise ValueError(f"Unsupported dataset profile: {task_type}")
        return "{Question}"

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if task_type not in DATASET_RANGES:
            raise ValueError(f"Unsupported dataset profile: {task_type}")
        value_range = DATASET_RANGES[task_type]
        if func == "format":
            return partial(format_reward, value_range=value_range)
        if func == "ccc":
            return partial(ccc_reward, value_range=value_range)
        raise ValueError(f"Unsupported reward function: {func}")
