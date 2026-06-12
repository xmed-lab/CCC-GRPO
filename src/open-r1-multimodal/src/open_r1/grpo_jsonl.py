import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import GRPOConfig, VLMGRPOTrainer
from open_r1.vlm_modules import Qwen2VLModule


@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Colon-separated JSONL annotation paths"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Colon-separated image roots matching data_file_paths"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "ccc"],
        metadata={"help": "CCC-GRPO rewards: format and ccc"},
    )
    task_type: str = field(
        default=None,
        metadata={"help": "Dataset profile: agedb, imdb_wiki, imdb_movie, or boneage"},
    )
    max_pixels: Optional[int] = 12845056
    min_pixels: Optional[int] = 3136
    max_anyres_num: Optional[int] = 12
    is_reward_customized_from_vlm_module: bool = True


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def _resolve_image_path(image_folder, image_path):
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_folder, image_path)


def load_jsonl_dataset(data_file_paths, image_folders, question_template):
    data_files = data_file_paths.split(":")
    image_roots = image_folders.split(":")
    if len(data_files) != len(image_roots):
        raise ValueError("Each annotation file must have one matching image folder")

    records = []
    for data_file, image_root in zip(data_files, image_roots):
        with open(data_file, encoding="utf-8") as input_file:
            for line in input_file:
                item = json.loads(line)
                image_paths = item.get("image", [])
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                image_paths = [
                    _resolve_image_path(image_root, path) for path in image_paths
                ]

                conversations = item["conversations"]
                problem = conversations[0]["value"].replace("<image>", "").strip()
                solution = str(conversations[1]["value"]).strip()
                records.append(
                    {
                        "image_path": image_paths,
                        "problem": problem,
                        "solution": f"<answer> {solution} </answer>",
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    *(
                                        {"type": "image", "text": None}
                                        for _ in image_paths
                                    ),
                                    {
                                        "type": "text",
                                        "text": question_template.format(
                                            Question=problem
                                        ),
                                    },
                                ],
                            }
                        ],
                    }
                )
    return Dataset.from_list(records)


def main(script_args, training_args, model_args):
    if "qwen" not in model_args.model_name_or_path.lower():
        raise ValueError("This release supports Qwen2-VL and Qwen2.5-VL only")

    vlm_module = Qwen2VLModule()
    question_template = vlm_module.get_question_template(script_args.task_type)
    reward_funcs = [
        vlm_module.select_reward_func(name, script_args.task_type)
        for name in script_args.reward_funcs
    ]
    dataset = load_jsonl_dataset(
        script_args.data_file_paths,
        script_args.image_folders,
        question_template,
    )

    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
