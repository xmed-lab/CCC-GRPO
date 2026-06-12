import pathlib
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.vlm_modules import Qwen2VLModule
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()
tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_file_paths: str = field(default=None, metadata={'help': "Paths to data files, separated by ':'"})
    image_folders: str = field(default=None, metadata={'help': "Paths to image folders, separated by ':'"})
    val_split_ratio: float = field(default=0.0, metadata={'help': 'Ratio of validation split, default 0.0'})
    reward_funcs: list[str] = field(default_factory=lambda : ['format', 'global_rank'], metadata={'help': 'Reward functions: format, format-boneage, global_rank, global_rank_boneage'})
    max_pixels: Optional[int] = field(default=12845056, metadata={'help': 'Maximum number of pixels for the image (for QwenVL)'})
    min_pixels: Optional[int] = field(default=3136, metadata={'help': 'Minimum number of pixels for the image (for QwenVL)'})
    task_type: Optional[str] = field(default='reg', metadata={'help': 'Regression task type'})

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

def get_vlm_module(model_name_or_path):
    if 'qwen' in model_name_or_path.lower():
        return Qwen2VLModule
    else:
        raise ValueError(f'Unsupported model: {model_name_or_path}')

def main(script_args, training_args, model_args):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print('using vlm module:', vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)
    reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    print('reward_funcs:', reward_funcs)
    import json
    from datasets import Dataset
    data_files = script_args.data_file_paths.split(':')
    image_folders = script_args.image_folders.split(':')
    if len(data_files) != len(image_folders):
        raise ValueError('Number of data files must match number of image folders')
    all_data = []
    for (data_file, image_folder) in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    if isinstance(item['image'], str):
                        item['image_path'] = [item['image']]
                        del item['image']
                    elif isinstance(item['image'], list):
                        item['image_path'] = [image for image in item['image']]
                        del item['image']
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    item['solution'] = str(solution_value)
                del item['conversations']
                all_data.append(item)
    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            return {'image_path': [p for p in example['image_path']], 'problem': example['problem'], 'solution': f"<answer> {example['solution']} </answer>", 'prompt': [{'role': 'user', 'content': [*({'type': 'image', 'text': None} for _ in range(len(example['image_path']))), {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}]}]}
        else:
            return {'problem': example['problem'], 'solution': f"<answer> {example['solution']} </answer>", 'prompt': [{'role': 'user', 'content': [{'type': 'text', 'text': question_prompt.format(Question=example['problem'])}]}]}
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    trainer_cls = VLMGRPOTrainer
    print('using trainer:', trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    trainer = trainer_cls(model=model_args.model_name_or_path, reward_funcs=reward_funcs, args=training_args, vlm_module=vlm_module_cls(), train_dataset=splits['train'], eval_dataset=splits.get('validation') if training_args.eval_strategy != 'no' else None, peft_config=get_peft_config(model_args), freeze_vision_modules=model_args.freeze_vision_modules, attn_implementation=model_args.attn_implementation, max_pixels=script_args.max_pixels, min_pixels=script_args.min_pixels)
    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
if __name__ == '__main__':
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    (script_args, training_args, model_args) = parser.parse_args_and_config()
    if training_args.deepspeed and 'zero3' in training_args.deepspeed:
        print('zero3 is used, qwen2_5vl forward monkey patch is applied')
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
