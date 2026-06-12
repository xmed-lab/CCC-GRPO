from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature
import json
import os


IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



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


with open(AGE_BIN_WEIGHTS_PATH_movie, "r") as f:
    AGE_BIN_WEIGHTS_movie = {
        int(k): float(v) for k, v in json.load(f).items()
    }


class InvernVLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    def get_vlm_key(self):
        return "internvl"
        
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        assert "InternVL" in model_id, f"model_id must contain 'InternVL', but got {model_id}"
        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # The model class of InternVL when being mapped has been determined by its config
        model_cls = AutoModel
        # InternVL should be inputted with "trust_remote_code=True"
        model_init_kwargs["trust_remote_code"] = True
        # "use_cache" should be removed
        model_init_kwargs.pop("use_cache", None)
        # "flash_attention_2" should be modified to "use_flash_attn" in InternVL
        if "flash_attention_2" in model_init_kwargs.get("attn_implementation", ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")
        return model_cls

    def post_model_init(self, model, processing_class):
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
    
    def is_embeds_input(self):
        return True

    def get_processing_class(self):
        return AutoProcessor
    
    def get_eos_token_id(self, processing_class):
        eos_token_id = processing_class.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id
        
    def get_vision_modules_keywords(self):
        return ['vision_model']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_flags']
    
    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = []
        for example in inputs:
            template = self.conv_template.copy()
            conversation_list = example["prompt"]
            system_message = extract_system_message(conversation_list)
            if system_message is not None:
                template.system_message = system_message
            
            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)
            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
            prompts_text.append(query)
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # Process images
        full_pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_values = self._load_image(img, input_size=self.model_config.vision_config.image_size, max_num=processing_class.max_anyres_num)
            full_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)
        
        # Process prompts
        queries = []
        image_idx = 0
        for query in prompts_text:
            while "<image>" in query:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                image_idx += 1
            queries.append(query)
        assert image_idx == len(num_patches_list)
        
        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        # Only support pure-image data currently (each sample should contain the image)
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
        
        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs, None

    def _load_image(self, image: Image.Image, input_size: int=448, max_num:int=12):
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        # images = image
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "reg":
                return "{Question} Please output the final answer in <answer> </answer> tags."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the InternVL model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?</answer>"
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
        """Calculate IoU reward between predicted bounding box from InternVL model and ground truth bounding box."""
        """Adopt soft iou reward here"""
        import re
        import os
        import json
        from datetime import datetime
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
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for i, (content, sol) in enumerate(zip(contents, solution)):
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
                # MAX_RANGE = 228.0   # BoneAge 最大月数
                # abs_err = abs(pred - gt)
                # reward = max(0.0, 1.0 - abs_err / MAX_RANGE)



                # =========================this is only used for Pure MAE Reweighting Training Setting=====================
                age = int(round(gt))          # or int(gt)
                # age = max(0, min(99, age))    # safety
                age = max(1, min(228, age))    # safety

                w = AGE_BIN_WEIGHTS_boneage.get(age, 1.0)

                rel_error = abs(pred - gt)
                # base_reward = max(0.0, 1 - rel_error * 0.1)

                MAX_RANGE = 228.0   # BoneAge 最大月数
                base_reward = max(0.0, 1.0 - rel_error / MAX_RANGE)

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
    def select_reward_func(func: str, task_type: str):
        if func == "rank":
            match task_type:
                case "rec":
                    return iou_reward
                case "reg":
                    # return Qwen2VLModule.age_reward
                    return InvernVLModule.rank_reward_mse
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "reg":
                    return InvernVLModule.format_reward_age
                    # return Qwen2VLModule.format_reward_fundus
                    # return Qwen2VLModule.format_reward_age_multi
                case "rec":
                    return InvernVLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format-boneage":
            match task_type:
                case "reg":
                    return InvernVLModule.format_reward_boneage
                case "rec":
                    return InvernVLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression":
            match task_type:
                case "reg":
                    return InvernVLModule.regression_reward
                    # return Qwen2VLModule.regression_reward_fundus
                    # return Qwen2VLModule.regression_reward_age_multi
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-boneage":
            match task_type:
                case "reg":
                    return InvernVLModule.regression_reward_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-agedb":
            match task_type:
                case "reg":
                    return InvernVLModule.regression_reward_agedb
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-imdb-wiki":
            match task_type:
                case "reg":
                    return InvernVLModule.regression_reward_imdb_wiki
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "regression-movie":
            match task_type:
                case "reg":
                    return InvernVLModule.regression_reward_movie
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "visualquality":
            match task_type:
                case "reg":
                    return InvernVLModule.visualquality_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "visualquality-boneage":
            match task_type:
                case "reg":
                    return InvernVLModule.visualquality_reward_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "diversity":
            match task_type:
                case "reg":
                    return InvernVLModule.diversity_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank":
            match task_type:
                case "reg":
                    # return Qwen2VLModule.age_reward_global_ccc_memory_counter_direction
                    # return Qwen2VLModule.age_reward_global_ccc_with_memory
                    return InvernVLModule.age_reward_global_ccc
                    # return Qwen2VLModule.fundus_reward_global_ccc_with_memory
                    # return Qwen2VLModule.fundus_reward_global_ccc_interval
                    # return Qwen2VLModule.age_reward_global_ccc_spear
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_rank_boneage":
            match task_type:
                case "reg":
                    return InvernVLModule.age_reward_global_ccc_boneage
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "global_diff_rank":
            match task_type:
                case "reg":
                    return InvernVLModule.age_reward_diff_ccc
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")








def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]
    processed_list = []
    
    for item in conversation_list:
        role = item["role"]
        content = item["content"]
        
        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    return processed_list

def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]
    return None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images