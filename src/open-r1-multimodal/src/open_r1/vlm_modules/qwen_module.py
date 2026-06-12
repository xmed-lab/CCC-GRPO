from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import numpy as np

class Qwen2VLModule(VLMBaseModule):

    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return 'qwen'

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if 'Qwen2-VL' in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif 'Qwen2.5-VL' in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f'Unsupported model: {model_id}')
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
        prompts_text = [maybe_apply_chat_template(example, processing_class)['prompt'] for example in inputs]
        return prompts_text

    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False):
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(text=prompts_text, images=images, return_tensors=return_tensors, padding=padding, padding_side=padding_side, add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(text=prompts_text, return_tensors=return_tensors, padding=padding, padding_side=padding_side, add_special_tokens=add_special_tokens)
        return (prompt_inputs, additional_output)

    @staticmethod
    def get_question_template(task_type: str):
        if task_type != 'reg':
            raise ValueError(f'Unsupported task type: {task_type}')
        return '{Question}'

    @staticmethod
    def format_reward_age(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        MIN_AGE = 0.0
        MAX_AGE = 100.0
        pattern = '<answer>\\s*(-?\\d+(?:\\.\\d+)?)\\s*</answer>'
        completion_contents = [completion[0]['content'] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime('%d-%H-%M-%S-%f')
        debug_mode = os.getenv('DEBUG_MODE') == 'true'
        log_path = os.getenv('LOG_PATH')
        if debug_mode:
            f = open(log_path.replace('.txt', '_format.txt'), 'a', encoding='utf-8')
            f.write(f'\n------------- {current_time} Format reward -------------\n')
        for content in completion_contents:
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write('→ INVALID: format error\n')
                continue
            try:
                value = float(match.group(1))
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write('→ INVALID: parse error\n')
                continue
            if MIN_AGE <= value <= MAX_AGE:
                rewards.append(0.5)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write(f'→ VALID: value={value}\n')
            else:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write(f'→ INVALID: value={value} out of range\n')
        if debug_mode:
            f.close()
        return rewards

    @staticmethod
    def format_reward_boneage(completions, **kwargs):
        import re
        import os
        from datetime import datetime
        MIN_AGE = 1.0
        MAX_AGE = 228.0
        pattern = '<answer>\\s*(-?\\d+(?:\\.\\d+)?)\\s*</answer>'
        completion_contents = [completion[0]['content'] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime('%d-%H-%M-%S-%f')
        debug_mode = os.getenv('DEBUG_MODE') == 'true'
        log_path = os.getenv('LOG_PATH')
        if debug_mode:
            f = open(log_path.replace('.txt', '_format.txt'), 'a', encoding='utf-8')
            f.write(f'\n------------- {current_time} Format reward -------------\n')
        for content in completion_contents:
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write('→ INVALID: format error\n')
                continue
            try:
                value = float(match.group(1))
            except Exception:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write('→ INVALID: parse error\n')
                continue
            if MIN_AGE <= value <= MAX_AGE:
                rewards.append(0.5)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write(f'→ VALID: value={value}\n')
            else:
                rewards.append(0.0)
                if debug_mode:
                    f.write(f'Content: {content}\n')
                    f.write(f'→ INVALID: value={value} out of range\n')
        if debug_mode:
            f.close()
        return rewards

    @staticmethod
    def age_reward_global_ccc(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime
        from scipy.stats import spearmanr
        device = kwargs.get('device')
        n_gen = kwargs.get('num_generations', 4)
        DEBUG = os.getenv('DEBUG_MODE') == 'true'
        log_path_rank = os.getenv('LOG_PATH', './debug_log.txt').replace('.txt', '_ccc.txt')

        def extract_first_number(model_answer):
            match = re.search('-?\\d+(\\.\\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.uniform(1, 100)
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search('<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)
        gt_list = [sol[0] for sol in reshaped_solution]
        contents = [completion[0]['content'] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]
        (batch_pred, batch_mean, batch_var) = ([], [], [])
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall('<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)
            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])
        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])
        rewards = []
        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
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
                    ccc = 2 * cov_xy / denom
                if np.isnan(ccc):
                    ccc = 0.0
                reward = ccc
                rewards.append(float(reward))
                if DEBUG:
                    with open(log_path_rank, 'a', encoding='utf-8') as f:
                        f.write('\n================ CCC Pair =================\n')
                        f.write(f'Sample i={i}, Gen j={j}\n')
                        f.write(f'GT_i = {gt_i}\n')
                        f.write(f'pred_i_j = {pred_i_j}\n')
                        f.write(f'Pred list = {pred_list}\n')
                        f.write(f'GT list   = {gt_list_cmp}\n')
                        f.write(f'mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n')
                        f.write(f'var_x={var_x:.3f}, var_y={var_y:.3f}\n')
                        f.write(f'cov_xy={cov_xy:.3f}\n')
                        f.write(f'CCC = {ccc:.4f}, Reward = {reward:.4f}\n')
                        f.write(f'final_reward = {reward:.4f}\n')
                        f.write('===========================================\n')
        return rewards

    @staticmethod
    def age_reward_global_ccc_boneage(completions, solution, **kwargs):
        import re
        import torch
        import os
        import random
        import numpy as np
        from datetime import datetime
        from scipy.stats import spearmanr
        device = kwargs.get('device')
        n_gen = kwargs.get('num_generations', 4)
        DEBUG = os.getenv('DEBUG_MODE') == 'true'
        log_path_rank = os.getenv('LOG_PATH', './debug_log.txt').replace('.txt', '_ccc.txt')

        def extract_first_number(model_answer):
            match = re.search('-?\\d+(\\.\\d+)?', model_answer)
            if match:
                return float(match.group())
            else:
                return random.uniform(1, 228)
        reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
        for i in range(len(reshaped_solution)):
            for j in range(len(reshaped_solution[i])):
                sol_match = re.search('<answer>(.*?)</answer>', reshaped_solution[i][j])
                g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
                reshaped_solution[i][j] = float(g)
        gt_list = [sol[0] for sol in reshaped_solution]
        contents = [completion[0]['content'] for completion in completions]
        reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]
        (batch_pred, batch_mean, batch_var) = ([], [], [])
        for i in range(len(reshaped_content)):
            cur_pred_list = []
            for j in range(len(reshaped_content[i])):
                content_matches = re.findall('<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
                cur_pred_list.append(pred)
            batch_pred.append(cur_pred_list)
            t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
            batch_mean.append([t.mean()])
            batch_var.append([t.var()])
        batch_size = len(batch_pred)
        n_gen = len(batch_pred[0])
        rewards = []
        for i in range(batch_size):
            for j in range(n_gen):
                pred_i_j = batch_pred[i][j]
                gt_i = gt_list[i]
                pred_list = [pred_i_j] + [batch_mean[z][0].item() for z in range(batch_size) if z != i]
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
                    ccc = 2 * cov_xy / denom
                if np.isnan(ccc):
                    ccc = 0.0
                reward = ccc
                rewards.append(float(reward))
                if DEBUG:
                    with open(log_path_rank, 'a', encoding='utf-8') as f:
                        f.write('\n================ CCC Pair =================\n')
                        f.write(f'Sample i={i}, Gen j={j}\n')
                        f.write(f'GT_i = {gt_i}\n')
                        f.write(f'pred_i_j = {pred_i_j}\n')
                        f.write(f'Pred list = {pred_list}\n')
                        f.write(f'GT list   = {gt_list_cmp}\n')
                        f.write(f'mu_x={mu_x:.3f}, mu_y={mu_y:.3f}\n')
                        f.write(f'var_x={var_x:.3f}, var_y={var_y:.3f}\n')
                        f.write(f'cov_xy={cov_xy:.3f}\n')
                        f.write(f'CCC = {ccc:.4f}, Reward = {reward:.4f}\n')
                        f.write(f'final_reward = {reward:.4f}\n')
                        f.write('===========================================\n')
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if task_type != 'reg':
            raise ValueError(f'Unsupported task type: {task_type}')
        reward_funcs = {'format': Qwen2VLModule.format_reward_age, 'format-boneage': Qwen2VLModule.format_reward_boneage, 'global_rank': Qwen2VLModule.age_reward_global_ccc, 'global_rank_boneage': Qwen2VLModule.age_reward_global_ccc_boneage}
        if func not in reward_funcs:
            raise ValueError(f'Unsupported reward function: {func}')
        return reward_funcs[func]
