# CCC-GRPO

Official implementation of the ICML 2026 paper:

[**Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**](https://arxiv.org/abs/2605.01402)

[[Paper]](https://arxiv.org/abs/2605.01402) | [[Dataset]](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs)

<p align="left">
  <img src="figures/MLLM_Numerical_Fig3.png" width="800">
</p>


## Highlights

- **First DIR benchmark for MLLMs**  
  We introduce a deep imbalanced regression benchmark for multimodal large language models across four datasets.

- **Distribution-aware reinforcement learning**  
  CCC-GRPO optimizes a batch-level CCC reward to improve numerical prediction under long-tailed target distributions.

- **No additional reward model required**  
  The CCC reward is simple, lightweight, bounded, and can be directly integrated into GRPO-style optimization.

- **Better long-tailed numerical prediction**  
  CCC-GRPO improves regression performance by encouraging both point-wise accuracy and global distributional alignment.

<p align="left">
  <img src="figures/MLLM_Numerical_Fig2.png" width="800">
</p>





## CCC Reward
The simplest CCC reward implementation can be found in:

```text
src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py
```

Core implementation:


```python
def age_reward_global_ccc(completions, solution, **kwargs):
    device = kwargs.get("device")
    n_gen = kwargs.get("num_generations", 4)

    reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
    for i in range(len(reshaped_solution)):
        for j in range(len(reshaped_solution[i])):
            sol_match = re.search(r"<answer>(.*?)</answer>", reshaped_solution[i][j])
            g = sol_match.group(1).strip() if sol_match else reshaped_solution[i][j].strip()
            reshaped_solution[i][j] = float(g)
    gt_list = [sol[0] for sol in reshaped_solution]

    contents = [completion[0]["content"] for completion in completions]
    reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

    batch_pred, batch_mean = [], []
    for i in range(len(reshaped_content)):
        cur_pred_list = []
        for j in range(len(reshaped_content[i])):
            content_matches = re.findall(r"<answer>(.*?)</answer>", reshaped_content[i][j], re.DOTALL)
            student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
            pred = extract_first_number(student_answer)
            cur_pred_list.append(pred)

        batch_pred.append(cur_pred_list)
        t = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
        batch_mean.append([t.mean()])

    rewards = []
    batch_size = len(batch_pred)
    n_gen = len(batch_pred[0])

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
            ccc = 0.0 if denom == 0 else (2 * cov_xy) / denom
            ccc = 0.0 if np.isnan(ccc) else ccc
            rewards.append(float(ccc))

    return rewards
```

## Benchmark
<p align="left">
  <img src="figures/MLLM_Numerical_3.png" width="800">
</p>

We evaluate MLLMs on four deep imbalanced regression datasets:

| Dataset | Train | Test | Target |
| --- | ---: | ---: | --- |
| AgeDB-DIR | 12,208 | 2,140 | Age (years) |
| IMDB-WIKI-DIR | 81,911 | 11,016 | Age (years) |
| IMDB-Movie-DIR | 7,049 | 1,203 | IMDb movie score |
| BoneAge-DIR | 12,528 | 1,508 | Bone maturity (months) |

The processed datasets are available at: [DIR_MLLM](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs⁠)

Scripts:

- `bash scripts/train_ccc_grpo.sh agedb`
- `bash scripts/train_ccc_grpo.sh imdb_movie`
- `bash scripts/train_ccc_grpo.sh boneage`
- `bash scripts/train_ccc_grpo.sh imdb_wiki`
- `bash scripts/eval_ccc_grpo.sh agedb /path/to/checkpoint-100 100`

---

## Installation

This repository is built upon the VLM-R1 multimodal training framework. Please follow their instructions for dependencies installation.

---

## Training

CCC-GRPO can be trained by specifying the proposed CCC reward function during GRPO optimization.


A typical training command is:

```bash
bash scripts/train_ccc_grpo.sh agedb
```

---

## Evaluation

After training, the model can be evaluated on the DIR benchmark datasets with:

```bash
bash scripts/eval_ccc_grpo.sh agedb /path/to/checkpoint-100 100
```





## Citation

```bibtex
@misc{du2026injectingdistributionalawarenessmllms,
  title={Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression},
  author={Yao Du and Shanshan Song and Xiaomeng Li},
  year={2026},
  eprint={2605.01402},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.01402},
}
```
