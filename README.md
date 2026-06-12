# CCC-GRPO

Official code for the ICML 2026 paper **Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**.

[[Paper](https://arxiv.org/abs/2605.01402)] [[Dataset](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs)]

<p align="center">
  <img src="figures/MLLM_Numerical_Fig3.png" width="800">
</p>

## Code

The uploaded training and evaluation files are copied directly from the original experiment directory:

```text
scripts/train/run_grpo_age_lora_ccc.sh
scripts/train/run_grpo_imdbwiki_lora_ccc.sh
scripts/train/run_grpo_movie_lora_ccc.sh
scripts/train/run_grpo_boneage_lora_ccc.sh
src/eval/run_eval_single_step.py
```

The CCC reward used in the experiments is `age_reward_global_ccc` in:

```text
src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py
```

The original scripts retain the experiment paths and hyperparameters. Before running, replace `REPO_HOME`, `data_paths`, `image_folders`, `model_path`, and `output_dir` with paths on your machine.

## Data

| Dataset | Train | Test |
| --- | ---: | ---: |
| AgeDB-DIR | 12,208 | 2,140 |
| IMDB-WIKI-DIR | 81,911 | 11,016 |
| IMDB-Movie-DIR | 7,049 | 1,203 |
| BoneAge-DIR | 12,528 | 1,508 |

Download the processed images and annotations from [Hugging Face](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs).

## Installation

```bash
bash setup.sh
```

## Citation

```bibtex
@misc{du2026injectingdistributionalawarenessmllms,
  title={Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression},
  author={Yao Du and Shanshan Song and Xiaomeng Li},
  year={2026},
  eprint={2605.01402},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
