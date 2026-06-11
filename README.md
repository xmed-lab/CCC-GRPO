# CCC-GRPO

This repository contains the public code release for the ICML 2026 paper:

**Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**

The implementation is built on top of `VLM-R1`. Our main contribution is the adaptation from generic VLM-R1 training to deep imbalanced regression in MLLMs through:

- task-specific regression prompts
- batch-level `CCC`-style reward design
- benchmark construction for four MLLM DIR datasets

## Included Scope

This public release only includes the paper datasets:

- `AgeDB-DIR`
- `IMDB-WIKI-DIR`
- `IMDB-Movie-DIR`
- `BoneAge-DIR`

It does **not** include unrelated historical experiments for gaze, fundus, BIWI, MPII, EchoNet, InternVL, or large hyperparameter sweep scripts.

## Repository Structure

- `src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`
  - main GRPO training entry for json/jsonl regression data
- `src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`
  - prompt and reward implementation used for the regression tasks
- `src/eval/run_eval_single_step.py`
  - evaluation entry
- `weights/`
  - target-bin weighting files used in the experiments
- `data/`
  - final train/test annotations and dataset preparation metadata for the four paper datasets

## Dataset Release

The image assets and final public annotations are released on Hugging Face:

- `ChanganYao/DeepImbalancedRegressionForMLLMs`

The annotation files in this repository use relative paths such as `images/...`. They are intended to match each dataset subset in the Hugging Face release.

## Benchmark Summary

| Dataset | Train | Test | Target |
| --- | ---: | ---: | --- |
| AgeDB-DIR | 12,208 | 2,140 | Age (years) |
| IMDB-WIKI-DIR | 81,911 | 11,016 | Age (years) |
| IMDB-Movie-DIR | 7,049 | 1,203 | IMDb movie score |
| BoneAge-DIR | 12,528 | 1,508 | Bone maturity (months) |

## Figures

<p align="center">
  <img src="figures/MLLM_Numerical_Fig3.png" width="850">
</p>

<p align="center">
  <img src="figures/MLLM_Numerical_Fig2.png" width="850">
</p>

![intro](figures/MLLM_Numerical_3.png)

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
