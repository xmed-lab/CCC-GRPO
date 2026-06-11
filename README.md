# CCC-GRPO

Official code release for the ICML 2026 paper:

**Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**

[[arXiv]](https://arxiv.org/abs/2605.01402)

## Overview

We study **deep imbalanced regression (DIR)** in multimodal large language models. Standard MLLM training pipelines treat numerical prediction either as token classification or as point-wise scalar regression, which often leads to **regression-to-the-mean** under long-tailed continuous targets.

CCC-GRPO addresses this issue by injecting **distributional awareness** into reinforcement learning for MLLMs. Instead of rewarding each sample independently, our method introduces **batch-level relational supervision** so that the model is optimized not only for individual correctness, but also for preserving the global structure of the target distribution.

## Main Contributions

- We present the **first systematic DIR benchmark for MLLMs**, covering four naturally imbalanced regression datasets: `AgeDB-DIR`, `IMDB-WIKI-DIR`, `IMDB-Movie-DIR`, and `BoneAge-DIR`.
- We show that **point-wise supervision is insufficient** for long-tailed numerical prediction in MLLMs.
- We propose **CCC-GRPO**, a distribution-aware reinforcement learning objective that introduces **batch-level concordance structure** into reward design.
- We provide a practical MLLM training pipeline for DIR based on `VLM-R1`, together with benchmark annotations and evaluation code.

## Method Intuition

The key idea is simple:

- standard SFT or regression reward treats each prediction independently
- DIR requires preserving **relative ordering** and **global distribution shape**
- CCC-GRPO therefore uses a **batch-level reward** instead of only per-sample correctness

In other words, the model should not only predict a plausible number for one image, but also produce a set of predictions whose structure agrees with the target distribution.

## A Simple Reward Example

Below is a compact example showing the kind of distribution-aware supervision used in our codebase:

```python
def compute_ce_dis_loss(self, logits, y, d):
    list_target = list(range(d))
    target = torch.Tensor(list_target).to("cuda:0")
    target = torch.unsqueeze(target, 1)
    ls_weight = []

    for i in range(len(y)):
        label_inv_ranks = torch.abs(y[i] - target).transpose(0, 1)
        label_inv_ranks_norm = (
            torch.abs(y[i] - target).transpose(0, 1)
            / torch.sum(label_inv_ranks, dim=1)
            * (d - 1)
        )
        label_inv_ranks_norm = torch.squeeze(label_inv_ranks_norm, 0)
        label_inv_ranks_norm[y[i]] = 1.0
        ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()
        ls_weight.append(ls_label_inv_ranks_norm)

    weight = torch.Tensor(ls_weight).to("cuda:0")
    logits_weight = logits * weight
    loss = self.ce_loss_func(logits_weight, y)
    return loss
```

This snippet illustrates the central idea: predictions are not treated as isolated point targets. Instead, the loss is shaped by the **relative position of labels in the target space**, which encourages more distribution-sensitive optimization.

## Benchmark Summary

| Dataset | Train | Test | Target | Domain |
| --- | ---: | ---: | --- | --- |
| AgeDB-DIR | 12,208 | 2,140 | Age (years) | In-the-wild faces |
| IMDB-WIKI-DIR | 81,911 | 11,016 | Age (years) | Web-scale faces |
| IMDB-Movie-DIR | 7,049 | 1,203 | IMDb movie score | Movie posters |
| BoneAge-DIR | 12,528 | 1,508 | Bone maturity (months) | Medical imaging |

## Repository Structure

- `src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`
  - training entry for GRPO on regression-style `json/jsonl` data
- `src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`
  - prompt construction and reward implementation
- `src/eval/run_eval_single_step.py`
  - evaluation entry
- `weights/`
  - distribution-related weighting files used by the released benchmarks
- `data/`
  - released annotations and dataset construction metadata

## Dataset Release

The benchmark data are released at:

- `https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs`

Each subset contains:

- an `images/` folder
- the released train annotation file
- the released balanced test annotation file

## Quick Start

The simplest way to understand the release is:

1. prepare a dataset in the same `json/jsonl` format as the files in `data/`
2. use `grpo_jsonl.py` as the training entry
3. use `qwen_module.py` for the prompt and reward logic
4. use `run_eval_single_step.py` for evaluation

Minimal file-level entry points:

```bash
python src/open-r1-multimodal/src/open_r1/grpo_jsonl.py
python src/eval/run_eval_single_step.py
```

If you want to adapt the method to a new regression task, the two places to read first are:

- `src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`
- `data/`

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
