# CCC-GRPO

Official code for **Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression** (ICML 2026).

[[Paper](https://arxiv.org/abs/2605.01402)] [[Dataset](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs)]

<p align="center">
  <img src="figures/MLLM_Numerical_Fig3.png" width="800">
</p>

CCC-GRPO introduces batch-level concordance correlation coefficient (CCC) rewards for multimodal deep imbalanced regression:

```python
mu_x, mu_y = predictions.mean(), targets.mean()
cov_xy = ((predictions - mu_x) * (targets - mu_y)).mean()
ccc = 2 * cov_xy / (
    predictions.var() + targets.var() + (mu_x - mu_y) ** 2
)
```

The implementation is in
`src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`.

## Benchmarks

| Dataset | Train | Test | Reward range | Epochs | Batch/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| AgeDB-DIR | 12,208 | 2,140 | 0-100 | 4 | 16 |
| IMDB-WIKI-DIR | 81,911 | 11,016 | 0-100 | 2 | 16 |
| IMDB-Movie-DIR | 7,049 | 1,203 | 0-100 | 6 | 24 |
| BoneAge-DIR | 12,528 | 1,508 | 1-228 | 5 | 24 |

Task-specific prompts are stored in the released annotations and passed to the model unchanged. BoneAge labels reach 228 months, so its reward range follows the training labels rather than the `216 months` text in the original prompt.

## Run

```bash
git clone https://github.com/xmed-lab/CCC-GRPO.git
cd CCC-GRPO
bash setup.sh

python scripts/prepare_hf_data.py

bash scripts/train_ccc_grpo.sh agedb
bash scripts/train_ccc_grpo.sh imdb_wiki
bash scripts/train_ccc_grpo.sh imdb_movie
bash scripts/train_ccc_grpo.sh boneage
```

Set `MODEL_PATH`, `DATA_ROOT`, `OUTPUT_ROOT`, `NPROC_PER_NODE`, or `PER_DEVICE_BATCH_SIZE` to override the defaults.

Evaluation:

```bash
bash scripts/eval_ccc_grpo.sh agedb checkpoints/rl/ccc-grpo-agedb/checkpoint-100 100
```

The same command supports `imdb_wiki`, `imdb_movie`, and `boneage`.

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
