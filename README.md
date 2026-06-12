# CCC-GRPO

Official implementation of **CCC-GRPO: Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**, accepted at ICML 2026.

**[Yao Du](https://scholar.google.com.hk/citations?user=8krbrWsAAAAJ&hl=zh-CN), [Shanshan Song](https://scholar.google.com/citations?hl=zh-CN&user=EoNWyTcAAAAJ&view_op=list_works&sortby=pubdate&inst=1381320739207392350), [Xiaomeng Li](https://xmengli.github.io/)**



[[Paper](https://arxiv.org/abs/2605.01402)] [[Dataset](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs)]

## Highlights

- **Distribution-aware reinforcement learning.** CCC-GRPO uses batch-level Concordance Correlation Coefficient supervision to align the distribution of numerical predictions with the ground-truth distribution.
- **MLLM deep imbalanced regression benchmark.** We introduce four naturally imbalanced vision-language regression tasks covering facial age, movie ratings, and skeletal age.

<p align="center">
  <img src="figures/MLLM_Numerical_Fig2.png" width="900">
</p>

Unlike token-level SFT and point-wise GRPO rewards, CCC-GRPO evaluates each response in the context of the mini-batch. This provides distribution-level supervision for long-tailed regression targets.

## CCC Reward

<p align="center">
  <img src="figures/MLLM_Numerical_Fig3.png" width="900">
</p>

For each candidate response, CCC-GRPO combines its numerical prediction with the mean predictions of the other sample groups and computes their concordance with the corresponding targets. The implementation is available for:

- [AgeDB-DIR, IMDB-WIKI-DIR, and IMDB-Movie-DIR](src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py#L155-L234)
- [BoneAge-DIR](src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py#L237-L316)

## Benchmark

<p align="center">
  <img src="figures/MLLM_Numerical_3.png" width="900">
</p>

| Dataset | Train | Test | Target |
| --- | ---: | ---: | --- |
| AgeDB-DIR | 12,208 | 2,140 | Age (years) |
| IMDB-WIKI-DIR | 81,911 | 11,016 | Age (years) |
| IMDB-Movie-DIR | 7,049 | 1,203 | IMDb score |
| BoneAge-DIR | 12,528 | 1,508 | Bone maturity (months) |

The complete benchmark is hosted on [Hugging Face](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs) as WebDataset shards with train/test annotations.

```bash
hf download ChanganYao/DeepImbalancedRegressionForMLLMs \
  --repo-type dataset \
  --local-dir data
```

## Training

Update `REPO_HOME`, `data_paths`, `image_folders`, `model_path`, and `--output_dir` in the selected script.

| Dataset | Training script | Rewards |
| --- | --- | --- |
| AgeDB-DIR | `run_grpo_age_lora_ccc.sh` | `format`, `global_rank` |
| IMDB-WIKI-DIR | `run_grpo_imdbwiki_lora_ccc.sh` | `format`, `global_rank` |
| IMDB-Movie-DIR | `run_grpo_movie_lora_ccc.sh` | `format`, `global_rank` |
| BoneAge-DIR | `run_grpo_boneage_lora_ccc.sh` | `format-boneage`, `global_rank_boneage` |

```bash
bash scripts/train/run_grpo_age_lora_ccc.sh
bash scripts/train/run_grpo_imdbwiki_lora_ccc.sh
bash scripts/train/run_grpo_movie_lora_ccc.sh
bash scripts/train/run_grpo_boneage_lora_ccc.sh
```

## Evaluation

```bash
torchrun --nproc_per_node=2 src/eval/run_eval_single_step.py \
  --step <CHECKPOINT_STEP> \
  --run_name <RUN_NAME> \
  --dataset <DATASET_NAME> \
  --data_file <TEST_JSON> \
  --output_dir ./logs
```

Set the checkpoint root in `src/eval/run_eval_single_step.py` to the directory containing `<RUN_NAME>/checkpoint-<CHECKPOINT_STEP>`.

## Acknowledgement

This project is built on [VLM-R1](https://github.com/om-ai-lab/VLM-R1). We also thank [VisualQuality-R1](https://github.com/tianhewu/visualquality-r1) for its valuable open-source work.

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
