# CCC-GRPO

Official implementation of **CCC-GRPO: Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression**, accepted at ICML 2026.

[[Paper](https://arxiv.org/abs/2605.01402)] [[Dataset](https://huggingface.co/datasets/ChanganYao/DeepImbalancedRegressionForMLLMs)]

<p align="center">
  <img src="figures/MLLM_Numerical_Fig3.png" width="800">
</p>

CCC-GRPO optimizes multimodal large language models for deep imbalanced regression with a group-aware Concordance Correlation Coefficient (CCC) reward. For each sampled response, it evaluates the candidate together with the group-mean predictions of the other samples, directly aligning the prediction and target distributions.

The reward implementation is available for the [AgeDB-DIR, IMDB-WIKI-DIR, and IMDB-Movie-DIR target ranges](src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py#L155-L234) and the [BoneAge-DIR target range](src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py#L237-L316).

## Data

| Benchmark | Train | Test | Target |
| --- | ---: | ---: | --- |
| AgeDB-DIR | 12,208 | 2,140 | Age (years) |
| IMDB-WIKI-DIR | 81,911 | 11,016 | Age (years) |
| IMDB-Movie-DIR | 7,049 | 1,203 | IMDb score |
| BoneAge-DIR | 12,528 | 1,508 | Bone maturity (months) |

Download all four benchmarks from Hugging Face:

```bash
hf download ChanganYao/DeepImbalancedRegressionForMLLMs \
  --repo-type dataset \
  --local-dir data
```

## Training

Set `REPO_HOME`, `data_paths`, `image_folders`, `model_path`, and `--output_dir` in the selected script, then run:

```bash
bash scripts/train/run_grpo_age_lora_ccc.sh
bash scripts/train/run_grpo_imdbwiki_lora_ccc.sh
bash scripts/train/run_grpo_movie_lora_ccc.sh
bash scripts/train/run_grpo_boneage_lora_ccc.sh
```

AgeDB-DIR, IMDB-WIKI-DIR, and IMDB-Movie-DIR use `format + global_rank`; BoneAge-DIR uses `format-boneage + global_rank_boneage` for its target range.

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

This repository is developed from [VLM-R1](https://github.com/om-ai-lab/VLM-R1).

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
