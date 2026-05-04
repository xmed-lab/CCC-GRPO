# CCC-GRPO

This repository contains PyTorch implementation of "Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression (ICML 2026)".


Created by [Du Yao](https://duyao-art.github.io/), [Song Shanshan](https://scholar.google.com.hk/citations?hl=zh-CN&user=3I5VuhUAAAAJ), [Li Xiaomeng](https://xmengli.github.io/)\*


## Overview of CCC-GRPO
We formulate deep imbalanced regression in MLLMs as a distribution-aware reinforcement learning problem.

<p align="center">
    <img src="figures/MLLM_Numerical_Fig3.png" width="850"> <br>
  



## Comparison of training paradigms for numerical prediction in MLLMs. 

Left: SFT treats regression as token-level classification.

Middle: Standard GRPO applies point-wise scalar rewards to each generation. 

Right: CCC-GRPO introduces batch-level, distributionaware relational supervision.

<p align="center">
    <img src="figures/MLLM_Numerical_Fig2.png" width="850"> <br>
  


## Overview of the constructed DIR benchmark for MLLMs.

We reconstruct all datasets into a unified DIR benchmark tailored for MLLMs, where models are required to generate continuous values via token-based decoding under naturally skewed training distributions. In total, the benchmark covers over 129K samples. 

![intro](figures/MLLM_Numerical_3.png)
