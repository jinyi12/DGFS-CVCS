# Transition Path Sampling with Diffusion Path Samplers
[![arXiv](https://img.shields.io/badge/arXiv-2405.19961-84cc16)](https://arxiv.org/abs/2405.19961)

This repository contains the code to reproduce the results of the paper ["Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers"](https://arxiv.org/abs/2405.19961) (accepted by ICLR 2025).

3D videos of transition paths from our diffusion path sampler can be found in [project page](https://kiyoung98.github.io/tps-dps/).

## Installation
We provide a script for installing packages (assuming CUDA 11.x).
```
conda env create -f environment.yml
```

## Quickstart
We provide independent code `double-well.ipynb` for training diffusion path sampler to sample transition paths of a synthetic double-well system. 

## Steps to reproduce the results
We provide instructions to reproduce the results of aldp and train a new model. You can replace aldp with fast-folding proteins: chignolin, trpcage, bba, and bbl.

- **Sampling**: Run the following command to sample transition paths.
    ```
    bash scripts/sample/aldp.sh
    ```
- **Evaluation**: Run the following command to evaluate sampled paths.
    ```
    bash scripts/eval/aldp.sh
    ```
- **Training**: Run the following command to start training. For better results, please try many seeds.
    ```
    bash scripts/train/aldp.sh
    ```
