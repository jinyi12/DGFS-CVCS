# CV-Free Conformation Sampling based on DGFS

This repository is a modified version of [Diffusion Generative Flow Samplers (DGFS)](https://arxiv.org/abs/2310.02679), adapted for CV-Free Conformation Sampling.

Original DGFS work by [Dinghuai Zhang](https://zdhnarsil.github.io/), 
[Ricky Tian Qi Chen](https://rtqichen.github.io//),
[Cheng-Hao Liu](https://pchliu.github.io/), 
Aaron Courville, 
[Yoshua Bengio](https://yoshuabengio.org/).

We build upon DGFS, which proposes a novel sampler for continuous space sampling from given unnormalized densities based on stochastic optimal control 🤖 formulation and the probabilistic 🎲 GFlowNet framework.

<a href="https://imgse.com/i/pPOmv7T"><img src="https://z1.ax1x.com/2023/10/03/pPOmv7T.md.png" alt="pPOmv7T.png" border="0" /></a>

## Project Structure
- `target/`: Contains the target distribution code
- `gflownet/`: Contains the modified DGFS algorithm code for CV-Free sampling

## Examples

```bash
python -m gflownet.main target=gm dt=0.05
python -m gflownet.main target=funnel
python -m gflownet.main target=wells
```

## Checkpoints and Model Saving

The training process automatically saves checkpoints in the `checkpoints/` directory within your working directory. Three types of checkpoints are maintained:

1. **Latest Checkpoint** (`latest.pt`):
   - Saved periodically every `checkpoint_freq` steps
   - Used for resuming training from the last saved state
   - Contains model state, training step, and best metrics

2. **Best Model** (`best_model_step_{step}.pt`):
   - Saved when a new best logZ difference is achieved
   - Contains model state and evaluation metrics

3. **Final Model** (`final_model.pt`):
   - Saved at the end of training
   - Contains the final model state and training results

Training will automatically resume from the latest checkpoint if it exists in the working directory.

## Dependencies

Apart from commonly used torch, torchvision, numpy, scipy, matplotlib,
we use the following packages:
```bash
pip install hydra-core omegaconf submitit hydra-submitit-launcher
pip install wandb tqdm einops seaborn ipdb
```

## Configuration

Key training parameters in `configs/main.yaml`:
- `steps`: Total training steps
- `eval_freq`: Frequency of evaluation and checkpoint saving
- `eval_n`: Number of samples for estimating logZ
- `device`: GPU device ID (use -1 for CPU)

## Acknowledgments

This work builds upon the [DGFS paper](https://arxiv.org/abs/2310.02679). Please cite their work if you use this code:

```bibtex
@article{zhang2023diffusion,
  title={Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization},
  author={Zhang, Dinghuai and Chen, Ricky TQ and Liu, Cheng-Hao and Courville, Aaron and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2310.02679},
  year={2023}
}
```
