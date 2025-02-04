import sys, os
import gzip, pickle
from collections import defaultdict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf
import os.path as osp

import numpy as np
import torch

from gflownet.gflownet import get_alg, sample_traj
from gflownet.utils import seed_torch

torch.backends.cudnn.benchmark = True


def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
        cfg.work_directory = os.getcwd()
        cfg.gpu_type = (
            torch.cuda.get_device_name()
            if (torch.cuda.is_available() and cfg.device >= 0)
            else "CPU"
        )
        print(f"GPU type: {cfg.gpu_type}")

        cfg.task = cfg.target.dataset._target_.split(".")[-2]
        cfg.logr_min = cfg.rmin

        cfg.batch_size = cfg.bs
        cfg.weight_decay = cfg.wd
        cfg.sigma_interactive = cfg.sgmit
        if cfg.sigma_interactive <= 0:
            cfg.sigma_interactive = cfg.sigma
        cfg.t_end = cfg.dt * cfg.N
        cfg.subtb_lambda = cfg.stlam

        # Add checkpoint configuration
        cfg.checkpoint_dir = osp.join(cfg.work_directory, "checkpoints")
        cfg.checkpoint_freq = cfg.eval_freq  # Save at same frequency as eval
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    del cfg.d, cfg.bs, cfg.wd, cfg.sgmit, cfg.stlam, cfg.rmin
    return cfg


@hydra.main(config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg = refine_cfg(cfg)
    device = torch.device(
        f"cuda:{cfg.device:d}"
        if torch.cuda.is_available() and cfg.device >= 0
        else "cpu"
    )

    seed_torch(cfg.seed)
    print(f"Device: {device}, GPU type: {cfg.gpu_type}")
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    data = instantiate(cfg.target.dataset)
    true_logz = data.gt_logz()
    if true_logz is not None:
        print(f"True logZ={true_logz:.4f}")

    def logr_fn_detach(x):
        logr = -data.energy(x).detach()
        logr = torch.where(torch.isinf(logr), cfg.logr_min, logr)
        logr = torch.clamp(logr, min=cfg.logr_min)
        return logr

    gflownet = get_alg(cfg, data)
    gflownet.to(device)

    metric_best = 100.0
    best_model_path = None

    # Load checkpoint if exists
    latest_checkpoint = osp.join(cfg.checkpoint_dir, "latest.pt")
    if osp.exists(latest_checkpoint):
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        gflownet.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint["step"]
        metric_best = checkpoint["metric_best"]
        if "best_model_path" in checkpoint:
            best_model_path = checkpoint["best_model_path"]
        print(f"Resuming from step {start_step}")
    else:
        start_step = 0

    for step_idx in range(start_step, cfg.steps):
        ######### eval
        gflownet.eval()
        if step_idx % cfg.eval_freq == 0 or step_idx >= cfg.steps - 1:
            traj, eval_info = gflownet.eval_step(cfg.eval_n, logr_fn_detach)
            print(
                f"EVALUATION: step={step_idx}:",
                " ".join([f"{k}={v:.3f}" for k, v in eval_info.items()]),
            )

            if true_logz is not None:
                logz_diff = abs(eval_info["logz"] - true_logz)
                print(f"logZ diff={logz_diff:.3f}")
                if logz_diff < metric_best:
                    metric_best = logz_diff
                    print(f"best metric: {metric_best:.2f} at step {step_idx}")

                    # Save best model
                    best_model_path = osp.join(
                        cfg.checkpoint_dir, f"best_model_step_{step_idx}.pt"
                    )
                    torch.save(
                        {
                            "step": step_idx,
                            "model_state_dict": gflownet.state_dict(),
                            "metric_best": metric_best,
                            "config": OmegaConf.to_container(cfg, resolve=True),
                            "eval_info": eval_info,
                        },
                        best_model_path,
                    )

        ######### rollout
        traj_batch, _ = sample_traj(
            gflownet,
            cfg,
            logr_fn_detach,
            batch_size=cfg.batch_size,
            sigma=cfg.sigma_interactive,
        )

        ######### training
        gflownet.train()
        gflownet.train_step(traj_batch)

        # Save checkpoint periodically
        if step_idx % cfg.checkpoint_freq == 0:
            checkpoint_path = osp.join(cfg.checkpoint_dir, "latest.pt")
            torch.save(
                {
                    "step": step_idx + 1,  # Save next step to resume from
                    "model_state_dict": gflownet.state_dict(),
                    "metric_best": metric_best,
                    "best_model_path": best_model_path,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                checkpoint_path,
            )

    # Save final model
    final_model_path = osp.join(cfg.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "step": cfg.steps,
            "model_state_dict": gflownet.state_dict(),
            "metric_best": metric_best,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "eval_info": eval_info,
        },
        final_model_path,
    )


if __name__ == "__main__":
    main()
