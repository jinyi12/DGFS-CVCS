import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from gflownet.gflownet import get_alg
from gflownet.main import refine_cfg


@hydra.main(config_path="gflownet/configs", config_name="main")
def visualize_results(cfg: DictConfig) -> None:
    # Override config with GM target and dt=0.05
    # cfg.target = "gm.yaml"
    cfg.dt = 0.05

    # Refine config as done in training
    cfg = refine_cfg(cfg)

    # Setup device
    device = torch.device(
        f"cuda:{cfg.device:d}"
        if torch.cuda.is_available() and cfg.device >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize target distribution and GFlowNet
    data = instantiate(cfg.target.dataset)
    gflownet = get_alg(cfg, data)
    gflownet.to(device)

    def logr_fn_detach(x):
        logr = -data.energy(x).detach()
        logr = torch.where(torch.isinf(logr), cfg.logr_min, logr)
        logr = torch.clamp(logr, min=cfg.logr_min)
        return logr

    # Load the best model if it exists
    # checkpoint_dir = os.path.join(cfg.work_directory, "checkpoints")
    checkpoint_dir = (
        "/hpc/dctrl/jy384/DGFS-CVCS/outputs/2025-02-04/14-55-14/checkpoints"
    )
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("final_model")]

    if model_files:
        # Load the latest best model
        latest_best = sorted(model_files)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_best)
        print(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        gflownet.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]
        print(f"Model from step {step} loaded successfully")
    else:
        print("No best model checkpoint found")
        return

    # Generate samples for visualization
    print("Generating samples for visualization...")
    gflownet.eval()
    with torch.no_grad():
        # Generate 15000 samples for visualization
        traj, eval_info = gflownet.eval_step(
            15000,
            logr_fn_detach,
        )
        print("Eval info:", eval_info)
        samples = traj[-1][1]  # Get final states from trajectories

        # Create visualization directory
        viz_dir = os.path.join(cfg.work_directory, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Generate visualizations
        print("Creating visualizations...")
        img_dict = gflownet.visualize(traj, data.energy, step=step)

        print("Visualization files created:")
        for viz_type, filepath in img_dict.items():
            full_path = os.path.join(viz_dir, filepath)
            # Move the file if it's not already in the visualization directory
            if os.path.dirname(filepath) != viz_dir:
                os.rename(filepath, full_path)
            print(f"- {viz_type}: {full_path}")


if __name__ == "__main__":
    visualize_results()
