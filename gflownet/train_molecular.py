import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from gflownet.molecular_agent import MolecularGFlowNetAgent
from gflownet.utils import loss2ess_info, setup_logging
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="aldp", type=str)

# Logger Config
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--eval_interval", default=1000, type=int)
parser.add_argument("--save_interval", default=5000, type=int)

# GFlowNet Config
parser.add_argument("--subtb_lambda", default=0.9, type=float)
parser.add_argument("--t_end", default=1.0, type=float)
parser.add_argument("--dt", default=0.01, type=float)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--xclip", default=10.0, type=float)
parser.add_argument("--nn_clip", default=100.0, type=float)
parser.add_argument("--lgv_clip", default=100.0, type=float)
parser.add_argument(
    "--task", default="molecular", type=str
)  # molecular task for this project

# Molecular Config
parser.add_argument("--start_state", default="c5", type=str)
# parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)

# Training Config
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)

# Sampling Config
parser.add_argument("--timestep", default=1, type=float)


def main():
    args = parser.parse_args()

    # First create a temporary task to get the system dimensions
    temp_cfg = OmegaConf.create(
        {
            "molecule": args.molecule,
            "start_state": args.start_state,
            "temperature": args.temperature,
            "friction": args.friction,
            "timestep": args.timestep,
        }
    )
    temp_task = MolecularTask(temp_cfg)

    # Now we can get the proper dimension for the full config
    # Multiply by 3 for x,y,z coordinates of each particle
    data_ndim = temp_task.num_particles * 3

    # Convert args to OmegaConf for compatibility with GFlowNet
    cfg = OmegaConf.create(
        {
            # System dimensions
            "data_ndim": data_ndim,
            # GFlowNet parameters
            "seed": args.seed,
            "device": args.device,
            "subtb_lambda": args.subtb_lambda,
            "t_end": args.t_end,
            "dt": args.dt,
            "sigma": args.sigma,
            "weight_decay": args.weight_decay,
            "xclip": args.xclip,
            "nn_clip": args.nn_clip,
            "lgv_clip": args.lgv_clip,
            "N": int(args.t_end / args.dt),
            "batch_size": args.batch_size,
            "task": args.task,
            # Network configuration (add these based on GFlowNet requirements)
            "f_func": {
                "_target_": "gflownet.network.FourierMLP",
                "num_layers": 4,
                "channels": 128,
                "in_shape": (data_ndim,),
                "out_shape": (data_ndim,),
            },
            "g_func": {
                "_target_": "gflownet.network.IdentityOne",
            },
            "f": "tgrad",  # Use temperature-scaled gradient for molecular systems
            "lr": args.learning_rate,
            "zlr": args.learning_rate * 0.1,  # Typically lower learning rate for flow
            # Molecular system parameters
            "molecule": args.molecule,
            "start_state": args.start_state,
            "temperature": args.temperature,
            "friction": args.friction,
            "timestep": args.timestep,
        }
    )

    # Setup directories and logging
    save_dir = os.path.join(args.save_dir, args.molecule)
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(save_dir)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"System dimension (num_particles * 3): {data_ndim}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize MDs first
    mds = MolecularMDs(cfg, cfg.batch_size)
    logger.info(f"System dimension (num_particles * 3): {mds.num_particles * 3}")

    # Initialize agent with MDs
    agent = MolecularGFlowNetAgent(cfg, mds)

    # Setup temperature annealing
    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_epochs
    )

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_ess = 0

    for epoch in range(args.num_epochs):
        current_temp = temperatures[epoch]
        logger.info(f"Epoch {epoch}, Temperature: {current_temp:.2f}")

        # Sample trajectories using MDs
        traj, sample_info = agent.sample(args.batch_size, mds, current_temp)

        # Training step
        loss_info = agent.train(traj)
        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            logger.info(
                f"Step {global_step}: "
                f"Loss: {loss_info['loss_train']:.4f}, "
                f"LogZ: {loss_info['logz_model']:.4f}"
            )

        # Evaluation
        if global_step % args.eval_interval == 0:
            agent.gfn.eval()
            with torch.no_grad():
                eval_traj, eval_info = agent.sample(args.num_samples, mds, current_temp)
                ess_info = loss2ess_info(eval_info)
                logger.info(
                    f"Eval Step {global_step}: "
                    f"ESS: {ess_info['ess']:.4f}, "
                    f"ESS%: {ess_info['ess_percent']:.2f}%"
                )

                # Save best model
                if ess_info["ess"] > best_ess:
                    best_ess = ess_info["ess"]
                    torch.save(
                        agent.gfn.state_dict(), os.path.join(save_dir, "best_model.pt")
                    )

        # Regular model saving
        if global_step % args.save_interval == 0:
            torch.save(
                agent.gfn.state_dict(),
                os.path.join(save_dir, f"model_step_{global_step}.pt"),
            )

    logger.info("Training completed!")

    # Save final model
    torch.save(agent.gfn.state_dict(), os.path.join(save_dir, "final_model.pt"))


if __name__ == "__main__":
    main()
