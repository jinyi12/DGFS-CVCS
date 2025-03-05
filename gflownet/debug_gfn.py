import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
import sys
import logging

from gflownet.molecular_agent import MolecularGFlowNetAgent
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("debug_gfn")


def parse_args():
    parser = argparse.ArgumentParser(description="Debug GFlowNet Diffusion Sampler")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    parser.add_argument("--molecule", default="aldp", type=str)
    parser.add_argument("--temperature", default=300, type=float)
    parser.add_argument("--dt", default=0.01, type=float, help="Timestep for diffusion")
    parser.add_argument(
        "--t_end", default=1.0, type=float, help="End time for diffusion"
    )
    parser.add_argument(
        "--sigma", default=0.1, type=float, help="Noise level for diffusion"
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Smaller batch for debugging"
    )
    parser.add_argument(
        "--num_epochs", default=3, type=int, help="Number of training steps"
    )
    parser.add_argument("--subtb_lambda", default=0.9, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--debug_level", default=3, type=int, help="1-5, higher means more verbose"
    )
    parser.add_argument("--clip_loss", action="store_true", help="Enable loss clipping")
    return parser.parse_args()


def debug_coef_matrix(coef_matrix, lambda_val, N):
    """Debug the SubTB coefficient matrix"""
    logger.info(f"Lambda value: {lambda_val}")
    logger.info(f"N (trajectory length): {N}")
    logger.info(f"Coef matrix shape: {coef_matrix.shape}")

    # Check for NaN/Inf in coef matrix
    if torch.isnan(coef_matrix).any() or torch.isinf(coef_matrix).any():
        logger.error("NaN or Inf values found in coefficient matrix!")

    # Check matrix values
    logger.info(
        f"Coef matrix min: {coef_matrix.min().item()}, max: {coef_matrix.max().item()}"
    )

    # Print part of the matrix for inspection
    logger.info("First 5x5 submatrix:")
    logger.info(coef_matrix[:5, :5])


def debug_trajectories(traj):
    """Debug trajectory data"""
    t0, x0, r0 = traj[0]  # First state
    t_end, x_end, r_end = traj[-1]  # Last state

    # Check trajectory length
    logger.info(f"Trajectory length: {len(traj)}")

    # Check for NaN/Inf in states and rewards
    for i, (t, x, r) in enumerate(traj):
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error(f"NaN/Inf in state at timestep {i}, time {t.item()}")

        if torch.isnan(r).any() or torch.isinf(r).any():
            logger.error(f"NaN/Inf in reward at timestep {i}, time {t.item()}")

    # Log state and reward statistics
    logger.info(f"Initial time: {t0.item()}, Final time: {t_end.item()}")
    logger.info(f"Initial state min: {x0.min().item()}, max: {x0.max().item()}")
    logger.info(f"Final state min: {x_end.min().item()}, max: {x_end.max().item()}")
    logger.info(f"Initial reward min: {r0.min().item()}, max: {r0.max().item()}")
    logger.info(f"Final reward min: {r_end.min().item()}, max: {r_end.max().item()}")

    # Track state changes
    total_delta = (x_end - x0).abs().mean().item()
    logger.info(f"Average state change (start to end): {total_delta:.6f}")


def debug_loss_computation(flows, log_pf, log_pb, data_ndim, coef):
    """Debug the loss computation step by step"""

    # Check inputs for NaN/Inf
    for name, tensor in [("flows", flows), ("log_pf", log_pf), ("log_pb", log_pb)]:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.error(f"NaN/Inf in {name}!")
            logger.info(
                f"{name} min: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all NaN'}"
            )
            logger.info(
                f"{name} max: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all NaN'}"
            )

    # Calculate diff_logp
    diff_logp = log_pf - log_pb
    logger.info(
        f"diff_logp min: {diff_logp.min().item()}, max: {diff_logp.max().item()}"
    )

    # Calculate diff_logp_padded
    batch_size = flows.shape[0]
    diff_logp_padded = torch.cat(
        (torch.zeros(batch_size, 1, device=diff_logp.device), diff_logp.cumsum(dim=-1)),
        dim=1,
    )
    logger.info(
        f"diff_logp_padded min: {diff_logp_padded.min().item()}, max: {diff_logp_padded.max().item()}"
    )

    # Calculate A1
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    logger.info(f"A1 min: {A1.min().item()}, max: {A1.max().item()}")

    # Calculate A2 = flows[:, :, None] - flows[:, None, :] + A1
    A2 = flows[:, :, None] - flows[:, None, :] + A1
    logger.info(
        f"A2 (before normalization) min: {A2.min().item()}, max: {A2.max().item()}"
    )

    # Normalize and square
    A2_norm = (A2 / data_ndim).pow(2).mean(dim=0)
    logger.info(
        f"A2 (after normalization) min: {A2_norm.min().item()}, max: {A2_norm.max().item()}"
    )

    # Calculate final loss
    loss_matrix = A2_norm * coef
    logger.info(
        f"loss_matrix min: {loss_matrix.min().item()}, max: {loss_matrix.max().item()}"
    )

    # Get upper triangular part and sum
    loss = torch.triu(loss_matrix, diagonal=1).sum()
    logger.info(f"Final loss value: {loss.item()}")

    # Calculate contribution from individual terms (for top contributors)
    loss_entries = torch.triu(loss_matrix, diagonal=1)
    if loss_entries.numel() > 0:
        top_values, _ = torch.topk(loss_entries.flatten(), min(5, loss_entries.numel()))
        logger.info(f"Top 5 loss contributions: {top_values}")

    return loss


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info(f"Debug level: {args.debug_level}")
    logger.info(f"Starting debug session with molecule: {args.molecule}")
    logger.info(f"Using device: {args.device}")

    # Create temporary config for MolecularTask
    temp_cfg = OmegaConf.create(
        {
            "molecule": args.molecule,
            "start_state": "c5",  # Default
            "temperature": args.temperature,
            "friction": 0.001,
            "timestep": 1.0,
        }
    )

    # Create temporary task to get dimensions
    temp_task = MolecularTask(temp_cfg)
    data_ndim = temp_task.num_particles * 3

    logger.info(
        f"Molecule has {temp_task.num_particles} particles, dimension: {data_ndim}"
    )

    # Create full config
    cfg = OmegaConf.create(
        {
            "data_ndim": data_ndim,
            "seed": args.seed,
            "device": args.device,
            "subtb_lambda": args.subtb_lambda,
            "t_end": args.t_end,
            "dt": args.dt,
            "sigma": args.sigma,
            "weight_decay": 1e-4,
            "xclip": 10.0,
            "nn_clip": 100.0,
            "lgv_clip": 100.0,
            "N": int(args.t_end / args.dt),
            "batch_size": args.batch_size,
            "task": "molecular",
            # Network configuration
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
            "f": "tgrad",
            "lr": args.learning_rate,
            "zlr": args.learning_rate * 0.1,
            # Molecular parameters
            "molecule": args.molecule,
            "start_state": "c5",
            "temperature": args.temperature,
            "friction": 0.001,
            "timestep": 1.0,
            "learning_rate": args.learning_rate,
        }
    )

    # Create MDs for potential energy calculation
    mds = MolecularMDs(cfg, num_systems=args.batch_size)
    logger.info(f"Created MolecularMDs with {mds.num_particles} particles")

    # Create MolecularGFlowNetAgent
    agent = MolecularGFlowNetAgent(cfg, temp_task, mds)
    logger.info("Created MolecularGFlowNetAgent")

    # Monkey patch the train_loss method to include debugging if clip_loss is True
    original_train_loss = agent.gfn.train_loss

    def debug_train_loss(traj):
        logits_dict = agent.gfn.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = (
            logits_dict["flows"],
            logits_dict["log_pf"],
            logits_dict["log_pb"],
        )

        # Log tensor statistics
        logger.info("--- Debug tensor statistics ---")
        logger.info(
            f"flows shape: {flows.shape}, min: {flows.min().item()}, max: {flows.max().item()}"
        )
        logger.info(
            f"log_pf shape: {log_pf.shape}, min: {log_pf.min().item()}, max: {log_pf.max().item()}"
        )
        logger.info(
            f"log_pb shape: {log_pb.shape}, min: {log_pb.min().item()}, max: {log_pb.max().item()}"
        )

        # Normal loss calculation but with debugging
        loss = debug_loss_computation(
            flows, log_pf, log_pb, agent.gfn.data_ndim, agent.gfn.coef
        )

        # Clip loss if enabled
        if args.clip_loss and (torch.isnan(loss) or torch.isinf(loss) or loss > 1e10):
            logger.warning(f"Clipping extreme loss value: {loss.item()}")
            loss = torch.clamp(loss, max=1e4)

        # Calculate model logZ
        logZ_model = agent.gfn.flow(
            traj[0][0][None, None].to(agent.device), traj[0][1][:1, :].to(agent.device)
        ).detach()

        info = {
            "loss_dlogp": loss.item(),
            "logz_model": logZ_model.mean().item(),
            "max_flow": flows.max().item(),
            "min_flow": flows.min().item(),
        }

        return loss, info

    # Only patch if debug level is high enough
    if args.debug_level >= 3:
        agent.gfn.train_loss = debug_train_loss

    # Debug coefficient matrix
    if args.debug_level >= 2:
        debug_coef_matrix(agent.gfn.coef, args.subtb_lambda, int(args.t_end / args.dt))

    # Training loop
    logger.info("Starting debug training loop")
    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.num_epochs} ===")

        # Sample trajectories
        try:
            logger.info("Sampling trajectories...")
            traj, info = agent.sample(args.batch_size)

            # Debug trajectory data
            if args.debug_level >= 2:
                debug_trajectories(traj)

            # Debug sample info
            logger.info(f"Sampling info: {info}")

            # Train on trajectories
            logger.info("Training on trajectories...")
            loss_info = agent.train(traj)

            logger.info(f"Training loss: {loss_info['loss_train']}")
            logger.info(f"Training info: {loss_info}")

        except Exception as e:
            logger.error(f"Exception during epoch {epoch+1}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    logger.info("Debug session complete")


if __name__ == "__main__":
    main()
