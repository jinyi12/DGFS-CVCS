import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from gflownet.molecular_agent import MolecularGFlowNetAgent
from gflownet.utils import loss2ess_info, setup_logging
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs
from gflownet.boltzmann_target import MolecularBoltzmannTarget
from gflownet.tps_dps_loss import MolecularTPSDPSTrainer

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

# New options for dynamics choice and algorithm
parser.add_argument(
    "--use_openmm_int",
    action="store_true",
    help="Use OpenMM for dynamics integration (state updates)",
)
parser.add_argument(
    "--use_tps_dps",
    action="store_true",
    help="Use TPS-DPS algorithm with Boltzmann density",
)

# Add debugging options
parser.add_argument(
    "--enable_loss_clipping",
    action="store_true",
    help="Enable loss clipping and extra debugging for high loss values",
)
parser.add_argument(
    "--debug_level",
    type=int,
    default=0,
    help="Set debugging verbosity (0-4, higher is more verbose)",
)

# Training Config
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
# parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)

# Sampling Config
parser.add_argument("--timestep", default=1, type=float)


def main():
    args = parser.parse_args()

    # Setup logging and result directories
    log_dir = os.path.join(args.save_dir, args.molecule)
    setup_logging(log_dir)

    # Print key configuration settings
    print(f"Running with molecule: {args.molecule}")
    print(f"Temperature: {args.temperature}K")
    print(f"Device: {args.device}")
    print(f"Algorithm: {'TPS-DPS' if args.use_tps_dps else 'GFlowNet SubTB'}")
    print(f"Integration: {'OpenMM' if args.use_openmm_int else 'GFlowNet diffusion'}")
    print(
        f"Debugging: {'Enabled' if args.debug_level > 0 else 'Disabled'}, Loss clipping: {'Enabled' if args.enable_loss_clipping else 'Disabled'}"
    )

    # Create positions directory for saving
    os.makedirs(os.path.join(log_dir, "positions"), exist_ok=True)

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
            # Additional parameters
            "learning_rate": args.learning_rate,
            "use_tps_dps": args.use_tps_dps,
            "use_openmm_int": args.use_openmm_int,
        }
    )

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create molecular dynamics simulator for either integration or just potential energy
    # Always create MDs for potential energy calculation
    mds = MolecularMDs(cfg, num_systems=args.batch_size)
    print(f"Created OpenMM simulator with {mds.num_particles} particles")

    # Create the base molecular task
    task = temp_task

    # Initialize agent with both task and MDs
    agent = MolecularGFlowNetAgent(cfg, task, mds)

    # Configure debugging options if enabled
    if args.enable_loss_clipping:
        print("Enabling loss clipping and trajectory checking")
        agent.gfn._check_trajectories = True
    else:
        # Disable these checks for production runs
        agent.gfn._check_trajectories = False

    # Set debug verbosity
    if args.debug_level > 0:
        print(f"Setting debug level to {args.debug_level}")

    # Setup trainer based on chosen algorithm
    if args.use_tps_dps:
        # Initialize TPS-DPS with Boltzmann target
        print("Using TPS-DPS algorithm with Boltzmann density target")
        target_measure = MolecularBoltzmannTarget(cfg, agent)
        trainer = MolecularTPSDPSTrainer(cfg, agent, target_measure)
    else:
        print("Using original GFlowNet with SubTrajectoryBalance")
        trainer = None  # We'll use agent.train directly

    # Print summary of configuration
    print(f"Training molecular GFlowNet for {args.molecule}")
    print(
        f"Using {'OpenMM integration' if args.use_openmm_int else 'GFlowNet diffusion'} for state updates"
    )
    print(f"Using OpenMM for potential energy calculations in both cases")

    # Calculate temperature schedule - linear decay from start to end
    temp_schedule = np.linspace(
        args.start_temperature, args.end_temperature, args.num_epochs
    )

    # Training loop
    for epoch in range(args.num_epochs):
        temperature = temp_schedule[epoch]
        print(f"Epoch {epoch+1}/{args.num_epochs} - Temperature: {temperature}K")

        # Set temperature for sampling
        agent.set_temperature(temperature)

        # Sample a batch of trajectories
        traj, info = agent.sample(args.batch_size)

        # Extra debugging for high sample values if enabled
        if args.debug_level >= 3:
            print(f"Sampling complete. Trajectory length: {len(traj)}")
            print(f"Max state value: {info['x_max']:.4f}")
            t0, x0, r0 = traj[0]  # First state
            print(
                f"Initial state stats - min: {x0.min().item():.4f}, max: {x0.max().item():.4f}"
            )
            print(
                f"Initial reward stats - min: {r0.min().item():.4f}, max: {r0.max().item():.4f}"
            )

        # Train on trajectories
        try:
            if args.use_tps_dps:
                # Use TPS-DPS training
                loss_info = trainer.train_on_gflownet_traj(traj)
            else:
                # Use original GFlowNet training
                loss_info = agent.train(traj)

            # Check for extremely high loss
            if "loss_train" in loss_info and (
                loss_info["loss_train"] > 1e10
                or np.isnan(loss_info["loss_train"])
                or np.isinf(loss_info["loss_train"])
            ):
                print(
                    f"WARNING: Extremely high loss detected: {loss_info['loss_train']}"
                )
                if args.enable_loss_clipping:
                    print("Loss clipping enabled, continuing training...")
                else:
                    print(
                        "Consider enabling --enable_loss_clipping to handle high losses"
                    )

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            print(traceback.format_exc())
            print("Skipping this training iteration")
            continue

        # Print loss information
        if (epoch + 1) % args.log_interval == 0:
            if args.use_tps_dps:
                print(
                    f"Epoch {epoch+1} - Loss: {loss_info['loss']:.6f}, "
                    f"Log Z: {loss_info['log_z']:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1} - Loss: {loss_info['loss_train']:.6f}, "
                    f"LogZ: {loss_info['logz_model']:.6f}"
                )

        # Evaluate model
        if (epoch + 1) % args.eval_interval == 0:
            with torch.no_grad():
                # Use end temperature for evaluation
                agent.set_temperature(args.end_temperature)
                eval_traj, eval_info = agent.sample(args.batch_size)

                # For GFlowNet we can compute ESS
                if not args.use_tps_dps:
                    ess_info = loss2ess_info(eval_info)
                    print(
                        f"Evaluation - ESS: {ess_info['ess']:.4f}, "
                        f"ESS%: {ess_info['ess_percent']:.2f}%"
                    )
                else:
                    print(f"Evaluation - Max Position: {eval_info['x_max']:.6f}")

        # Save model
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(log_dir, f"model_epoch_{epoch+1}.pt")

            if args.use_tps_dps:
                # Save both policy network and log_z
                torch.save(
                    {
                        "policy": agent.gfn.state_dict(),
                        "log_z": trainer.loss.log_z,
                    },
                    model_path,
                )
            else:
                # Just save the GFlowNet
                torch.save(agent.gfn.state_dict(), model_path)

            print(f"Saved model to {model_path}")

    # Save final model
    final_model_path = os.path.join(log_dir, "model_final.pt")
    if args.use_tps_dps:
        torch.save(
            {
                "policy": agent.gfn.state_dict(),
                "log_z": trainer.loss.log_z,
            },
            final_model_path,
        )
    else:
        torch.save(agent.gfn.state_dict(), final_model_path)

    print(f"Training complete! Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
