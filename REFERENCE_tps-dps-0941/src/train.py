import os
import torch
import argparse
from dynamics.mds import MDs
from utils.logging import Log
from dps import DiffusionPathSampler

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="aldp", type=str)

# Logger Config
parser.add_argument("--save_dir", default="results", type=str)

# Policy Config
parser.add_argument("--bias", default="force", type=str)

# Sampling Config
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1, type=float)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--num_samples", default=16, type=int)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)

# Training Config
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)
parser.add_argument("--num_rollouts", default=1000, type=int)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--log_z_lr", default=1e-3, type=float)
parser.add_argument(
    "--policy_lr",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--buffer_size",
    default=1000,
    type=int,
)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--trains_per_rollout",
    default=1000,
    type=int,
)

args = parser.parse_args()

if __name__ == "__main__":
    args.training = True
    args.save_dir = f"{args.save_dir}/{args.molecule}/{args.bias}"
    for name in ["policies", "positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    torch.manual_seed(args.seed)

    mds = MDs(args)
    log = Log(args, mds)
    agent = DiffusionPathSampler(args, mds)

    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )

    log.info("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")

        agent.sample(args, mds, temperatures[rollout])
        log.sample(rollout, agent.policy)
        loss = agent.train(args, mds)
        log.logger.info(f"loss: {loss}")
    log.info("End training")
