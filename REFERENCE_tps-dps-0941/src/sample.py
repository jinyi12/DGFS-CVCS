import os
import torch
import argparse
from dynamics.mds import MDs
from dps import DiffusionPathSampler

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--model_path", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--save_dir", default="paths", type=str)
parser.add_argument("--molecule", default="aldp", type=str)

# Policy Config
parser.add_argument("--bias", default="force", type=str)

# Sampling Config
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1, type=float)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)


args = parser.parse_args()

if __name__ == "__main__":
    args.training = False
    args.save_dir = f"{args.save_dir}/{args.molecule}/{args.bias}"
    for name in ["positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    mds = MDs(args)
    agent = DiffusionPathSampler(args, mds)
    agent.policy.load_state_dict(torch.load(args.model_path))
    agent.sample(args, mds, args.temperature)
