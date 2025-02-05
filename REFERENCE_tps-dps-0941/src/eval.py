import torch
import argparse

from utils.plot import Plot
from dynamics import dynamics
from utils.metrics import Metric
from torch.distributions import Normal

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="aldp", type=str)
parser.add_argument("--save_dir", default="paths/aldp/force", type=str)

# Sampling Config
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--timestep", default=1, type=float)

args = parser.parse_args()


class Eval:
    def __init__(self, args):
        self.get_md_info(args)
        self.log_prob = Normal(0, self.std).log_prob

        self.plot = Plot(args, self)
        self.metric = Metric(args, self)

    def get_md_info(self, args):
        md = getattr(dynamics, args.molecule.title())(args, args.end_state)
        self.target_position = torch.tensor(
            md.position, dtype=torch.float, device=args.device
        ).unsqueeze(0)
        md = getattr(dynamics, args.molecule.title())(args, args.start_state)
        self.start_position = torch.tensor(
            md.position, dtype=torch.float, device=args.device
        ).unsqueeze(0)
        self.num_particles = md.num_particles
        self.heavy_atoms = md.heavy_atoms
        self.energy_function = md.energy_function
        self.std = torch.tensor(
            md.std,
            dtype=torch.float,
            device=args.device,
        )
        self.m = torch.tensor(
            md.m,
            dtype=torch.float,
            device=args.device,
        ).unsqueeze(-1)

    def __call__(self):
        log = {}
        log.update(self.metric())
        print(
            f"RMSD: {log['rmsd']:.2f} ± {log['rmsd_std']:.2f} "
            f"THP: {log['thp']:.2f} "
            f"ETP: {log['etp']:.2f} ± {log['etp_std']:.2f}"
        )

        self.plot()


if __name__ == "__main__":
    Eval(args)()
