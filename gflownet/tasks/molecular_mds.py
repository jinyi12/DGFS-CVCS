import torch
from tqdm import tqdm
from .molecular import MolecularTask
from torch.distributions import Normal


class MolecularMDs:
    """Manages multiple molecular dynamics simulations in parallel"""

    def __init__(self, cfg, num_samples):
        self.device = cfg.device
        self.num_samples = num_samples

        # Initialize first MD to get system info
        self.template_task = MolecularTask(cfg)
        self.num_particles = self.template_task.num_particles
        self.heavy_atoms = self.template_task.heavy_atoms

        # Initialize multiple MD simulations
        self.mds = self._init_mds(cfg)

        # Get initial positions
        self.start_position = torch.tensor(
            self.template_task.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        # Setup thermal noise distribution
        self.std = torch.sqrt(
            2
            * cfg.timestep
            * cfg.friction
            * cfg.temperature
            / torch.tensor(self.template_task.m, device=self.device).unsqueeze(-1)
        )
        self.log_prob = Normal(0, self.std).log_prob

    def _init_mds(self, cfg):
        """Initialize multiple independent MD simulations"""
        mds = []
        for _ in tqdm(range(self.num_samples), desc="Initializing MD simulations"):
            md = MolecularTask(cfg)
            mds.append(md)
        return mds

    def step(self, force):
        """Step all simulations forward with given forces"""
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        """Get current positions and forces from all simulations"""
        positions, forces = [], []
        for i in range(self.num_samples):
            position, force = self.mds[i].get_state()
            positions.append(position)
            forces.append(force)

        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        forces = torch.tensor(forces, dtype=torch.float, device=self.device)
        return positions, forces

    def reset(self):
        """Reset all simulations"""
        for i in range(self.num_samples):
            self.mds[i].reset()

    def set_temperature(self, temperature):
        """Update temperature for all simulations"""
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)
