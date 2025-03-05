import torch
import numpy as np
from torch import nn


class TPSDPSLoss(nn.Module):
    """
    Implementation of the TPS-DPS algorithm loss,
    as described in the reference code.

    Loss is formulated as: loss = (log_z + log_bpm - log_tm).square().mean()
    """

    def __init__(self, cfg):
        """
        Initialize the TPS-DPS loss.

        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        # Initialize log_z parameter to be learned
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def forward(self, log_bpm, log_tm):
        """
        Compute the TPS-DPS loss.

        Args:
            log_bpm: Log biased path measure, shape [batch_size]
            log_tm: Log target measure, shape [batch_size]

        Returns:
            torch.Tensor: Scalar loss
        """
        # Implement the core TPS-DPS loss function
        loss = (self.log_z + log_bpm - log_tm).square().mean()
        return loss


class MolecularTPSDPSTrainer:
    """
    Trainer class for molecular systems using TPS-DPS algorithm.
    Integrates with MolecularGFlowNetAgent and BoltzmannTargetMeasure.
    """

    def __init__(self, cfg, agent, target_measure):
        """
        Initialize the TPS-DPS trainer.

        Args:
            cfg: Configuration object
            agent: MolecularGFlowNetAgent instance
            target_measure: BoltzmannTargetMeasure instance
        """
        self.cfg = cfg
        self.agent = agent
        self.target_measure = target_measure
        self.device = cfg.device

        # Initialize loss
        self.loss = TPSDPSLoss(cfg).to(self.device)

        # Create optimizer
        # Combine policy network and log_z parameters
        policy_params = list(agent.gfn.f_func.parameters())
        policy_params.append(self.loss.log_z)

        self.optimizer = torch.optim.Adam(
            [
                {"params": policy_params, "lr": cfg.learning_rate},
            ]
        )

    def compute_log_bpm(self, positions, forces):
        """
        Compute the log biased path measure from trajectory data.

        Args:
            positions: Positions tensor [batch_size, timesteps, num_particles, 3]
            forces: Forces tensor [batch_size, timesteps, num_particles, 3]

        Returns:
            torch.Tensor: Log biased path measure [batch_size]
        """
        batch_size = positions.shape[0]
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.cfg.timestep

        # Get policy outputs for bias force
        policy_outputs = []
        pos_reshaped = positions[:, :-1].reshape(-1, positions.shape[2], 3)
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), positions.shape[1] - 1
        )

        # Get bias forces for each position
        biases = []
        for i in range(positions.shape[1] - 1):
            pos_slice = positions[:, i]
            pos_flat = pos_slice.reshape(batch_size, -1)
            timestep = torch.full(
                (batch_size, 1), i * self.cfg.timestep, device=self.device
            )
            bias = self.agent.gfn.f(timestep, pos_flat).detach()
            bias = bias.reshape(batch_size, positions.shape[2], 3)
            biases.append(bias)

        biases = torch.stack(
            biases, dim=1
        )  # [batch_size, timesteps-1, num_particles, 3]

        # Compute means according to TPS-DPS
        means = (
            1 - self.cfg.friction * self.cfg.timestep
        ) * velocities + self.cfg.timestep / self.target_measure.m * (
            forces[:, :-1] + biases
        )

        # Compute log probability
        log_bpm = self.target_measure.log_prob(velocities[:, 1:] - means[:, :-1]).mean(
            (1)
        )

        return log_bpm

    def train_on_trajectories(self, positions, forces):
        """
        Train on a batch of trajectories.

        Args:
            positions: Positions tensor [batch_size, timesteps, num_particles, 3]
            forces: Forces tensor [batch_size, timesteps, num_particles, 3]

        Returns:
            dict: Training information
        """
        self.optimizer.zero_grad()

        # Compute log biased path measure (log_bpm)
        log_bpm = self.compute_log_bpm(positions, forces)

        # Compute log target measure (log_tm)
        log_tm, _ = self.target_measure(positions, forces)

        # Compute loss
        loss = self.loss(log_bpm, log_tm)

        # Backpropagation
        loss.backward()

        # Optimize
        self.optimizer.step()

        # Return information
        return {
            "loss": loss.item(),
            "log_z": self.loss.log_z.item(),
            "log_bpm": log_bpm.mean().item(),
            "log_tm": log_tm.mean().item(),
        }

    def train_on_gflownet_traj(self, traj):
        """
        Adapt GFlowNet trajectory format for TPS-DPS training.

        Args:
            traj: List of tuples (t, x, r) representing a trajectory
                 Or for OpenMM integration with TPS-DPS, a list with [first_state, info_dict]
                 where info_dict contains "positions" and "forces".

        Returns:
            dict: Training information
        """
        # Check if this is from OpenMM integration with TPS-DPS (special format)
        if len(traj) == 2 and isinstance(traj[1], dict) and "positions" in traj[1]:
            # Extract data from OpenMM integration
            info = traj[1]
            positions = info["positions"].to(self.device)
            forces = info["forces"].to(self.device)

            # Train with this data
            return self.train_on_trajectories(positions, forces)

        # Regular GFlowNet trajectory
        batch_size = traj[0][1].shape[0]
        num_timesteps = len(traj)
        num_particles = traj[0][1].shape[1] // 3

        # Extract positions
        positions = torch.stack(
            [
                x.to(self.device).reshape(batch_size, num_particles, 3)
                for _, x, _ in traj
            ],
            dim=1,
        )

        # Compute forces for each position in the trajectory
        forces = torch.zeros_like(positions)
        for t in range(num_timesteps):
            # Get current positions
            pos_with_grad = positions[:, t].clone().detach().requires_grad_(True)
            pos_flat = pos_with_grad.reshape(batch_size, -1)

            # Compute potential energy
            potential = self.agent.get_potential_energy(pos_flat)
            potential_sum = potential.sum()

            # Compute forces from gradient of potential
            potential_sum.backward()
            forces[:, t] = -pos_with_grad.grad.reshape(batch_size, num_particles, 3)

            # Get policy bias at this timestep/position
            time = torch.tensor(t * self.cfg.dt, device=self.device).reshape(1, 1)
            time = time.repeat(batch_size, 1)
            bias = self.agent.gfn.f(time, pos_flat.detach())
            bias = bias.reshape(batch_size, num_particles, 3)

            # Add bias to forces (TPS-DPS expects total force including bias)
            forces[:, t] += bias

        # Train with this data
        return self.train_on_trajectories(positions, forces)
