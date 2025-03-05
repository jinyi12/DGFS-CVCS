import torch
import numpy as np
from torch import nn


class TargetMeasure(nn.Module):
    """
    Base target measure class following the structure of TPS-DPS algorithm.
    This implementation provides a framework similar to the reference TPS-DPS code.
    """

    def __init__(self, args):
        """
        Initialize the target measure class.

        Args:
            args: Configuration object containing parameters
        """
        super().__init__()
        self.sigma = args.sigma
        self.timestep = getattr(args, "timestep", 1.0)
        self.friction = getattr(args, "friction", 0.001)

        # These would be set by subclasses or later
        self.m = None  # Mass
        self.log_prob = None  # Log probability function for velocities

    def __call__(self, positions, forces):
        """
        Compute the target measure for trajectories.

        Args:
            positions: Trajectory positions [batch_size, timesteps, num_particles, 3]
            forces: Trajectory forces [batch_size, timesteps, num_particles, 3]

        Returns:
            tuple: (log_tm, final_idx)
                - log_tm: Log target measure with shape [batch_size]
                - final_idx: Final indices with shape [batch_size]
        """
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri = self.terminal_state_measure(positions)

        log_tm = log_upm + log_ri
        # Since we use the terminal state, final_idx is the last timestep
        final_idx = torch.full(
            (positions.shape[0],),
            positions.shape[1] - 1,
            device=positions.device,
            dtype=torch.long,
        )

        return log_tm, final_idx

    def unbiased_path_measure(self, positions, forces):
        """
        Compute the unbiased path measure component.

        Args:
            positions: Trajectory positions [batch_size, timesteps, num_particles, 3]
            forces: Trajectory forces [batch_size, timesteps, num_particles, 3]

        Returns:
            torch.Tensor: Unbiased path measure with shape [batch_size]
        """
        # Default implementation (should be overridden by subclasses)
        return torch.zeros(positions.shape[0], device=positions.device)

    def terminal_state_measure(self, positions):
        """
        Compute the measure for the terminal state.
        This replaces the relaxed indicator in the reference implementation.

        Args:
            positions: Trajectory positions [batch_size, timesteps, num_particles, 3]

        Returns:
            torch.Tensor: Terminal state measure with shape [batch_size]
        """
        # Default implementation (should be overridden by subclasses)
        return torch.zeros(positions.shape[0], device=positions.device)


class BoltzmannTargetMeasure(TargetMeasure):
    """
    A target measure that samples from the Boltzmann distribution using
    a potential energy function: exp(-beta * U(x)).

    This is a variant of TPS-DPS where the 'relaxed_indicator' is replaced
    with the Boltzmann density for the terminal state.
    """

    def __init__(self, args, potential_fn=None, mass=None):
        """
        Initialize the Boltzmann target measure.

        Args:
            args: Configuration object containing parameters
            potential_fn: Function that computes the potential energy of a state
            mass: Mass tensor for the particles
        """
        super().__init__(args)
        self.args = args
        self.beta = 1.0 / (args.temperature * 8.3144621e-3)  # 1/(kB*T) in mol/kJ
        self.potential_fn = potential_fn

        # Set mass if provided
        if mass is not None:
            self.set_mass(mass)

    def set_mass(self, mass):
        """
        Set the mass parameter and initialize the log probability function.

        Args:
            mass: Mass tensor for particles
        """
        self.m = mass
        # Initialize log_prob based on mass (similar to TPS-DPS)
        kB = 8.3144621e-3  # Boltzmann constant in kJ/(mol·K)
        self.T = self.args.temperature
        self.kBT = kB * self.T

        def log_prob_fn(v):
            """
            Log probability function for Maxwell-Boltzmann distribution.

            Args:
                v: Velocity tensor

            Returns:
                torch.Tensor: Log probability
            """
            # Based on Maxwell-Boltzmann distribution
            log_gaussian = -0.5 * (self.m / self.kBT)[:, None] * v.square()
            return log_gaussian.sum(dim=(-2, -1))

        self.log_prob = log_prob_fn

    def set_potential(self, potential_fn):
        """Set the potential energy function after initialization."""
        self.potential_fn = potential_fn

    def set_temperature(self, temperature):
        """Update the temperature (and thus beta) for the Boltzmann distribution."""
        self.beta = 1.0 / (temperature * 8.3144621e-3)  # 1/(kB*T) in mol/kJ

    def terminal_state_measure(self, positions):
        """
        Compute the Boltzmann density for the terminal state.
        This replaces the relaxed_indicator in the original TPS-DPS.

        Args:
            positions: Trajectory positions [batch_size, timesteps, num_particles, 3]

        Returns:
            torch.Tensor: Log Boltzmann weights with shape [batch_size]
        """
        if self.potential_fn is None:
            raise ValueError("Potential energy function not set")

        # Get the terminal state (last timestep)
        terminal_positions = positions[:, -1]

        # Reshape to the expected format if needed
        batch_size = terminal_positions.shape[0]
        flattened_positions = terminal_positions.reshape(batch_size, -1)

        # Compute potential energy
        potential = self.potential_fn(flattened_positions)

        # Compute log Boltzmann weight: -beta * U(x)
        log_boltzmann = -self.beta * potential

        return log_boltzmann

    def unbiased_path_measure(self, positions, forces):
        """
        Compute the unbiased path measure component following TPS-DPS.

        Args:
            positions: Trajectory positions [batch_size, timesteps, num_particles, 3]
            forces: Trajectory forces [batch_size, timesteps, num_particles, 3]

        Returns:
            torch.Tensor: Unbiased path measure with shape [batch_size]
        """
        if self.log_prob is None or self.m is None:
            raise ValueError("Mass or log_prob not set")

        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        means = (
            1 - self.friction * self.timestep
        ) * velocities + self.timestep / self.m * forces[:, :-1]

        log_upm = self.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1))
        return log_upm


class MolecularBoltzmannTarget(BoltzmannTargetMeasure):
    """
    Specific implementation of Boltzmann target measure for molecular systems,
    which integrates with molecular dynamics simulators to compute the potential energy.
    """

    def __init__(self, args, molecular_agent=None):
        """
        Initialize with molecular agent for potential calculation.

        Args:
            args: Configuration object
            molecular_agent: MolecularGFlowNetAgent instance for potential calculation
        """
        super().__init__(args)
        self.molecular_agent = molecular_agent

        # If agent is provided, set up everything
        if molecular_agent is not None:
            self.set_molecular_agent(molecular_agent)

    def set_molecular_agent(self, agent):
        """
        Set the molecular agent and configure potential function and mass.

        Args:
            agent: MolecularGFlowNetAgent instance
        """
        self.molecular_agent = agent

        # Set the potential function
        self.set_potential(agent.get_potential_energy)

        # Get mass from MDs or task
        if agent.use_openmm:
            self.set_mass(agent.mds.m)
        elif hasattr(agent.task, "mass"):
            self.set_mass(agent.task.mass)
        else:
            # Default mass if not available (should be overridden)
            default_mass = torch.ones(
                (1, agent.task.num_particles), device=agent.device
            )
            self.set_mass(default_mass)

    def compute_log_tm_from_traj(self, traj):
        """
        Compute log target measure directly from a GFlowNet trajectory.

        This is a convenience method for integrating with GFlowNet training.

        Args:
            traj: List of tuples (t, x, r) representing a trajectory

        Returns:
            torch.Tensor: Log target measure with shape [batch_size]
        """
        # Extract the final state from the trajectory
        final_state = traj[-1][1]  # (t, x, r) tuple, x is at index 1

        # Move to the device where computation happens
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, device=self.molecular_agent.device)

        # Use terminal state measure for GFlowNet trajectory
        return self.terminal_state_measure(final_state.unsqueeze(1))
