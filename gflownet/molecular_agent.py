import torch
from gflownet.molecular_gfn import MolecularGFlowNet, sample_molecular_traj
from gflownet.gflownet import (
    sample_traj,
    fl_inter_logr,
)  # reuse the sampling function from the original file
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs


class MolecularGFlowNetAgent:
    """
    The MolecularGFlowNetAgent integrates the MolecularGFlowNet algorithm with
    molecular simulation, supporting both GFlowNet diffusion updates and
    OpenMM integration for state transitions.
    """

    def __init__(self, cfg, task, mds):
        """Initialize agent with config, molecular task, and MD simulator.

        Args:
            cfg: Configuration object
            task: MolecularTask instance for task definition
            mds: MolecularMDs instance for OpenMM integration and potential energy
        """
        self.cfg = cfg
        self.device = cfg.device

        # Store both task and MDs
        self.task = task  # For task-specific operations
        self.mds = mds  # For MD simulations and potential energy

        # Validate that task and mds have compatible configurations
        if self.task.num_particles != self.mds.num_particles:
            raise ValueError(
                f"Task and MDs have different particle counts: "
                f"{self.task.num_particles} vs {self.mds.num_particles}"
            )

        # Set flag for using OpenMM for integration (state updates)
        self.use_openmm_int = getattr(cfg, "use_openmm_int", False)

        # Create GFlowNet with the appropriate task
        self.gfn = MolecularGFlowNet(cfg, task=self.task).to(self.device)

        # Current temperature (will be updated during training)
        self.temperature = getattr(cfg, "temperature", 300)

    def get_potential_energy(self, x):
        """Get potential energy for given positions.

        Args:
            x: Flattened positions tensor [batch_size, num_particles*3]

        Returns:
            torch.Tensor: Potential energies [batch_size]
        """
        # Use MDs for potential energy calculation
        return self.mds.get_potential_energy(x)

    def set_temperature(self, temperature):
        """Set the temperature for both MDs and internal state.

        Args:
            temperature: Temperature value in Kelvin
        """
        self.temperature = temperature
        self.mds.set_temperature(temperature)
        self.gfn.set_temperature(temperature)  # Update GFlowNet's temperature as well

    def sample_with_diffusion(self, num_samples):
        """
        Sample a batch of trajectories using the original GFlowNet diffusion update
        instead of MD simulation with OpenMM.

        Args:
            num_samples: Number of trajectories to sample

        Returns:
            tuple: (traj, info) containing trajectories and sampling info
        """
        self.gfn.eval()
        with torch.no_grad():
            # Use the molecular-specific sampling function that starts from the correct initial state
            traj, info = sample_molecular_traj(
                self.gfn,
                self.cfg,
                self.gfn.logr_fn,
                batch_size=num_samples,
                potential_fn=self.get_potential_energy,  # Use our unified potential energy function
                temperature=self.temperature,  # Pass current temperature
            )
        return traj, info

    def sample_with_openmm(self, num_samples):
        """Sample trajectories using OpenMM integration.

        Args:
            num_samples: Number of trajectories to sample

        Returns:
            tuple: (traj, info) containing trajectories and sampling info
        """
        device = self.gfn.device
        positions = torch.zeros(
            (
                num_samples,
                int(self.cfg.t_end / self.cfg.dt) + 1,
                self.mds.num_particles,
                3,
            ),
            device=device,
        )
        forces = torch.zeros_like(positions)

        # Reset all simulations to start state
        self.mds.reset()

        # Get initial state from all simulations
        position, force = self.mds.report()
        positions[:, 0] = position
        forces[:, 0] = force

        # Build trajectory
        traj = []
        t = torch.tensor(0.0)
        x = position.reshape(num_samples, -1)

        # Compute initial reward using potential energy
        potential = self.get_potential_energy(x)
        # Use GFlowNet's method for Boltzmann reward
        fl_logr = self.gfn.get_boltzmann_reward(x, potential, self.temperature)

        traj.append((t, x.cpu(), fl_logr.cpu()))

        # Sample trajectories
        for s in range(1, int(self.cfg.t_end / self.cfg.dt) + 1):
            cur_t = torch.tensor(s * self.cfg.dt, device=device)

            # Get GFlowNet policy output (force bias)
            bias = self.gfn.f(cur_t, x.detach()).detach()
            bias = bias.reshape(num_samples, self.mds.num_particles, 3)

            # Step all simulations with bias force
            self.mds.step(bias)

            # Get updated states
            position, force = self.mds.report()
            positions[:, s] = position
            forces[:, s] = force - bias  # Store corrected forces for TPS-DPS

            # Update trajectory
            x = position.reshape(num_samples, -1)

            # Compute reward using potential energy
            potential = self.get_potential_energy(x)
            # Use GFlowNet's method for Boltzmann reward
            fl_logr = self.gfn.get_boltzmann_reward(x, potential, self.temperature)

            traj.append((cur_t.cpu(), x.cpu(), fl_logr.cpu()))

        # Reset all simulations
        self.mds.reset()

        # Compute trajectory info
        logw = self.gfn.log_weight(traj)
        info = {
            "pis_logw": logw,
            "x_max": positions.abs().max().item(),
            "positions": positions.detach().cpu(),
            "forces": forces.detach().cpu(),
        }

        # For TPS-DPS, we need to return the info in a format that can be
        # consumed by the trainer
        if self.use_openmm_int and getattr(self.cfg, "use_tps_dps", False):
            return [traj[0], info]  # Just need first state and the positions/forces
        else:
            return traj, info

    def sample(self, num_samples):
        """Sample trajectories using either GFlowNet diffusion or OpenMM.

        Args:
            num_samples: Number of trajectories to sample

        Returns:
            tuple: (traj, info) containing trajectories and sampling info
        """
        if self.use_openmm_int:
            return self.sample_with_openmm(num_samples)
        else:
            return self.sample_with_diffusion(num_samples)

    def train(self, traj):
        """Perform a training step given a batch of trajectories."""
        self.gfn.train()
        loss_info = self.gfn.train_step(traj)
        return loss_info
