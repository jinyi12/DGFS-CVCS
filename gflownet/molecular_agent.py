import torch
from gflownet.molecular_gfn import MolecularGFlowNet
from gflownet.gflownet import (
    sample_traj,
)  # reuse the sampling function from the original file
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs


class MolecularGFlowNetAgent:
    """
    The MolecularGFlowNetAgent integrates the MolecularGFlowNet algorithm with
    the molecular simulation task, following the structure of DiffusionPathSampler.
    """

    def __init__(self, cfg, mds):
        """Initialize agent with config and molecular dynamics simulator.

        Args:
            cfg: Configuration object
            mds: MolecularMDs instance managing multiple MD simulations
        """
        self.cfg = cfg
        # Create GFlowNet with template task from mds
        self.device = cfg.device
        self.gfn = MolecularGFlowNet(cfg, task=mds.template_task).to(self.device)

    # def sample(self, num_samples):
    #     """
    #     Sample a batch of trajectories using the GFlowNet policy.
    #     """
    #     self.gfn.eval()
    #     with torch.no_grad():
    #         traj, info = sample_traj(
    #             self.gfn, self.cfg, self.gfn.logr_fn, batch_size=num_samples
    #         )
    #     return traj, info

    def sample(self, num_samples, mds, temperature):
        """Sample trajectories using OpenMM integration.

        Args:
            num_samples: Number of trajectories to sample
            mds: MolecularMDs instance to use for sampling
            temperature: Current temperature for sampling

        Returns:
            tuple: (traj, info) containing trajectories and sampling info
        """
        device = self.gfn.device
        print("Device of gfn:", device)
        positions = torch.zeros(
            (num_samples, int(self.cfg.t_end / self.cfg.dt) + 1, mds.num_particles, 3),
            device=device,
        )
        print("Positions shape:", positions.shape)
        print(
            "1st index of position cfg.t_end / cfg.dt shape:",
            positions[:, 0].shape,
        )
        forces = torch.zeros_like(positions)

        # Get initial state from all simulations
        position, force = mds.report()
        print("Position shape:", position.shape)
        print("Force shape:", force.shape)
        positions[:, 0] = position
        forces[:, 0] = force

        # Reset all simulations
        mds.reset()
        mds.set_temperature(temperature)

        # Build trajectory
        traj = []
        t = torch.tensor(0.0)
        x = position.reshape(num_samples, -1)
        fl_logr = self.gfn.logr_fn(x)
        traj.append((t, x.cpu(), fl_logr.cpu()))

        # Sample trajectories
        for s in range(1, int(self.cfg.t_end / self.cfg.dt) + 1):

            cur_t = torch.tensor(s * self.cfg.dt, device=device)

            # check device of cur_t, and x
            # print(f"cur_t device: {cur_t.device}, x device: {x.device}")

            # Get GFlowNet policy output (force bias)
            bias = self.gfn.f(cur_t, x.detach()).detach()
            bias = bias.reshape(num_samples, mds.num_particles, 3)

            # Step all simulations with bias force
            mds.step(bias)

            # Get updated states
            position, force = mds.report()
            positions[:, s] = position
            forces[:, s] = force - 1e-6 * torch.tensor(
                bias, device=device
            )  # kJ/(mol*nm) -> (da*nm)/fs**2

            # Update trajectory
            x = position.reshape(num_samples, -1)
            fl_logr = self.gfn.logr_fn(x)
            traj.append((cur_t.cpu(), x.cpu(), fl_logr.cpu()))
            # traj.append((cur_t, x, fl_logr))

        # Reset all simulations
        mds.reset()

        # Compute trajectory info
        logw = self.gfn.log_weight(traj)
        info = {"pis_logw": logw, "x_max": positions.abs().max().item()}

        return traj, info

    def train(self, traj):
        """Perform a training step given a batch of trajectories."""
        self.gfn.train()
        loss_info = self.gfn.train_step(traj)
        return loss_info
