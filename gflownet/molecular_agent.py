import torch
from gflownet.molecular_gfn import MolecularGFlowNet
from gflownet.gflownet import (
    sample_traj,
)  # reuse the sampling function from the original file
from gflownet.tasks.molecular import MolecularTask


class MolecularGFlowNetAgent:
    """
    The MolecularGFlowNetAgent integrates the MolecularGFlowNet algorithm with
    the molecular simulation task. It provides a simple interface for sampling
    trajectories and training the model, following the structure of the reference
    TPS/DPS codebase.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Instantiate the molecular task (which sets up the OpenMM system, etc.)
        self.task = MolecularTask(cfg)
        # Create the molecular-specific GFlowNet agent with the task
        self.gfn = MolecularGFlowNet(cfg, task=self.task)
        self.device = self.gfn.device

    def sample(self, num_samples):
        """
        Sample a batch of trajectories using the GFlowNet policy.
        """
        self.gfn.eval()
        with torch.no_grad():
            traj, info = sample_traj(
                self.gfn, self.cfg, self.gfn.logr_fn, batch_size=num_samples
            )
        return traj, info

    def train(self, traj):
        """
        Perform a training step given a batch of trajectories.
        """
        self.gfn.train()
        loss_info = self.gfn.train_step(traj)
        return loss_info
