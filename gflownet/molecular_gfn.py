import torch
import numpy as np
from torch import nn
from einops import rearrange

# Import necessary functions and classes from the original gflownet code
from gflownet.gflownet import (
    DetailedBalance,
    normal_logp,
    cal_subtb_coef_matrix,
    sample_traj,
    fl_inter_logr,
)


def sample_molecular_traj(gfn, config, logreward_fn, batch_size=None, sigma=None):
    """
    Sample a trajectory for molecular systems, starting from a defined molecular state.

    Args:
        gfn (MolecularGFlowNet): The GFlowNet to sample from.
        config (dict): The configuration.
        logreward_fn (callable): The log reward function.
        batch_size (int, optional): The batch size. Defaults to None.
        sigma (float, optional): The noise level. Defaults to None.

    Returns:
        tuple: A tuple containing the trajectory and information.
            - traj: list of tuples, each containing a time, a state, and a reward.
            - info: dictionary containing information about the trajectory.
    """
    if batch_size is None:
        batch_size = config.batch_size
    device = gfn.device
    if sigma is None:
        sigma = config.sigma

    # Start from the defined molecular state instead of zeros
    x = gfn.task.get_start_state(batch_size).to(device)  # shape: (b, d)
    fl_logr = fl_inter_logr(x, logreward_fn, config, cur_t=0.0, sigma=sigma)
    traj = [(torch.tensor(0.0), x.cpu(), fl_logr.cpu())]
    inter_loss = torch.zeros(batch_size).to(device)

    x_max = 0.0
    for cur_t in torch.arange(0, config.t_end, config.dt).to(device):
        x, uw_term, u2_term = gfn.step_forward(cur_t, x, config.dt, sigma=sigma)
        x = x.detach()
        fl_logr = (
            fl_inter_logr(x, logreward_fn, config, cur_t=cur_t + config.dt, sigma=sigma)
            .detach()
            .cpu()
        )
        traj.append((cur_t.cpu() + config.dt, x.detach().cpu(), fl_logr))
        inter_loss += (u2_term + uw_term).detach()
        x_max = max(x_max, x.abs().max().item())

    pis_terminal = -gfn.nll_prior(x) - logreward_fn(x)
    pis_log_weight = inter_loss + pis_terminal
    info = {"pis_logw": pis_log_weight, "x_max": x_max}

    return traj, info


class MolecularGFlowNet(DetailedBalance):
    """
    MolecularGFlowNet is a modified GFlowNet algorithm tailored for
    sampling from molecular Boltzmann target densities. It removes
    extraneous experiments (e.g. funnel, manywell) and focuses on
    high-dimensional molecular tasks.
    """

    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)
        self.Lambda = float(cfg.subtb_lambda)
        # Use cfg.N as the trajectory length for molecular simulations.
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N))
        self.register_buffer("coef", coef, persistent=False)

    def get_flow_logp_from_traj(self, traj, debug=False):
        """
        Get the flow log probability from a trajectory.

        Args:
            traj (list): A list of tuples, each containing a time, a state, and a reward.
                            - t: time, shape: []
                            - x: state, shape: [b, d]
                            - r: reward, shape: [b]
            debug (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            dict: A dictionary containing the flow log probability, the forward probability, and the backward probability.
                - flows: shape: [b, T]
                - log_pf: shape: [b, T]
                - log_pb: shape: [b, T]
        #"""
        # print("Length of traj:", len(traj))
        # print("First element of traj:", traj[0])
        # print("traj[0][1].shape:", traj[0][1].shape)
        batch_size = traj[0][1].shape[0]  # Get batch size from first trajectory element
        xs = [
            x.to(self.device) for (t, x, r) in traj
        ]  # States from trajectory, shape: [T, b, d]
        ts = [
            t[None].to(self.device).repeat(batch_size, 1) for (t, x, r) in traj
        ]  # Times

        # print("xs shape:", xs[0].shape)
        # print("ts shape:", ts[0].shape)

        state = torch.cat(xs[:-1], dim=0)  # All states except last (T*b, d)
        next_state = torch.cat(xs[1:], dim=0)  # All states except first (T*b, d)
        time = torch.cat(ts[:-1], dim=0)  # All times except last (T*b, 1)
        next_time = torch.cat(ts[1:], dim=0)  # All times except first (T*b, 1)

        # print("state shape:", state.shape)
        # print("next_state shape:", next_state.shape)
        # print("time shape:", time.shape)
        # print("next_time shape:", next_time.shape)

        log_pf = self.log_pf(time, state, next_state)  # Forward probabilities
        log_pb = self.log_pb(next_time, next_state, state)  # Backward probabilities
        log_pf = rearrange(log_pf, "(T b) -> b T", b=batch_size)
        log_pb = rearrange(log_pb, "(T b) -> b T", b=batch_size)

        states = torch.cat(xs, dim=0)  # All states ((T+1)*b, d)
        times = torch.cat(ts, dim=0)  # All times ((T+1)*b, 1)
        flows = self.flow(times, states).squeeze(
            -1
        )  # Calculate flows ((T+1)*b,1) -> ((T+1)*b,)
        flows = rearrange(
            flows, "(T1 b) -> b T1", b=batch_size
        )  # Reshape from [T+1*b] to [b, T+1]

        # Handle rewards with potential extra dimension
        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)
        if logrs.dim() > 1:
            # If rewards have shape [T*b, d], flatten the last dimension
            logrs = logrs.view(logrs.shape[0], -1).sum(-1)
        # Now reshape to [b, T]
        T1 = len(traj)  # number of timesteps
        logrs = logrs.view(T1, batch_size).t()

        flows = flows + logrs  # (b, T+1)

        logr_terminal = self.logr_from_traj(traj)  # (b,)

        # print("logr_terminal shape:", logr_terminal.shape)
        flows[:, -1] = logr_terminal

        # print("Shape of log_pf:", log_pf.shape)  # (b, T)
        # print("Shape of log_pb:", log_pb.shape)  # (b, T)
        # print("Shape of flows:", flows.shape)  # (b, T+1)

        return {"log_pf": log_pf, "log_pb": log_pb, "flows": flows}

    def train_loss(self, traj):
        batch_size = traj[0][1].shape[0]
        print("Getting flow logp from traj")
        logits_dict = self.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = (
            logits_dict["flows"],
            logits_dict["log_pf"],
            logits_dict["log_pb"],
        )
        diff_logp = log_pf - log_pb
        # Check shapes before concatenating
        # print("diff_logp shape:", diff_logp.shape)
        # print("torch.zeros(batch_size, 1).shape:", torch.zeros(batch_size, 1).shape)
        # print("diff_logp.cumsum(dim=-1).shape:", diff_logp.cumsum(dim=-1).shape)

        # Shape of [b, T]
        diff_logp_padded = torch.cat(
            (torch.zeros(batch_size, 1).to(diff_logp), diff_logp.cumsum(dim=-1)), dim=1
        )
        # Compute A1 such that A1[:, i, j] is the cumulative sum of diff_logp between timesteps i and j.
        A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
        A2 = flows[:, :, None] - flows[:, None, :] + A1
        A2 = (A2 / self.data_ndim).pow(2).mean(dim=0)
        loss = torch.triu(A2 * self.coef, diagonal=1).sum()
        info = {"loss_dlogp": loss.item()}
        # Compute a model logZ estimate from flow evaluated on the initial state.
        logZ_model = self.flow(
            traj[0][0][None, None].to(self.device), traj[0][1][:1, :].to(self.device)
        ).detach()
        info["logz_model"] = logZ_model.mean().item()
        return loss, info

    def train_step(self, traj):
        """Override train_step to use molecular trajectory sampling"""
        self.train()
        loss, info = self.train_loss(traj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info["loss_train"] = loss.item()
        return info

    def eval_step(self, num_samples, logreward_fn=None):
        print("Starting eval step")
        self.eval()
        if logreward_fn is None:
            logreward_fn = self.logr_fn
        # Use molecular trajectory sampling instead of default
        traj, sample_info = sample_molecular_traj(
            self, self.cfg, logreward_fn, batch_size=num_samples
        )
        logw = self.log_weight(traj)
        logZ_eval = torch.logsumexp(-logw, dim=0) - np.log(num_samples)
        logZ_elbo = torch.mean(-logw, dim=0)
        info = {"logz": logZ_eval.item(), "logz_elbo": logZ_elbo.item()}
        return traj, info

    def visualize(self, traj, logreward_fn=None, step=-1):
        # For molecular tasks (which are high-dimensional), standard 2D visualization is not applicable.
        print("Visualization is not supported for molecular tasks.")

    def zero(self, batch_size, device=None):
        """Override zero to use molecular starting state instead of zeros"""
        device = self.device if device is None else device
        return self.task.get_start_state(batch_size).to(device)
