import torch
import numpy as np
from torch import nn
from einops import rearrange
# Import necessary functions and classes from the original gflownet code
from gflownet.gflownet import DetailedBalance, normal_logp, cal_subtb_coef_matrix, sample_traj

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
        self.register_buffer('coef', coef, persistent=False)
    
    def get_flow_logp_from_traj(self, traj, debug=False):
        batch_size = traj[0][1].shape[0]
        xs = [x.to(self.device) for (t, x, r) in traj]
        ts = [t[None].to(self.device).repeat(batch_size, 1) for (t, x, r) in traj]
        
        state = torch.cat(xs[:-1], dim=0)
        next_state = torch.cat(xs[1:], dim=0)
        time = torch.cat(ts[:-1], dim=0)
        next_time = torch.cat(ts[1:], dim=0)
        log_pf = self.log_pf(time, state, next_state)
        log_pb = self.log_pb(next_time, next_state, state)
        log_pf = rearrange(log_pf, "(T b) -> b T", b=batch_size)
        log_pb = rearrange(log_pb, "(T b) -> b T", b=batch_size)
        
        states = torch.cat(xs, dim=0)
        times = torch.cat(ts, dim=0)
        flows = self.flow(times, states).squeeze(-1)
        flows = rearrange(flows, "(T1 b) -> b T1", b=batch_size)
        
        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)
        logrs = rearrange(logrs, "(T1 b) -> b T1", b=batch_size)
        flows = flows + logrs
        
        logr_terminal = self.logr_from_traj(traj)
        flows[:, -1] = logr_terminal
        
        return {"log_pf": log_pf, "log_pb": log_pb, "flows": flows}
    
    def train_loss(self, traj):
        batch_size = traj[0][1].shape[0]
        logits_dict = self.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = logits_dict["flows"], logits_dict["log_pf"], logits_dict["log_pb"]
        diff_logp = log_pf - log_pb
        diff_logp_padded = torch.cat(
            (torch.zeros(batch_size, 1).to(diff_logp), diff_logp.cumsum(dim=-1)),
            dim=1
        )
        # Compute A1 such that A1[:, i, j] is the cumulative sum of diff_logp between timesteps i and j.
        A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
        A2 = flows[:, :, None] - flows[:, None, :] + A1
        A2 = (A2 / self.data_ndim).pow(2).mean(dim=0)
        loss = torch.triu(A2 * self.coef, diagonal=1).sum()
        info = {"loss_dlogp": loss.item()}
        # Compute a model logZ estimate from flow evaluated on the initial state.
        logZ_model = self.flow(traj[0][0][None, None].to(self.device),
                               traj[0][1][:1, :].to(self.device)).detach()
        info["logz_model"] = logZ_model.mean().item()
        return loss, info
    
    def train_step(self, traj):
        self.train()
        loss, info = self.train_loss(traj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info["loss_train"] = loss.item()
        return info
    
    def eval_step(self, num_samples, logreward_fn=None):
        self.eval()
        if logreward_fn is None:
            logreward_fn = self.logr_fn
        traj, sample_info = sample_traj(self, self.cfg, logreward_fn, batch_size=num_samples)
        logw = self.log_weight(traj)
        logZ_eval = torch.logsumexp(-logw, dim=0) - np.log(num_samples)
        logZ_elbo = torch.mean(-logw, dim=0)
        info = {"logz": logZ_eval.item(),
                "logz_elbo": logZ_elbo.item()}
        return traj, info
    
    def visualize(self, traj, logreward_fn=None, step=-1):
        # For molecular tasks (which are high-dimensional), standard 2D visualization is not applicable.
        print("Visualization is not supported for molecular tasks.") 