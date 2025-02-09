# gflownet/__init__.py

```py
# Make gflownet a proper Python package
from .network import FourierMLP, TimeConder
from .gflownet import DetailedBalance, sample_traj
from .molecular_agent import MolecularGFlowNetAgent
from .utils import loss2ess_info

__all__ = [
    "FourierMLP",
    "TimeConder",
    "DetailedBalance",
    "sample_traj",
    "MolecularGFlowNetAgent",
    "loss2ess_info",
]

```

# gflownet/configs/main.yaml

```yaml
defaults:
  - _self_
  - target: gm.yaml

d: 0 # device
seed: 0

lr: 1e-4
zlr: 1e-3
stlam: 2. # subtb lambda
wd: 1e-7 # weight decay
bs: 256

tstep: 100 # only for transition based buffer
steps: 5000 # training steps
log_freq: 20
eval_freq: 100
eval_n: 2000 # number of samples for estimating logZ

dt: 0.01 # step size, h in the paper
N: 100
sigma: 1.
sgmit: -1.
data_ndim: ${target.ndim}
xclip: -1 # clip x (state) to avoid overflow
rmin: -1e6 # clip the lowest log reward value

f_func:
  _target_: gflownet.network.FourierMLP
  in_shape: ${data_ndim}
  out_shape: ${data_ndim}
  num_layers: 2
  channels: 64
  res: False

f: "tgrad" # "f" or "tgrad"
nn_clip: 1e4
lgv_clip: 1e4

g_func:
  _target_: gflownet.network.IdentityOne

```

# gflownet/gflownet.py

```py
import copy
from time import time
from hydra.utils import instantiate

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from gflownet.network import FourierMLP, TimeConder
from gflownet.utils import normal_logp, num_to_groups, loss2ess_info
from target.plot import viz_sample2d, viz_kde2d, viz_contour_sample2d


def get_alg(cfg, task=None):
    # alg = SubTrajectoryBalanceTransitionBased(cfg, task=task)
    alg = SubTrajectoryBalanceTrajectoryBased(cfg, task=task)
    return alg


def fl_inter_logr(x, logreward_fn, config, cur_t, sigma=None):  # x: (bs, dim)
    """
    Calculate the intermediate log reward for a trajectory.

    Args:
        x (torch.Tensor): The state, shape: [b, d]
        logreward_fn (callable): The log reward function
        config (dict): The configuration
        cur_t (float): The current time
        sigma (float, optional): The noise level. Defaults to None.

    Returns:
        torch.Tensor: The intermediate log reward, shape: [b]
    """
    if sigma is None:
        sigma = config.sigma

    if isinstance(cur_t, torch.Tensor):
        # print(cur_t.shape) # (bs, 1) or (,)
        if cur_t.ndim <= 1:
            cur_t = cur_t.item()
    ratio = cur_t / config.t_end
    # assert 0 <= ratio <= 1

    coef = max(np.sqrt(0.01 * config.t_end), np.sqrt(cur_t))  # cur_t could be 0
    logp0 = normal_logp(x, 0.0, coef * sigma)

    # print(
    #     "Shapes of x, logp0, logreward_fn(x):",
    #     x.shape,  # [b, d]
    #     logp0.shape,  # [b]
    #     logreward_fn(x).shape,  # it is now [b, 1]
    # )

    # ! Since logreward_fn(x) is now [b, 1], the result is [b, b] since the broadcast
    # ! logp0 is [b], so the result is [b, b]
    #! Fixed this issue by squeezing the last dimension of logreward_fn(x) in the potential calculation
    fl_logr = logreward_fn(x) * ratio + logp0 * (1 - ratio)

    return fl_logr


def sample_traj(gfn, config, logreward_fn, batch_size=None, sigma=None):
    """
    Sample a trajectory from the GFlowNet.

    Args:
        gfn (GFlowNet): The GFlowNet to sample from.
        config (dict): The configuration.
        logreward_fn (callable): The log reward function.
        batch_size (int, optional): The batch size. Defaults to None.
        sigma (float, optional): The noise level. Defaults to None.

    Returns:
        tuple: A tuple containing the trajectory and information.
            - traj: list of tuples, each containing a time, a state, and a reward.
                - t: time, a scalar, shape: []
                - x: state, shape: [b, d]
                - r: reward, shape: [b]
            - info: dictionary containing information about the trajectory.
    """
    if batch_size is None:
        batch_size = config.batch_size
    device = gfn.device
    if sigma is None:
        sigma = config.sigma

    #! For usual DGFS, we start from a Dirac distribution at zero,
    #! but for molecular GFNs, we start from a given starting state of configuration
    x = gfn.zero(batch_size).to(device)  # shape: (b, d)
    fl_logr = fl_inter_logr(x, logreward_fn, config, cur_t=0.0, sigma=sigma)
    # check if the log reward is of shape [b, 1]
    print("fl_logr.shape:", fl_logr.shape)  # Now it is [b, b], why?
    traj = [(torch.tensor(0.0), x.cpu(), fl_logr.cpu())]  # a tuple of (t, x, logr)
    inter_loss = torch.zeros(batch_size).to(device)

    x_max = 0.0
    for cur_t in torch.arange(0, config.t_end, config.dt).to(device):
        x, uw_term, u2_term = gfn.step_forward(cur_t, x, config.dt, sigma=sigma)
        # print("Current time:", cur_t)
        # print("x output of step_forward:", x)
        # No issue here, checked
        x = x.detach()
        fl_logr = (
            fl_inter_logr(x, logreward_fn, config, cur_t=cur_t + config.dt, sigma=sigma)
            .detach()
            .cpu()
        )
        # check if detach is needed
        # print("Trying x.detach().cpu():", x.detach().cpu()) # no issue
        traj.append((cur_t.cpu() + config.dt, x.detach().cpu(), fl_logr))
        inter_loss += (u2_term + uw_term).detach()
        x_max = max(x_max, x.abs().max().item())

    pis_terminal = -gfn.nll_prior(x) - logreward_fn(x)
    pis_log_weight = inter_loss + pis_terminal
    info = {"pis_logw": pis_log_weight, "x_max": x_max}

    return traj, info


class GFlowNet(nn.Module):
    """
    For PIS modeling: s0 is fixed to be zero
    thus PB(s0|s1) == 1
    """

    def __init__(self, cfg, task=None):
        super().__init__()
        self.cfg = cfg

        self.data_ndim = cfg.data_ndim  # int(np.prod(data_shape))
        self.register_buffer("x0", torch.zeros((1, self.data_ndim)))  # for pis modeling
        self.t_end = cfg.t_end
        self.dt = cfg.dt

        self.g_func = instantiate(cfg.g_func)
        self.f_func = instantiate(cfg.f_func)  # learnable
        self.nn_clip = cfg.nn_clip
        self.lgv_clip = cfg.lgv_clip
        self.task = task
        self.grad_fn = task.score
        self.select_f(f_format=cfg.f)
        self.xclip = cfg.xclip
        self.logr_fn = lambda x: -task.energy(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def sigma(self):  # simply return a float w/o grad
        return self.cfg.sigma

    def save(self, path):
        raise NotImplementedError

    def get_optimizer(self):
        return torch.optim.Adam(self.param_ls, weight_decay=self.cfg.weight_decay)

    def select_f(self, f_format=None):
        _fn = self.f_func
        if f_format == "f":

            def _fn(t, x):
                return torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)

        elif f_format in ["t_tnet_grad", "tgrad"]:
            self.lgv_coef = TimeConder(self.cfg.f_func.channels, 1, 3)

            def _fn(t, x):
                grad = torch.nan_to_num(self.grad_fn(x))
                grad = torch.clip(grad, -self.lgv_clip, self.lgv_clip)

                f = torch.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                lgv_coef = self.lgv_coef(t)
                return f - lgv_coef * grad

        else:
            raise RuntimeError
        self.param_fn = _fn

    # t: scalar; state: (b, state_dim)
    def f(self, t, state):  # same as self.param_fn
        x = torch.nan_to_num(state)
        control = self.param_fn(t, x)  # .flatten(start_dim=1)
        return control

    def zero(self, batch_size, device=None):
        device = self.device if device is None else device
        return self.x0.expand(batch_size, -1).to(device)

    def nll_prior(self, state):  # nll on terminal state
        return -normal_logp(state, 0.0, np.sqrt(self.t_end) * self.sigma)

    # t -> t + dt; n -> n + 1; here t is scalar tensor
    def step_forward(self, t, state, dt, sigma=None, return_drift_scale=False):
        sigma = self.sigma if sigma is None else sigma
        std_noise = self.g_func(t, state) * torch.randn_like(state)

        noise_term = std_noise * sigma * np.sqrt(dt)
        pre_drift = self.f(t, state)
        # drift_scale = pre_drift.norm(dim=-1).mean()
        next_state = state + pre_drift * sigma * dt + noise_term
        u2_term = 0.5 * pre_drift.pow(2).sum(dim=-1) * dt
        uw_term = (pre_drift * std_noise).sum(dim=1) * np.sqrt(dt)

        if self.xclip > 0:  # avoid nan
            next_state = torch.clip(next_state, -self.xclip, self.xclip)

        return next_state, uw_term, u2_term

    # t -> t - dt; n -> n - 1
    def step_backward(self, t, state, dt):  # not used
        std_noise = self.g_func(t, state) * torch.randn_like(state)
        # n = (t - dt) / dt
        coef = (t - dt) / t  # = n/(n + 1)
        mean = self.x0 * dt / t + coef * state
        noise_term = std_noise * self.sigma * np.sqrt(dt)
        prev_state = mean + coef.sqrt() * noise_term
        return prev_state.detach()

    def log_pf(self, t, state, next_state, dt=None):  # t: (bs, 1), dt: float
        assert t.ndim == state.ndim == next_state.ndim == 2
        dt = self.dt if dt is None else dt
        sigma = self.cfg.sigma
        mean = state + self.f(t, state) * sigma * dt
        log_pf = normal_logp(next_state, mean, sigma * np.sqrt(dt))
        return log_pf

    def log_pb(self, t, state, prev_state, dt=None):  # t: (bs, 1), dt: float
        assert t.ndim == state.ndim == prev_state.ndim == 2
        dt = self.dt if dt is None else dt
        sigma = self.cfg.sigma

        mean = self.x0 * dt / t + (t - dt) / t * state
        sigma_pb = ((t - dt) / t).sqrt() * np.sqrt(dt) * sigma
        log_pb = normal_logp(prev_state, mean, sigma_pb)

        # first step is from Dirac on x0 to a Gaussian, thus PB == 1
        # sigma_pb.min = 0 => nan
        first_step_mask = (t <= dt).squeeze(dim=1)
        # log_pb[first_step_mask] = 0.
        log_pb = torch.where(first_step_mask, torch.zeros_like(log_pb), log_pb)
        return log_pb

    def log_pf_and_pb_traj(self, traj):
        batch_size = traj[0][1].shape[0]

        xs = [x for (t, x, r) in traj]
        state = torch.cat(xs[:-1], dim=0).to(self.device)  # (N*b, d)
        next_state = torch.cat(xs[1:], dim=0).to(self.device)  # (N*b, d)
        ts = [repeat(t[None], "one -> b one", b=batch_size) for (t, x, r) in traj]
        time = torch.cat(ts[:-1], dim=0).to(self.device)  # (N*b, 1)
        next_time = torch.cat(ts[1:], dim=0).to(self.device)  # (N*b, 1)

        log_pb = self.log_pb(next_time, next_state, state)
        log_pb = log_pb.detach()

        if self.cfg.task in ["cox"]:  # save cuda memory
            start_idx = 0
            log_pf = torch.zeros((0,)).to(self.device)
            for bs in num_to_groups(log_pb.shape[0], 1000):
                log_pf_curr = self.log_pf(
                    time[start_idx : start_idx + bs],
                    state[start_idx : start_idx + bs],
                    next_state[start_idx : start_idx + bs],
                )
                log_pf = torch.cat([log_pf, log_pf_curr], dim=0)
                start_idx += bs
        else:
            log_pf = self.log_pf(time, state, next_state)

        return log_pf, log_pb

    def log_weight(self, traj):  # "log q - log p" in VAE notation
        batch_size = traj[0][1].shape[0]
        logr = self.logr_from_traj(traj)
        log_pf, log_pb = self.log_pf_and_pb_traj(traj)  # (N*b,)
        dlogp = reduce(log_pf - log_pb, "(N b) -> b", "sum", b=batch_size)
        return dlogp - logr

    def logr_from_traj(self, traj):
        """
        Get the log reward from a trajectory.

        Args:
            traj (list): A list of tuples, each containing a time, a state, and a reward.
                            - t: time, shape: [b, 1]
                            - x: state, shape: [b, d]
                            - r: reward, shape: [b, 1]

        Returns:
            torch.Tensor: The log reward, shape: [b, 1]
        """
        # check if the reward is of shape [b, 1]
        print("traj[-1][2].shape:", traj[-1][2].shape)
        return traj[-1][2].to(self.device)

    @torch.no_grad()
    def eval_step(self, num_samples, logreward_fn=None):
        if logreward_fn is None:
            logreward_fn = self.logr_fn
        self.eval()

        if self.cfg.task in ["cox"]:  # save cuda memory
            bs_ls = num_to_groups(num_samples, 250)
            pis_logw = None
            x_max = 0.0
            for bs in bs_ls:
                traj_curr, sample_info = sample_traj(
                    self, self.cfg, logreward_fn, batch_size=bs
                )
                logw_curr = sample_info["pis_logw"]
                x_max = max(x_max, sample_info["x_max"])
                if pis_logw is None:
                    pis_logw = logw_curr
                    traj = traj_curr
                else:
                    pis_logw = torch.cat([pis_logw, logw_curr], dim=0)

            print(f"logw_pis={pis_logw.mean().item():.8e}")
            logw = pis_logw
        else:
            traj, sample_info = sample_traj(
                self, self.cfg, logreward_fn, batch_size=num_samples
            )
            pis_logw = sample_info["pis_logw"]
            x_max = sample_info["x_max"]
            logw = self.log_weight(traj)
            print(
                f"logw={logw.mean().item():.8e}, logw_pis={pis_logw.mean().item():.8e}"
            )

        # pis_logZ = torch.logsumexp(-pis_logw, dim=0) - np.log(num_samples)
        # Z = \int R(x) dx = E_{PF(tau)}[R(x)PB(tau|x)/PF(tau)]]
        logZ_eval = torch.logsumexp(-logw, dim=0) - np.log(num_samples)  # (bs,) -> ()
        logZ_elbo = torch.mean(-logw, dim=0)
        info = {"logz": logZ_eval.item(), "logz_elbo": logZ_elbo.item(), "x_max": x_max}

        return traj, info

    def visualize(self, traj, logreward_fn=None, step=-1):
        state = traj[-1][1].detach().cpu()
        step_str = f"-{step}" if step >= 0 else ""

        data_ndim = self.data_ndim
        lim = 7
        if self.cfg.task in ["funnel"]:
            data_ndim = 2
            state = state[:, :2]

        if data_ndim == 2:
            dist_img_path = f"dist{step_str}.png"
            viz_sample2d(state, None, dist_img_path, lim=lim)
            viz_sample2d(state, None, f"dist{step_str}.pdf", lim=lim)

            kde_img_path = f"kde{step_str}.png"
            viz_kde2d(state, None, kde_img_path, lim=lim)
            viz_kde2d(state, None, f"kde{step_str}.pdf", lim=lim)

            alpha = 0.8
            n_contour_levels = 20

            def logp_func(x):
                return -self.task.energy(x.cuda()).cpu()

            contour_img_path = f"contour{step_str}.png"
            viz_contour_sample2d(
                state,
                contour_img_path,
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )
            viz_contour_sample2d(
                state,
                f"contour{step_str}.pdf",
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )

            img_dict = {
                "distribution": dist_img_path,
                "KDE": kde_img_path,
                "contour": contour_img_path,
            }

            return img_dict

        if self.cfg.task in ["manywell"]:
            lim = 3
            x13 = state[:, 0:3:2]  # 1st and 3rd dimension
            dist_img_path = f"distx13{step_str}.png"
            viz_sample2d(x13, None, dist_img_path, lim=lim)
            kde_img_path = f"kdex13{step_str}.png"
            viz_kde2d(x13, None, kde_img_path, lim=lim)

            alpha = 0.8
            n_contour_levels = 20

            def logp_func(x_2d):
                x = torch.zeros((x_2d.shape[0], self.data_ndim))
                x[:, 0] = x_2d[:, 0]
                x[:, 2] = x_2d[:, 1]
                return -self.task.energy(x)

            contour_img_path = f"contourx13{step_str}.png"
            viz_contour_sample2d(
                x13,
                contour_img_path,
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )
            viz_contour_sample2d(
                x13,
                f"contourx13{step_str}.pdf",
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )

            x23 = state[:, 1:3]  # 2nd and 3rd dimension
            dist_img_path2 = f"distx23{step_str}.png"
            viz_sample2d(x23, None, dist_img_path2, lim=lim)
            viz_kde2d(x23, None, f"kdex23{step_str}.png", lim=lim)

            def logp_func(x_2d):
                x = torch.zeros((x_2d.shape[0], self.data_ndim))
                x[:, 1] = x_2d[:, 0]
                x[:, 2] = x_2d[:, 1]
                return -self.task.energy(x)

            contour_img_path2 = f"contourx23{step_str}.png"
            viz_contour_sample2d(
                x23,
                contour_img_path2,
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )
            viz_contour_sample2d(
                x23,
                f"contourx23{step_str}.pdf",
                logp_func,
                lim=lim,
                alpha=alpha,
                n_contour_levels=n_contour_levels,
            )

            return {
                "distribution": dist_img_path,
                "distribution2": dist_img_path2,
                "KDE": kde_img_path,
                "contour": contour_img_path,
                "contour2": contour_img_path2,
            }


class DetailedBalance(GFlowNet):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)
        self.flow = FourierMLP(
            self.data_ndim,
            1,
            num_layers=cfg.f_func.num_layers,
            channels=cfg.f_func.channels,
            zero_init=True,
        )
        self.param_ls = [
            {"params": self.f_func.parameters(), "lr": self.cfg.lr},
            {"params": self.flow.parameters(), "lr": self.cfg.zlr},
        ]
        if hasattr(self, "lgv_coef"):
            self.param_ls.append(
                {"params": self.lgv_coef.parameters(), "lr": self.cfg.lr}
            )
        self.optimizer = self.get_optimizer()

    def save(self, path="alg.pt"):
        self.eval()
        save_dict = {
            "f_func": self.f_func.state_dict(),
            "flow": self.flow.state_dict(),
        }
        torch.save(save_dict, path)

    def load(self, path="alg.pt"):
        save_dict = torch.load(path)
        self.f_func.load_state_dict(save_dict["f_func"])
        self.flow.load_state_dict(save_dict["flow"])


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


class SubTrajectoryBalanceTransitionBased(DetailedBalance):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)

        self.Lambda = float(cfg.subtb_lambda)
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N))  # (N+1, N+1)
        self.register_buffer("coef", coef, persistent=False)

    def train_step(self, traj):
        self.train()
        batch_size = traj[0][1].shape[0]

        xs = [x.to(self.device) for (t, x, r) in traj]
        states = torch.cat(xs, dim=0)  # ((N+1)*b, d)
        states = rearrange(states, "(T1 b) d -> b T1 d", b=batch_size)  # (b, N+1, d)

        ts = [t[None].to(self.device) for (t, x, r) in traj]
        times = torch.cat(ts, dim=0)  # (N+1, 1)
        time_b = repeat(times, "T1 -> T1 one", one=1)  # (N+1, 1)
        # time_coef = (time_b / self.t_end).squeeze(-1)  # (N+1, 1) -> (N+1,)

        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)  # ((N+1)*b, 1)
        logrs = rearrange(logrs, "(T1 b) -> b T1", b=batch_size)  # (b, N+1)

        info = {"logR": logrs[:, -1].mean().item()}
        for b_idx in range(batch_size):
            state_b = states[b_idx]  # (N+1, d)
            log_pf = self.log_pf(time_b[:-1], state_b[:-1], state_b[1:])  # (N,)
            log_pb = self.log_pb(time_b[1:], state_b[1:], state_b[:-1])  # (N,)

            flow_b = self.flow(time_b, state_b).squeeze(-1)  # (N+1, 1) -> (N+1,)

            flow_b = flow_b + logrs[b_idx]  # (N+1,)
            flow_b[-1] = logrs[b_idx][-1]  # (1,)

            diff_logp = log_pf - log_pb  # (N, )
            diff_logp_padded = torch.cat(
                (torch.zeros(1).to(diff_logp), diff_logp.cumsum(dim=-1)), dim=0
            )  # (N+1,)
            # this means A1[i, j] = diff_logp[i:j].sum(dim=-1)
            A1 = diff_logp_padded.unsqueeze(0) - diff_logp_padded.unsqueeze(
                1
            )  # (N+1, N+1)

            A2 = flow_b[:, None] - flow_b[None, :] + A1  # (N+1, N+1)
            if not torch.all(torch.isfinite(A2)):
                import ipdb

                ipdb.set_trace()

            A2 = (A2 / self.data_ndim).pow(2)  # (N+1, N+1)
            # torch.triu() is useless here
            loss = torch.triu(A2 * self.coef, diagonal=1).sum()
            info["loss_dlogp"] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        info["loss_train"] = loss.item()
        return info


class SubTrajectoryBalanceTrajectoryBased(DetailedBalance):
    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)

        self.Lambda = float(cfg.subtb_lambda)
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N))  # (T+1, T+1)
        self.register_buffer("coef", coef, persistent=False)

    def get_flow_logp_from_traj(self, traj, debug=False):
        batch_size = traj[0][1].shape[0]
        xs = [x.to(self.device) for (t, x, r) in traj]
        ts = [
            t[None].to(self.device).repeat(batch_size, 1) for (t, x, r) in traj
        ]  # slightly faster

        state = torch.cat(xs[:-1], dim=0)  # (T*b, d)
        next_state = torch.cat(xs[1:], dim=0)  # (T*b, d)
        time = torch.cat(ts[:-1], dim=0)  # (T*b, 1)
        next_time = torch.cat(ts[1:], dim=0)  # (T*b, 1)
        log_pf = self.log_pf(time, state, next_state)
        log_pb = self.log_pb(next_time, next_state, state)
        log_pf = rearrange(log_pf, "(T b) -> b T", b=batch_size)
        log_pb = rearrange(log_pb, "(T b) -> b T", b=batch_size)

        states = torch.cat(xs, dim=0)  # ((T+1)*b, d)
        times = torch.cat(ts, dim=0)  # ((T+1)*b, 1)
        flows = self.flow(times, states).squeeze(-1)  # ((T+1)*b, 1) -> ((T+1)*b,)
        flows = rearrange(flows, "(T1 b) -> b T1", b=batch_size)  # (b, T+1)

        rs = [r.to(self.device) for (t, x, r) in traj]
        logrs = torch.cat(rs, dim=0)  # ((T+1)*b, 1)
        logrs = rearrange(logrs, "(T1 b) -> b T1", b=batch_size)  # (b, T+1)
        flows = flows + logrs  # (b, T+1)

        logr_terminal = self.logr_from_traj(traj)  # (b,)
        flows[:, -1] = logr_terminal

        logits_dict = {"log_pf": log_pf, "log_pb": log_pb, "flows": flows}
        return logits_dict

    def train_loss(self, traj):
        batch_size = traj[0][1].shape[0]
        logits_dict = self.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = (
            logits_dict["flows"],
            logits_dict["log_pf"],
            logits_dict["log_pb"],
        )
        diff_logp = log_pf - log_pb  # (b, T)

        diff_logp_padded = torch.cat(
            (torch.zeros(batch_size, 1).to(diff_logp), diff_logp.cumsum(dim=-1)), dim=1
        )
        # this means A1[:, i, j] = diff_logp[:, i:j].sum(dim=-1)
        A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(
            2
        )  # (b, T+1, T+1)

        A2 = flows[:, :, None] - flows[:, None, :] + A1  # (b, T+1, T+1)
        A2 = (A2 / self.data_ndim).pow(2).mean(dim=0)  # (T+1, T+1)
        # torch.triu() is useless here
        loss = torch.triu(A2 * self.coef, diagonal=1).sum()
        info = {"loss_dlogp": loss.item()}

        logZ_model = self.flow(
            traj[0][0][None, None].to(self.device), traj[0][1][:1, :].to(self.device)
        ).detach()
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

```

# gflownet/main.py

```py
import sys, os
import gzip, pickle
from collections import defaultdict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict, OmegaConf
import os.path as osp

import numpy as np
import torch

from gflownet.gflownet import get_alg, sample_traj
from gflownet.utils import seed_torch

torch.backends.cudnn.benchmark = True


def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
        cfg.work_directory = os.getcwd()
        cfg.gpu_type = (
            torch.cuda.get_device_name()
            if (torch.cuda.is_available() and cfg.device >= 0)
            else "CPU"
        )
        print(f"GPU type: {cfg.gpu_type}")

        cfg.task = cfg.target.dataset._target_.split(".")[-2]
        cfg.logr_min = cfg.rmin

        cfg.batch_size = cfg.bs
        cfg.weight_decay = cfg.wd
        cfg.sigma_interactive = cfg.sgmit
        if cfg.sigma_interactive <= 0:
            cfg.sigma_interactive = cfg.sigma
        cfg.t_end = cfg.dt * cfg.N
        cfg.subtb_lambda = cfg.stlam

        # Add checkpoint configuration
        cfg.checkpoint_dir = osp.join(cfg.work_directory, "checkpoints")
        cfg.checkpoint_freq = cfg.eval_freq  # Save at same frequency as eval
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    del cfg.d, cfg.bs, cfg.wd, cfg.sgmit, cfg.stlam, cfg.rmin
    return cfg


@hydra.main(config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg = refine_cfg(cfg)
    device = torch.device(
        f"cuda:{cfg.device:d}"
        if torch.cuda.is_available() and cfg.device >= 0
        else "cpu"
    )

    seed_torch(cfg.seed)
    print(f"Device: {device}, GPU type: {cfg.gpu_type}")
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    data = instantiate(cfg.target.dataset)
    true_logz = data.gt_logz()
    if true_logz is not None:
        print(f"True logZ={true_logz:.4f}")

    def logr_fn_detach(x):
        logr = -data.energy(x).detach()
        logr = torch.where(torch.isinf(logr), cfg.logr_min, logr)
        logr = torch.clamp(logr, min=cfg.logr_min)
        return logr

    gflownet = get_alg(cfg, data)
    gflownet.to(device)

    metric_best = 100.0
    best_model_path = None

    # Load checkpoint if exists
    latest_checkpoint = osp.join(cfg.checkpoint_dir, "latest.pt")
    if osp.exists(latest_checkpoint):
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        gflownet.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint["step"]
        metric_best = checkpoint["metric_best"]
        if "best_model_path" in checkpoint:
            best_model_path = checkpoint["best_model_path"]
        print(f"Resuming from step {start_step}")
    else:
        start_step = 0

    for step_idx in range(start_step, cfg.steps):
        ######### eval
        gflownet.eval()
        if step_idx % cfg.eval_freq == 0 or step_idx >= cfg.steps - 1:
            traj, eval_info = gflownet.eval_step(cfg.eval_n, logr_fn_detach)
            print(
                f"EVALUATION: step={step_idx}:",
                " ".join([f"{k}={v:.3f}" for k, v in eval_info.items()]),
            )

            if true_logz is not None:
                logz_diff = abs(eval_info["logz"] - true_logz)
                print(f"logZ diff={logz_diff:.3f}")
                if logz_diff < metric_best:
                    metric_best = logz_diff
                    print(f"best metric: {metric_best:.2f} at step {step_idx}")

                    # Save best model
                    best_model_path = osp.join(
                        cfg.checkpoint_dir, f"best_model_step_{step_idx}.pt"
                    )
                    torch.save(
                        {
                            "step": step_idx,
                            "model_state_dict": gflownet.state_dict(),
                            "metric_best": metric_best,
                            "config": OmegaConf.to_container(cfg, resolve=True),
                            "eval_info": eval_info,
                        },
                        best_model_path,
                    )

        ######### rollout
        traj_batch, _ = sample_traj(
            gflownet,
            cfg,
            logr_fn_detach,
            batch_size=cfg.batch_size,
            sigma=cfg.sigma_interactive,
        )

        ######### training
        gflownet.train()
        gflownet.train_step(traj_batch)

        # Save checkpoint periodically
        if step_idx % cfg.checkpoint_freq == 0:
            checkpoint_path = osp.join(cfg.checkpoint_dir, "latest.pt")
            torch.save(
                {
                    "step": step_idx + 1,  # Save next step to resume from
                    "model_state_dict": gflownet.state_dict(),
                    "metric_best": metric_best,
                    "best_model_path": best_model_path,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                checkpoint_path,
            )

    # Save final model
    final_model_path = osp.join(cfg.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "step": cfg.steps,
            "model_state_dict": gflownet.state_dict(),
            "metric_best": metric_best,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "eval_info": eval_info,
        },
        final_model_path,
    )


if __name__ == "__main__":
    main()

```

# gflownet/molecular_agent.py

```py
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

```

# gflownet/molecular_gfn.py

```py
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

```

# gflownet/network.py

```py
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from einops import rearrange


def check_shape(cur_shape):
    if isinstance(cur_shape, Iterable):
        return tuple(cur_shape)
    elif isinstance(cur_shape, int):
        return tuple([cur_shape,])
    else:
        raise NotImplementedError(f"Type {type(cur_shape)} not support")

class IdentityOne:
    def __call__(self, t, y):
        del t
        return torch.ones_like(y)

class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(self, in_shape, out_shape, num_layers=2, channels=128,
                 zero_init=True, res=False):
        super().__init__()
        self.in_shape = check_shape(in_shape) # 2 -> (2,)
        self.out_shape = check_shape(out_shape)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
        )
        self.final_layer = nn.Linear(channels, int(np.prod(self.out_shape)))
        if zero_init:
            self.final_layer.weight.data.fill_(0.0)
            self.final_layer.bias.data.fill_(0.0)

        self.residual = res

    # cond: (1,) or (1, 1) or (bs, 1); inputs: (bs, d)
    # output: (bs, d_out)
    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            # (1, channels) * (bs, 1) + (1, channels)
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        ) # (bs, 2* channels) -> (bs, channels)
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1)) # (bs, d) -> (bs, channels)

        input = embed_ins + embed_cond
        out = self.layers(input)
        if self.residual:
            out = out + input
        out = self.final_layer(out) # (bs, channels) -> (bs, d)
        return out.view(-1, *self.out_shape)
```

# gflownet/tasks/__init__.py

```py

```

# gflownet/tasks/base.py

```py
import torch
from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Base class for all tasks in GFlowNet.
    
    This abstract class defines the interface that all tasks must implement
    for use with the GFlowNet training pipeline.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def energy(self, x):
        """Compute energy for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Energy values
        """
        pass

    @abstractmethod
    def score(self, x):
        """Compute score (gradient of negative energy) for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Score values (forces)
        """
        pass

    @abstractmethod
    def log_reward(self, x):
        """Compute log reward for given coordinates.
        
        Args:
            x (torch.Tensor): Input coordinates
            
        Returns:
            torch.Tensor: Log reward values
        """
        pass

    def get_state(self):
        """Get current state information.
        
        Returns:
            tuple: Current positions and forces
        """
        return None, None

    def setup_openmm_system(self):
        """Setup OpenMM system if needed.
        
        This is optional for tasks that don't use OpenMM.
        """
        pass 
```

# gflownet/tasks/molecular_base.py

```py
import numpy as np
import openmm.unit as unit
from abc import abstractmethod
from .base import BaseTask
import torch


class BaseMolecularDynamics(BaseTask):
    """Base class for molecular dynamics tasks in GFlowNet"""

    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.temperature * unit.kelvin
        self.friction = cfg.friction / unit.femtoseconds
        self.timestep = cfg.timestep * unit.femtoseconds

        # Setup system and get initial state
        self.pdb, self.integrator, self.simulation, self.external_force = self.setup()
        self.get_md_info()
        self.simulation.minimizeEnergy()
        self.position = self.get_state()[0]

    @abstractmethod
    def setup(self):
        """Setup OpenMM system - to be implemented by specific forcefields"""
        pass

    def get_md_info(self):
        """Get system information like masses and thermal noise parameters"""
        self.num_particles = self.simulation.system.getNumParticles()
        m = np.array(
            [
                self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(self.num_particles)
            ]
        )
        self.heavy_atoms = m > 1.1
        self.m = unit.Quantity(m, unit.dalton)
        # ... rest of get_md_info from BaseDynamics

    def step(self, forces):
        """Perform one MD step with given forces"""
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def get_state(self):
        """Get current positions and forces"""
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)
        forces = state.getForces().value_in_unit(
            unit.dalton * unit.nanometer / unit.femtosecond**2
        )
        return positions, forces

    def reset(self):
        """Reset simulation to initial state"""
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def set_temperature(self, temperature):
        """Update system temperature"""
        self.integrator.setTemperature(temperature * unit.kelvin)

    # Implement BaseTask abstract methods
    def energy(self, x):
        """Compute potential energy for given coordinates"""
        _, potentials = self.energy_function(x)
        return potentials

    def score(self, x):
        """Compute forces for given coordinates"""
        forces, _ = self.energy_function(x)
        return forces

    def log_reward(self, x):
        """Compute log reward as negative energy"""
        return -self.energy(x)

    def energy_function(self, positions):
        """Compute both forces and potential energy

        Args:
            positions: torch.Tensor of shape (batch_size, num_particles * 3)

        Returns:
            tuple: (forces, potentials) as numpy arrays
        """
        forces, potentials = [], []
        for pos in positions:
            # Convert PyTorch tensor to numpy array
            pos_np = pos.detach().cpu().numpy()
            # Reshape from (num_particles * 3,) to (num_particles, 3)
            pos_np = pos_np.reshape(-1, 3)
            # Create OpenMM-compatible positions with units
            pos_openmm = unit.Quantity(pos_np, unit.nanometer)

            # Set positions and compute state
            self.simulation.context.setPositions(pos_openmm)
            state = self.simulation.context.getState(getForces=True, getEnergy=True)

            # Get forces and potential energy
            force = state.getForces().value_in_unit(
                unit.dalton * unit.nanometer / unit.femtosecond**2
            )
            potential = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules / unit.mole
            )

            forces.append(
                np.array(force).flatten()
            )  # Convert to numpy array and flatten
            potentials.append(potential)

        # Stack and convert to torch tensors
        forces = torch.tensor(forces, dtype=positions.dtype, device=positions.device)
        potentials = torch.tensor(
            potentials, dtype=positions.dtype, device=positions.device
        )

        # Ensure potentials have shape (batch_size,)
        potentials = potentials.reshape(-1).squeeze(-1)

        # check if potentials is of shape (batch_size,)
        # print("potentials.shape:", potentials.shape)

        return forces, potentials

    def get_start_state(self, batch_size):
        """Get initial state replicated batch_size times.

        Args:
            batch_size (int): Number of copies to create

        Returns:
            torch.Tensor: Initial positions tensor of shape (batch_size, num_particles * 3)
        """
        # Convert OpenMM Vec3 positions to numpy array
        pos_np = np.array([[v.x, v.y, v.z] for v in self.position])
        # Convert to torch tensor and flatten
        pos_flat = torch.tensor(pos_np, dtype=torch.float32).reshape(
            -1
        )  # flatten to (num_particles * 3)
        # Repeat for batch size
        return pos_flat.repeat(batch_size, 1)  # Shape: (batch_size, num_particles * 3)

```

# gflownet/tasks/molecular_mds.py

```py
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

```

# gflownet/tasks/molecular.py

```py
from .molecular_base import BaseMolecularDynamics
import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator


class MolecularTask(BaseMolecularDynamics):
    """
    MolecularTask is a subclass of BaseMolecularDynamics that sets up the
    OpenMM system for a specific molecular dynamics simulation.
    """

    def __init__(self, cfg):
        # Store cfg before calling super().__init__
        self.cfg = cfg
        # These need to be set before calling setup() via super().__init__
        self.molecule = cfg.molecule
        self.start_state = cfg.start_state
        self.end_state = cfg.end_state if hasattr(cfg, "end_state") else None
        self.temperature = cfg.temperature
        self.friction = cfg.friction
        self.timestep = cfg.timestep
        # Now call super().__init__ which will use setup()
        super().__init__(cfg)

    def setup(self):
        """Setup OpenMM system with specific forcefield"""
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(f"data/{self.molecule}/{self.start_state}.pdb")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )

        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)

        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        return pdb, integrator, simulation, external_force

```

# gflownet/train_molecular.py

```py
import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from gflownet.molecular_agent import MolecularGFlowNetAgent
from gflownet.utils import loss2ess_info, setup_logging
from gflownet.tasks.molecular import MolecularTask
from gflownet.tasks.molecular_mds import MolecularMDs

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="aldp", type=str)

# Logger Config
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--eval_interval", default=1000, type=int)
parser.add_argument("--save_interval", default=5000, type=int)

# GFlowNet Config
parser.add_argument("--subtb_lambda", default=0.9, type=float)
parser.add_argument("--t_end", default=1.0, type=float)
parser.add_argument("--dt", default=0.01, type=float)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--xclip", default=10.0, type=float)
parser.add_argument("--nn_clip", default=100.0, type=float)
parser.add_argument("--lgv_clip", default=100.0, type=float)
parser.add_argument(
    "--task", default="molecular", type=str
)  # molecular task for this project

# Molecular Config
parser.add_argument("--start_state", default="c5", type=str)
# parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)

# Training Config
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
# parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)

# Sampling Config
parser.add_argument("--timestep", default=1, type=float)


def main():
    args = parser.parse_args()

    # First create a temporary task to get the system dimensions
    temp_cfg = OmegaConf.create(
        {
            "molecule": args.molecule,
            "start_state": args.start_state,
            "temperature": args.temperature,
            "friction": args.friction,
            "timestep": args.timestep,
        }
    )
    temp_task = MolecularTask(temp_cfg)

    # Now we can get the proper dimension for the full config
    # Multiply by 3 for x,y,z coordinates of each particle
    data_ndim = temp_task.num_particles * 3

    # Convert args to OmegaConf for compatibility with GFlowNet
    cfg = OmegaConf.create(
        {
            # System dimensions
            "data_ndim": data_ndim,
            # GFlowNet parameters
            "seed": args.seed,
            "device": args.device,
            "subtb_lambda": args.subtb_lambda,
            "t_end": args.t_end,
            "dt": args.dt,
            "sigma": args.sigma,
            "weight_decay": args.weight_decay,
            "xclip": args.xclip,
            "nn_clip": args.nn_clip,
            "lgv_clip": args.lgv_clip,
            "N": int(args.t_end / args.dt),
            "batch_size": args.batch_size,
            "task": args.task,
            # Network configuration (add these based on GFlowNet requirements)
            "f_func": {
                "_target_": "gflownet.network.FourierMLP",
                "num_layers": 4,
                "channels": 128,
                "in_shape": (data_ndim,),
                "out_shape": (data_ndim,),
            },
            "g_func": {
                "_target_": "gflownet.network.IdentityOne",
            },
            "f": "tgrad",  # Use temperature-scaled gradient for molecular systems
            "lr": args.learning_rate,
            "zlr": args.learning_rate * 0.1,  # Typically lower learning rate for flow
            # Molecular system parameters
            "molecule": args.molecule,
            "start_state": args.start_state,
            "temperature": args.temperature,
            "friction": args.friction,
            "timestep": args.timestep,
        }
    )

    # Setup directories and logging
    save_dir = os.path.join(args.save_dir, args.molecule)
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(save_dir)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"System dimension (num_particles * 3): {data_ndim}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize MDs first
    mds = MolecularMDs(cfg, cfg.batch_size)
    logger.info(f"System dimension (num_particles * 3): {mds.num_particles * 3}")

    # Initialize agent with MDs
    agent = MolecularGFlowNetAgent(cfg, mds)

    # Setup temperature annealing
    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_epochs
    )

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_ess = 0

    for epoch in range(args.num_epochs):
        current_temp = temperatures[epoch]
        logger.info(f"Epoch {epoch}, Temperature: {current_temp:.2f}")

        # Sample trajectories using MDs
        traj, sample_info = agent.sample(args.batch_size, mds, current_temp)

        # Training step
        loss_info = agent.train(traj)
        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            logger.info(
                f"Step {global_step}: "
                f"Loss: {loss_info['loss_train']:.4f}, "
                f"LogZ: {loss_info['logz_model']:.4f}"
            )

        # Evaluation
        if global_step % args.eval_interval == 0:
            agent.gfn.eval()
            with torch.no_grad():
                print("Mds num samples:", mds.num_samples)
                eval_traj, eval_info = agent.sample(args.batch_size, mds, current_temp)
                ess_info = loss2ess_info(eval_info)
                logger.info(
                    f"Eval Step {global_step}: "
                    f"ESS: {ess_info['ess']:.4f}, "
                    f"ESS%: {ess_info['ess_percent']:.2f}%"
                )

                # Save best model
                if ess_info["ess"] > best_ess:
                    best_ess = ess_info["ess"]
                    torch.save(
                        agent.gfn.state_dict(), os.path.join(save_dir, "best_model.pt")
                    )

        # Regular model saving
        if global_step % args.save_interval == 0:
            torch.save(
                agent.gfn.state_dict(),
                os.path.join(save_dir, f"model_step_{global_step}.pt"),
            )

    logger.info("Training completed!")

    # Save final model
    torch.save(agent.gfn.state_dict(), os.path.join(save_dir, "final_model.pt"))


if __name__ == "__main__":
    main()

```

# gflownet/utils.py

```py
import os, sys
from itertools import count
import pathlib
import functools
import socket
import logging
from datetime import datetime

import math
import random
import scipy
import numpy as np
import torch


######### System Utils


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


######### Pytorch Utils

import random


def seed_torch(seed, verbose=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))


def normal_logp(x, mean=0.0, sigma=1.0):
    """
    Calculate the log probability of a normal distribution.

    Args:
        x (torch.Tensor): The state, shape: [b, d]
        mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
        sigma (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.

    Returns:
        torch.Tensor: The log probability, shape: [b, 1]
    """
    assert x.ndim == 2
    if isinstance(sigma, torch.Tensor):
        assert sigma.ndim == 2
        log_sigma = torch.log(sigma)
    else:
        log_sigma = np.log(sigma)

    neg_logp = 0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * (x - mean) ** 2 / (sigma**2)
    return torch.sum(-neg_logp, dim=-1, keepdim=False)  # Return shape [b]


def loss2ess_info(loss):
    # ESS = (\sum w_i)^2 / \sum w_i^2
    # return ESS / N <= 1
    log_weight = -loss + loss.mean()
    log_numerator = 2 * torch.logsumexp(log_weight, dim=0)
    log_denominator = torch.logsumexp(2 * log_weight, dim=0)
    ess = torch.exp(log_numerator - log_denominator) / len(log_weight)
    return {"ess": ess.item()}


def setup_logging(save_dir):
    """
    Sets up logging configuration to output to both console and file.

    Args:
        save_dir (str): Directory where the log file will be saved

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("gflownet")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

```

# README.md

```md
# CV-Free Conformation Sampling based on DGFS

This repository is a modified version of [Diffusion Generative Flow Samplers (DGFS)](https://arxiv.org/abs/2310.02679), adapted for CV-Free Conformation Sampling.

Original DGFS work by [Dinghuai Zhang](https://zdhnarsil.github.io/), 
[Ricky Tian Qi Chen](https://rtqichen.github.io//),
[Cheng-Hao Liu](https://pchliu.github.io/), 
Aaron Courville, 
[Yoshua Bengio](https://yoshuabengio.org/).

We build upon DGFS, which proposes a novel sampler for continuous space sampling from given unnormalized densities based on stochastic optimal control 🤖 formulation and the probabilistic 🎲 GFlowNet framework.

<a href="https://imgse.com/i/pPOmv7T"><img src="https://z1.ax1x.com/2023/10/03/pPOmv7T.md.png" alt="pPOmv7T.png" border="0" /></a>

## Project Structure
- `target/`: Contains the target distribution code
- `gflownet/`: Contains the modified DGFS algorithm code for CV-Free sampling

## Examples

\`\`\`bash
python -m gflownet.main target=gm dt=0.05
python -m gflownet.main target=funnel
python -m gflownet.main target=wells
\`\`\`

## Checkpoints and Model Saving

The training process automatically saves checkpoints in the `checkpoints/` directory within your working directory. Three types of checkpoints are maintained:

1. **Latest Checkpoint** (`latest.pt`):
   - Saved periodically every `checkpoint_freq` steps
   - Used for resuming training from the last saved state
   - Contains model state, training step, and best metrics

2. **Best Model** (`best_model_step_{step}.pt`):
   - Saved when a new best logZ difference is achieved
   - Contains model state and evaluation metrics

3. **Final Model** (`final_model.pt`):
   - Saved at the end of training
   - Contains the final model state and training results

Training will automatically resume from the latest checkpoint if it exists in the working directory.

## Dependencies

Apart from commonly used torch, torchvision, numpy, scipy, matplotlib,
we use the following packages:
\`\`\`bash
pip install hydra-core omegaconf submitit hydra-submitit-launcher
pip install wandb tqdm einops seaborn ipdb
\`\`\`

## Configuration

Key training parameters in `configs/main.yaml`:
- `steps`: Total training steps
- `eval_freq`: Frequency of evaluation and checkpoint saving
- `eval_n`: Number of samples for estimating logZ
- `device`: GPU device ID (use -1 for CPU)

## Acknowledgments

This work builds upon the [DGFS paper](https://arxiv.org/abs/2310.02679). Please cite their work if you use this code:

\`\`\`bibtex
@article{zhang2023diffusion,
  title={Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization},
  author={Zhang, Dinghuai and Chen, Ricky TQ and Liu, Cheng-Hao and Courville, Aaron and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2310.02679},
  year={2023}
}
\`\`\`

```

# setup.py

```py
from setuptools import setup, find_packages

setup(
    name="gflownet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "omegaconf",
        "openmm",
        
    ]
)
```

# viz_gm_results.py

```py
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from gflownet.gflownet import get_alg
from gflownet.main import refine_cfg


@hydra.main(config_path="gflownet/configs", config_name="main")
def visualize_results(cfg: DictConfig) -> None:
    # Override config with GM target and dt=0.05
    # cfg.target = "gm.yaml"
    cfg.dt = 0.05

    # Refine config as done in training
    cfg = refine_cfg(cfg)

    # Setup device
    device = torch.device(
        f"cuda:{cfg.device:d}"
        if torch.cuda.is_available() and cfg.device >= 0
        else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize target distribution and GFlowNet
    data = instantiate(cfg.target.dataset)
    gflownet = get_alg(cfg, data)
    gflownet.to(device)

    # Load the best model if it exists
    # checkpoint_dir = os.path.join(cfg.work_directory, "checkpoints")
    checkpoint_dir = (
        "/hpc/dctrl/jy384/DGFS-CVCS/outputs/2025-02-04/14-55-14/checkpoints"
    )
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_model")]

    if model_files:
        # Load the latest best model
        latest_best = sorted(model_files)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_best)
        print(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        gflownet.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]
        print(f"Model from step {step} loaded successfully")
    else:
        print("No best model checkpoint found")
        return

    # Generate samples for visualization
    print("Generating samples for visualization...")
    gflownet.eval()
    with torch.no_grad():
        # Generate 15000 samples for visualization
        traj, eval_info = gflownet.eval_step(15000, data.energy)
        samples = traj[-1][1]  # Get final states from trajectories

        # Create visualization directory
        viz_dir = os.path.join(cfg.work_directory, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Generate visualizations
        print("Creating visualizations...")
        img_dict = gflownet.visualize(traj, data.energy, step=step)

        print("Visualization files created:")
        for viz_type, filepath in img_dict.items():
            full_path = os.path.join(viz_dir, filepath)
            # Move the file if it's not already in the visualization directory
            if os.path.dirname(filepath) != viz_dir:
                os.rename(filepath, full_path)
            print(f"- {viz_type}: {full_path}")


if __name__ == "__main__":
    visualize_results()

```

