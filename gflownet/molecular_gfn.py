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


def sample_molecular_traj(
    gfn,
    config,
    logreward_fn,
    batch_size=None,
    sigma=None,
    potential_fn=None,
    temperature=None,
):
    """
    Sample a trajectory for molecular systems, starting from a defined molecular state.

    Args:
        gfn (MolecularGFlowNet): The GFlowNet to sample from.
        config (dict): The configuration.
        logreward_fn (callable): The log reward function.
        batch_size (int, optional): The batch size. Defaults to None.
        sigma (float, optional): The noise level. Defaults to None.
        potential_fn (callable, optional): Function to compute potential energy.
                                         If provided, used to compute rewards.
        temperature (float, optional): Temperature in Kelvin for Boltzmann weights.
                                     If None, uses config.temperature.

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
    if temperature is None:
        temperature = getattr(config, "temperature", 300)

    # Print sampling parameters for debugging
    print(f"Sampling with temperature={temperature}K, sigma={sigma}")

    # Start from the defined molecular state instead of zeros
    x = gfn.task.get_start_state(batch_size).to(device)  # shape: (b, d)

    # Debug: Check initial state
    if torch.isnan(x).any():
        print("WARNING: NaN detected in initial state!")
        x = torch.nan_to_num(x)

    # Debug: Print initial state stats
    print(f"Initial state min={x.min().item():.4f}, max={x.max().item():.4f}")

    # Compute initial reward, either using potential energy or default logr
    if potential_fn is not None:
        # Use potential energy to compute reward
        try:
            potential = potential_fn(x)
            # Check for extreme values in potential
            if torch.isnan(potential).any() or torch.isinf(potential).any():
                print(
                    f"WARNING: Potential energy calculation returned NaN/Inf: {potential}"
                )
                potential = torch.nan_to_num(
                    potential, nan=1e5, posinf=1e5, neginf=-1e5
                )

            # Print potential energy stats
            print(
                f"Initial potential energy: min={potential.min().item():.4f}, max={potential.max().item():.4f}, mean={potential.mean().item():.4f}"
            )

            # Use GFlowNet's method to compute Boltzmann reward
            fl_logr = gfn.get_boltzmann_reward(x, potential, temperature)
        except Exception as e:
            print(f"Error in potential_fn: {e}")
            # Fall back to default reward computation
            fl_logr = fl_inter_logr(x, logreward_fn, config, cur_t=0.0, sigma=sigma)
    else:
        # Use default log reward calculation
        fl_logr = fl_inter_logr(x, logreward_fn, config, cur_t=0.0, sigma=sigma)

    # Check rewards for NaN
    if torch.isnan(fl_logr).any():
        print("WARNING: NaN in initial rewards!")
        fl_logr = torch.nan_to_num(fl_logr, nan=-1e5)

    traj = [(torch.tensor(0.0), x.cpu(), fl_logr.cpu())]
    inter_loss = torch.zeros(batch_size).to(device)

    x_max = 0.0
    step_count = 0

    # Track state deltas for debugging
    last_x = x.clone()

    for cur_t in torch.arange(0, config.t_end, config.dt).to(device):
        step_count += 1

        # Use the original GFlowNet diffusion update for state transitions
        try:
            x, uw_term, u2_term = gfn.step_forward(cur_t, x, config.dt, sigma=sigma)
            x = x.detach()

            # Calculate state delta for debugging
            delta = (x - last_x).abs().mean().item()
            if step_count % 10 == 0:
                print(f"Step {step_count}: State delta={delta:.6f}")
            last_x = x.clone()

            # Check for NaN/Inf in states
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN/Inf detected in state at time {cur_t}!")
                x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)

            # Check for extreme values in states
            x_max_current = x.abs().max().item()
            if x_max_current > 1000:  # Arbitrary threshold for extremely large values
                print(f"WARNING: Extremely large state value detected: {x_max_current}")
                x = torch.clamp(x, min=-1000, max=1000)

        except RuntimeError as e:
            print(f"Runtime error in step_forward: {e}")
            # Try to recover by using the previous state
            x = last_x.clone()
            uw_term = torch.zeros(batch_size).to(device)
            u2_term = torch.zeros(batch_size).to(device)

        # Compute reward at new state
        if potential_fn is not None:
            # Use potential energy to compute reward
            try:
                potential = potential_fn(x)

                # Check potential energy for extreme values
                if torch.isnan(potential).any() or torch.isinf(potential).any():
                    print(f"WARNING: NaN/Inf in potential at time {cur_t}!")
                    potential = torch.nan_to_num(
                        potential, nan=1e5, posinf=1e5, neginf=-1e5
                    )

                # Periodically print potential energy stats
                if step_count % 10 == 0:
                    print(
                        f"Step {step_count} potential: min={potential.min().item():.4f}, max={potential.max().item():.4f}"
                    )

                # Use GFlowNet's method to compute Boltzmann reward
                fl_logr = gfn.get_boltzmann_reward(x, potential, temperature)
                fl_logr = fl_logr.detach().cpu()
            except Exception as e:
                print(f"Error computing potential at time {cur_t}: {e}")
                # Fall back to default reward
                fl_logr = (
                    fl_inter_logr(
                        x, logreward_fn, config, cur_t=cur_t + config.dt, sigma=sigma
                    )
                    .detach()
                    .cpu()
                )
        else:
            # Use default log reward calculation
            fl_logr = (
                fl_inter_logr(
                    x, logreward_fn, config, cur_t=cur_t + config.dt, sigma=sigma
                )
                .detach()
                .cpu()
            )

        # Check rewards for NaN
        if torch.isnan(fl_logr).any():
            print(f"WARNING: NaN detected in reward at time {cur_t}!")
            fl_logr = torch.nan_to_num(fl_logr, nan=-1e5)

        traj.append((cur_t.cpu() + config.dt, x.detach().cpu(), fl_logr))

        # Check for extreme values in loss terms
        if torch.isnan(uw_term).any() or torch.isinf(uw_term).any():
            print(f"WARNING: NaN/Inf in uw_term at time {cur_t}!")
            uw_term = torch.nan_to_num(uw_term)

        if torch.isnan(u2_term).any() or torch.isinf(u2_term).any():
            print(f"WARNING: NaN/Inf in u2_term at time {cur_t}!")
            u2_term = torch.nan_to_num(u2_term)

        inter_loss += (u2_term + uw_term).detach()
        x_max = max(x_max, x.abs().max().item())

    # Terminal reward calculation
    if potential_fn is not None:
        # Use potential energy to compute terminal reward
        try:
            potential = potential_fn(x)

            # Check for extreme values
            if torch.isnan(potential).any() or torch.isinf(potential).any():
                print("WARNING: NaN/Inf in terminal potential energy!")
                potential = torch.nan_to_num(
                    potential, nan=1e5, posinf=1e5, neginf=-1e5
                )

            print(
                f"Terminal potential energy: min={potential.min().item():.4f}, max={potential.max().item():.4f}"
            )

            # Use GFlowNet's Boltzmann reward for terminal reward
            pis_terminal = gfn.get_boltzmann_reward(x, potential, temperature)
        except Exception as e:
            print(f"Error in terminal reward calculation: {e}")
            # Fall back to regular terminal reward
            pis_terminal = -gfn.nll_prior(x) - logreward_fn(x)
    else:
        # Use regular terminal reward calculation
        pis_terminal = -gfn.nll_prior(x) - logreward_fn(x)

    # Check terminal reward for NaN
    if torch.isnan(pis_terminal).any() or torch.isinf(pis_terminal).any():
        print("WARNING: NaN/Inf in terminal reward!")
        pis_terminal = torch.nan_to_num(pis_terminal, nan=-1e5, posinf=1e5, neginf=-1e5)

    # Check inter_loss for NaN
    if torch.isnan(inter_loss).any() or torch.isinf(inter_loss).any():
        print("WARNING: NaN/Inf in inter_loss!")
        inter_loss = torch.nan_to_num(inter_loss)

    pis_log_weight = inter_loss + pis_terminal

    # Check final log weight for NaN
    if torch.isnan(pis_log_weight).any() or torch.isinf(pis_log_weight).any():
        print("WARNING: NaN/Inf in pis_log_weight!")
        pis_log_weight = torch.nan_to_num(
            pis_log_weight, nan=0.0, posinf=1e5, neginf=-1e5
        )

    # Print final log weight stats
    print(
        f"Final log weight: min={pis_log_weight.min().item():.4f}, max={pis_log_weight.max().item():.4f}, mean={pis_log_weight.mean().item():.4f}"
    )

    info = {
        "pis_logw": pis_log_weight,
        "x_max": x_max,
        "final_positions": x.detach().cpu(),
        "num_steps": step_count,
        "inter_loss_mean": inter_loss.mean().item(),
        "terminal_reward_mean": pis_terminal.mean().item(),
    }

    return traj, info


class MolecularGFlowNet(DetailedBalance):
    """
    MolecularGFlowNet is a modified GFlowNet algorithm tailored for
    sampling from molecular Boltzmann target densities. It supports
    both SubTrajectoryBalance loss and integration with TPS-DPS.
    """

    def __init__(self, cfg, task=None):
        super().__init__(cfg, task)
        self.Lambda = float(cfg.subtb_lambda)
        # Use cfg.N as the trajectory length for molecular simulations.
        coef = cal_subtb_coef_matrix(self.Lambda, int(cfg.N))
        self.register_buffer("coef", coef, persistent=False)

        # Store task for accessing methods like potential energy
        self.task = task

        # Store temperature for Boltzmann weights
        self.temperature = getattr(cfg, "temperature", 300)

        # Track loss history for diagnostics
        self._loss_history = []

        # Set flag to enable trajectory checking
        self._check_trajectories = True

    def set_temperature(self, temperature):
        """Update temperature for Boltzmann weights"""
        self.temperature = temperature

    def get_boltzmann_reward(self, x, potential=None, temperature=None):
        """
        Compute Boltzmann reward as -beta * U(x).

        Args:
            x: State tensor [batch_size, ndim]
            potential: Optional pre-computed potential energy
            temperature: Optional temperature override

        Returns:
            torch.Tensor: Log Boltzmann weights [batch_size]
        """
        temp = temperature if temperature is not None else self.temperature
        beta = 1.0 / (temp * 8.3144621e-3)  # 1/(kB*T) in mol/kJ

        # Use provided potential or compute it
        if potential is None and hasattr(self.task, "energy"):
            potential = self.task.energy(x)

        if potential is None:
            # Fall back to regular reward if no potential available
            return -self.nll_prior(x) - self.logr_fn(x)

        # Debug: Check potential energy values for extreme outliers
        if torch.isnan(potential).any() or torch.isinf(potential).any():
            print(f"WARNING: Found NaN or Inf in potential energy: {potential}")
            # Replace NaN/Inf with large but finite values
            potential = torch.nan_to_num(potential, nan=1e5, posinf=1e5, neginf=-1e5)

        # Debug: Print max/min potential values to check for extreme values
        with torch.no_grad():
            if torch.rand(1).item() < 0.01:  # Print occasionally to avoid flooding logs
                print(
                    f"Potential energy stats: min={potential.min().item():.4f}, max={potential.max().item():.4f}, mean={potential.mean().item():.4f}"
                )
                print(f"Beta={beta:.6f}, T={temp}K")

        # Clip extreme potential values to avoid numerical issues
        potential = torch.clamp(potential, min=-1e5, max=1e5)

        # Calculate Boltzmann weight with numerical stability
        log_boltzmann = -beta * potential

        # Debug: Check for NaN in rewards
        if torch.isnan(log_boltzmann).any():
            print(f"WARNING: NaN in Boltzmann weights: {log_boltzmann}")
            log_boltzmann = torch.nan_to_num(log_boltzmann, nan=-1e5)

        return log_boltzmann

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
        """
        batch_size = traj[0][1].shape[0]  # Get batch size from first trajectory element
        xs = [
            x.to(self.device) for (t, x, r) in traj
        ]  # States from trajectory, shape: [T, b, d]
        ts = [
            t[None].to(self.device).repeat(batch_size, 1) for (t, x, r) in traj
        ]  # Times

        state = torch.cat(xs[:-1], dim=0)  # All states except last (T*b, d)
        next_state = torch.cat(xs[1:], dim=0)  # All states except first (T*b, d)
        time = torch.cat(ts[:-1], dim=0)  # All times except last (T*b, 1)
        next_time = torch.cat(ts[1:], dim=0)  # All times except first (T*b, 1)

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
        flows[:, -1] = logr_terminal

        return {"log_pf": log_pf, "log_pb": log_pb, "flows": flows}

    def check_and_fix_trajectory(self, traj):
        """Check trajectory for extreme values and fix them.

        Args:
            traj: List of tuples (t, x, r) representing a trajectory

        Returns:
            Fixed trajectory
        """
        if not self._check_trajectories:
            return traj

        print("Checking trajectory for extreme values...")
        fixed_traj = []

        # Check for NaN/Inf in all tensors
        for i, (t, x, r) in enumerate(traj):
            has_issue = False

            # Fix states
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN/Inf detected in state at time {t}")
                x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
                has_issue = True

            # Fix rewards
            if torch.isnan(r).any() or torch.isinf(r).any():
                print(f"WARNING: NaN/Inf detected in reward at time {t}")
                r = torch.nan_to_num(r, nan=-1e3, posinf=1e3, neginf=-1e3)
                has_issue = True

            # Check for extreme values
            if x.abs().max().item() > 1000:
                print(f"WARNING: Extreme values in state: {x.abs().max().item()}")
                x = torch.clamp(x, min=-1000, max=1000)
                has_issue = True

            if r.abs().max().item() > 1e5:
                print(f"WARNING: Extreme values in reward: {r.abs().max().item()}")
                r = torch.clamp(r, min=-1e5, max=1e5)
                has_issue = True

            fixed_traj.append((t, x, r))

            if has_issue and i == 0:
                print("Issues in initial state/reward!")

        print(f"Trajectory check complete. Length: {len(fixed_traj)}")
        return fixed_traj

    def optimized_train_loss(self, traj):
        """
        Optimized version of train_loss with checks for numerical stability.
        This is a drop-in replacement for the standard train_loss method.

        Args:
            traj: List of tuples (t, x, r) representing a trajectory

        Returns:
            Tuple of (loss, info)
        """
        # First check and fix trajectory data
        traj = self.check_and_fix_trajectory(traj)

        # Get batch size
        batch_size = traj[0][1].shape[0]

        # Get flow probabilities
        logits_dict = self.get_flow_logp_from_traj(traj)
        flows, log_pf, log_pb = (
            logits_dict["flows"],
            logits_dict["log_pf"],
            logits_dict["log_pb"],
        )

        # Print tensor statistics
        if torch.rand(1).item() < 0.1:  # Only print occasionally
            print(
                f"Flow tensor stats - min: {flows.min().item():.4f}, max: {flows.max().item():.4f}"
            )
            print(
                f"log_pf tensor stats - min: {log_pf.min().item():.4f}, max: {log_pf.max().item():.4f}"
            )
            print(
                f"log_pb tensor stats - min: {log_pb.min().item():.4f}, max: {log_pb.max().item():.4f}"
            )

        # Check for NaN/Inf values
        for name, tensor in [("flows", flows), ("log_pf", log_pf), ("log_pb", log_pb)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"WARNING: {name} contains NaN or Inf values!")
                # Replace with zeros to avoid further issues
                if name == "flows":
                    flows = torch.nan_to_num(
                        flows, nan=0.0, posinf=100.0, neginf=-100.0
                    )
                elif name == "log_pf":
                    log_pf = torch.nan_to_num(log_pf, nan=0.0)
                elif name == "log_pb":
                    log_pb = torch.nan_to_num(log_pb, nan=0.0)

        # Calculate diff_logp with checks
        diff_logp = log_pf - log_pb

        # Check for extreme values in diff_logp
        if diff_logp.abs().max().item() > 1e5:
            print(
                f"WARNING: Extreme values in diff_logp: {diff_logp.abs().max().item()}"
            )
            diff_logp = torch.clamp(diff_logp, min=-1e5, max=1e5)

        # Compute padded diff_logp
        diff_logp_padded = torch.cat(
            (
                torch.zeros(batch_size, 1, device=diff_logp.device),
                diff_logp.cumsum(dim=-1),
            ),
            dim=1,
        )

        # Check for extreme values in diff_logp_padded
        if diff_logp_padded.abs().max().item() > 1e5:
            print(
                f"WARNING: Extreme values in diff_logp_padded: {diff_logp_padded.abs().max().item()}"
            )
            diff_logp_padded = torch.clamp(diff_logp_padded, min=-1e5, max=1e5)

        # Compute A1
        A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)

        # Check for extreme values in A1
        if A1.abs().max().item() > 1e6:
            print(f"WARNING: Extreme values in A1: {A1.abs().max().item()}")
            A1 = torch.clamp(A1, min=-1e6, max=1e6)

        # Compute A2
        A2 = flows[:, :, None] - flows[:, None, :] + A1

        # Check for extreme values in A2
        if A2.abs().max().item() > 1e8:
            print(f"WARNING: Extreme values in A2: {A2.abs().max().item()}")
            A2 = torch.clamp(A2, min=-1e8, max=1e8)

        # Normalize A2
        A2 = (A2 / self.data_ndim).pow(2).mean(dim=0)

        # Check coefficient matrix
        if torch.isnan(self.coef).any() or torch.isinf(self.coef).any():
            print("WARNING: coef matrix contains NaN/Inf values!")
            self.coef = torch.nan_to_num(self.coef)

        # Compute loss safely
        loss_matrix = A2 * self.coef

        # Check for extreme values in loss_matrix
        if loss_matrix.abs().max().item() > 1e10:
            print(
                f"WARNING: Extreme values in loss_matrix: {loss_matrix.abs().max().item()}"
            )
            loss_matrix = torch.clamp(loss_matrix, min=-1e10, max=1e10)

        # Compute final loss
        loss = torch.triu(loss_matrix, diagonal=1).sum()

        # Final safety check for loss
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e10:
            print(f"WARNING: Extremely large loss detected: {loss.item()}")
            # Cap the loss to prevent training instability
            loss = torch.clamp(loss, max=1e4)

        # Compute model logZ
        logZ_model = self.flow(
            traj[0][0][None, None].to(self.device), traj[0][1][:1, :].to(self.device)
        ).detach()

        # Store loss history
        self._loss_history.append(loss.item())
        if len(self._loss_history) > 20:
            self._loss_history.pop(0)

        # Create info dict
        info = {
            "loss_dlogp": loss.item(),
            "logz_model": logZ_model.mean().item(),
            "max_flow": flows.max().item(),
            "min_flow": flows.min().item(),
        }

        return loss, info

    def train_loss(self, traj):
        """Override the original train_loss with the optimized version"""
        # This makes it a drop-in replacement
        return self.optimized_train_loss(traj)

    def train_step(self, traj):
        """Override train_step to use molecular trajectory sampling and the optimized loss"""
        self.train()

        # Use optimized loss calculation
        loss, info = self.train_loss(traj)

        # Print loss and compare to history
        if len(self._loss_history) > 0:
            avg_loss = sum(self._loss_history) / len(self._loss_history)
            print(f"Current loss: {loss.item():.6f}, Average loss: {avg_loss:.6f}")

        # Use gradient clipping to prevent explosions
        self.optimizer.zero_grad()
        loss.backward()

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)

        self.optimizer.step()
        info["loss_train"] = loss.item()
        return info

    def eval_step(self, num_samples, logreward_fn=None):
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

    def save(self, path):
        """Save model state dictionary."""
        torch.save(self.state_dict(), path)
