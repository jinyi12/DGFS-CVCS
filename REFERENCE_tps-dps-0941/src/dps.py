import torch
import numpy as np
from tqdm import tqdm
from utils.utils import *
from bias import BiasForce


class DiffusionPathSampler:
    def __init__(self, args, mds):
        self.policy = BiasForce(args, mds)
        self.target_measure = TargetMeasure(args, mds)

        if args.training:
            self.replay = ReplayBuffer(args, mds)

    def sample(self, args, mds, temperature):
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        forces = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )

        position, force = mds.report()

        positions[:, 0] = position
        forces[:, 0] = force

        mds.reset()
        mds.set_temperature(temperature)
        for s in tqdm(range(1, args.num_steps + 1), desc="Sampling"):
            bias = (
                self.policy(position.detach(), mds.target_position).squeeze().detach()
            )
            mds.step(bias)

            position, force = mds.report()

            positions[:, s] = position
            forces[:, s] = force - 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
        mds.reset()

        log_tm, final_idx = self.target_measure(positions, forces)

        if args.training:
            self.replay.add((positions, forces, log_tm))

        for i in range(args.num_samples):
            np.save(
                f"{args.save_dir}/positions/{i}.npy",
                positions[i][: final_idx[i] + 1].cpu().numpy(),
            )

    def train(self, args, mds):
        optimizer = torch.optim.Adam(
            [
                {"params": [self.policy.log_z], "lr": args.log_z_lr},
                {"params": self.policy.mlp.parameters(), "lr": args.policy_lr},
            ]
        )

        loss_sum = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):

            positions, forces, log_tm = self.replay.sample()

            velocities = (positions[:, 1:] - positions[:, :-1]) / args.timestep

            biases = 1e-6 * self.policy(
                positions.view(-1, positions.size(-2), positions.size(-1)),
                mds.target_position,
            )
            biases = biases.view(*positions.shape)

            means = (
                1 - args.friction * args.timestep
            ) * velocities + args.timestep / mds.m * (forces[:, :-1] + biases[:, :-1])

            log_bpm = mds.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2, 3))

            # Our implementation is based on results in appendix A.2
            log_z = self.policy.log_z
            loss = (log_z + log_bpm - log_tm).square().mean()
            loss.backward()

            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()
        loss = loss_sum / args.trains_per_rollout
        return loss


class ReplayBuffer:
    def __init__(self, args, mds):
        self.positions = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.forces = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.log_tm = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.buffer_size = args.buffer_size

    def add(self, data):
        indices = torch.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        (
            self.positions[indices],
            self.forces[indices],
            self.log_tm[indices],
        ) = data

    def sample(self):
        indices = torch.randint(0, min(self.idx, self.buffer_size), (self.batch_size,))
        return (
            self.positions[indices],
            self.forces[indices],
            self.log_tm[indices],
        )


class TargetMeasure:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.timestep = args.timestep
        self.friction = args.friction
        self.heavy_atoms = mds.heavy_atoms
        self.target_position = mds.target_position

        self.m = mds.m
        self.log_prob = mds.log_prob

    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri, final_idx = self.relaxed_indicator(positions, self.target_position)

        log_tm = log_upm + log_ri
        return log_tm, final_idx

    def unbiased_path_measure(self, positions, forces):
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        means = (
            1 - self.friction * self.timestep
        ) * velocities + self.timestep / self.m * forces[:, :-1]
        log_upm = self.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2, 3))
        return log_upm

    def relaxed_indicator(self, positions, target_position):
        positions = positions[:, :, self.heavy_atoms]
        target_position = target_position[:, self.heavy_atoms]
        log_ri = torch.zeros(positions.size(0), device=positions.device)
        final_idx = torch.zeros(
            positions.size(0), device=positions.device, dtype=torch.long
        )
        for i in range(positions.size(0)):
            log_ri[i], final_idx[i] = self.rbf(
                positions[i],
                target_position,
            ).max(0)
        return log_ri, final_idx

    def rbf(self, positions, target_position):
        R, t = kabsch(positions, target_position)
        positions = torch.matmul(positions, R.transpose(-2, -1)) + t
        log_ri = (
            -0.5 / self.sigma**2 * (positions - target_position).square().mean((-2, -1))
        )
        return log_ri
