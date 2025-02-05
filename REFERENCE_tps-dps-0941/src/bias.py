import torch
from torch import nn
from utils.utils import kabsch
from torch.nn.functional import softplus


class BiasForce(nn.Module):
    def __init__(self, args, mds):
        super().__init__()
        self.bias = args.bias

        self.heavy_atoms = mds.heavy_atoms
        self.num_particles = mds.num_particles
        if self.bias == "force":
            self.output_dim = mds.num_particles * 3
        elif self.bias == "pot":
            self.output_dim = 1
        elif self.bias == "scale":
            self.output_dim = mds.num_particles

        self.input_dim = mds.num_particles * (3 + 1)

        if args.molecule == "aldp":
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim),
            )
        elif args.molecule == "poly":
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_dim),
            )

        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.output_dim),
            )

        self.log_z = nn.Parameter(torch.tensor(0.0))

        self.to(args.device)

    def forward(self, pos, target):
        R, t = kabsch(pos[:, self.heavy_atoms], target[:, self.heavy_atoms])
        if self.bias == "pot":
            pos.requires_grad = True
        input = torch.matmul(pos, R.transpose(-2, -1)) + t

        dist = torch.norm(input - target, dim=-1, keepdim=True)
        input = torch.cat([input, dist], dim=-1)

        out = self.mlp(input.reshape(-1, self.input_dim))

        if self.bias == "force":
            force = out.view(*pos.shape)
            force = torch.matmul(force, R)
        elif self.bias == "pot":
            force = -torch.autograd.grad(out.sum(), pos, create_graph=True)[0]
        elif self.bias == "scale":
            target = torch.matmul(target - t, R)
            scale = softplus(out.view(*pos.shape[:-2], self.output_dim, 1))
            force = scale * (target - pos)
        return force
