import torch
import joblib
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import pyemma.coordinates as coor

from .utils import *


class Plot:
    def __init__(self, args, mds):
        self.device = args.device
        self.save_dir = args.save_dir
        self.molecule = args.molecule
        self.start_state = args.start_state
        self.num_samples = args.num_samples
        self.start_position = mds.start_position
        self.target_position = mds.target_position
        self.energy_function = mds.energy_function

    def __call__(self):
        positions, potentials = [], []
        for i in range(self.num_samples):
            position = np.load(f"{self.save_dir}/positions/{i}.npy").astype(np.float32)
            potential = self.energy_function(position)[1]
            positions.append(torch.from_numpy(position).to(self.device))
            potentials.append(potential)

        self.paths(positions)

    def paths(self, positions):
        zorder = 32
        circle_size = 1200
        saddle_size = 2400

        if self.molecule == "aldp":
            angle_1 = [6, 8, 14, 16]
            angle_2 = [1, 6, 8, 14]

            plt.clf()
            plt.close()
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            plt.xlim([-np.pi, np.pi])
            plt.ylim([-np.pi, np.pi])

            with open("./data/aldp/landscape.dat") as f:
                lines = f.readlines()

            dims = [90, 90]

            locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
            data = torch.zeros((int(dims[0]), int(dims[1])))

            i = 0
            for line in lines[1:]:
                splits = line[0:-1].split(" ")
                vals = [y for y in splits if y != ""]

                x = float(vals[0])
                y = float(vals[1])
                val = float(vals[-1])

                locations[i // 90, i % 90, :] = torch.tensor([x, y])
                data[i // 90, i % 90] = val
                i = i + 1

            xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
            ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
            x, y = np.meshgrid(xs, ys)
            inp = torch.tensor(np.array([x, y])).view(2, -1).T

            loc = locations.view(-1, 2)
            distances = torch.cdist(inp, loc.double(), p=2)
            index = distances.argmin(dim=1)

            a = torch.div(index, locations.shape[0], rounding_mode="trunc")
            b = index % locations.shape[0]

            z = data[a, b]
            z = z.view(y.shape[0], y.shape[1])

            plt.contourf(xs, ys, z, levels=100, zorder=0)

            cm = plt.get_cmap("gist_rainbow")
            ax.set_prop_cycle(
                color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
            )

            for position in positions:
                psi = compute_dihedral(position[:, angle_1, :]).detach().cpu().numpy()
                phi = compute_dihedral(position[:, angle_2, :]).detach().cpu().numpy()

                ax.plot(
                    phi,
                    psi,
                    marker="o",
                    linestyle="None",
                    markersize=2,
                    alpha=1.0,
                )

            start_psi = (
                compute_dihedral(self.start_position[:, angle_1, :])
                .detach()
                .cpu()
                .numpy()
            )
            start_phi = (
                compute_dihedral(self.start_position[:, angle_2, :])
                .detach()
                .cpu()
                .numpy()
            )

            target_psi = (
                compute_dihedral(self.target_position[:, angle_1, :])
                .detach()
                .cpu()
                .numpy()
            )
            target_phi = (
                compute_dihedral(self.target_position[:, angle_2, :])
                .detach()
                .cpu()
                .numpy()
            )

            phis_saddle = [-0.035, -0.017]
            psis_saddle = [1.605, -0.535]

            ax.scatter(
                phis_saddle,
                psis_saddle,
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=saddle_size,
                marker="*",
            )

            ax.scatter(
                start_phi,
                start_psi,
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=circle_size,
            )

            ax.scatter(
                target_phi,
                target_psi,
                edgecolors="w",
                c="w",
                zorder=zorder,
                s=circle_size,
            )

            plt.xlabel("\u03A6", fontsize=35, fontweight="medium")
            plt.ylabel("\u03A8", fontsize=35, fontweight="medium")
        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)

            cm = plt.get_cmap("gist_rainbow")
            ax.set_prop_cycle(
                color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
            )

            pmf = np.load(f"./data/{self.molecule}/pmf.npy")
            xs = np.load(f"./data/{self.molecule}/xs.npy")
            ys = np.load(f"./data/{self.molecule}/ys.npy")
            plt.pcolormesh(xs, ys, pmf.T, cmap="viridis")

            tica_model = joblib.load(f"./data/{self.molecule}/tica_model.pkl")
            feat = coor.featurizer(f"./data/{self.molecule}/{self.start_state}.pdb")
            feat.add_backbone_torsions(cossin=True)

            for position in positions:
                traj = md.Trajectory(
                    position.cpu().numpy(),
                    md.load(f"./data/{self.molecule}/{self.start_state}.pdb").topology,
                )
                feature = feat.transform(traj)
                tica = tica_model.transform(feature)
                ax.plot(
                    tica[:, 0],
                    tica[:, 1],
                    marker="o",
                    linestyle="None",
                    markersize=2,
                    alpha=1.0,
                )

            start_position = md.Trajectory(
                self.start_position.cpu().numpy(),
                md.load(f"./data/{self.molecule}/{self.start_state}.pdb").topology,
            )
            feature = feat.transform(start_position)
            start_tica = tica_model.transform(feature)
            ax.scatter(
                start_tica[:, 0],
                start_tica[:, 1],
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=circle_size,
            )

            target_position = md.Trajectory(
                self.target_position.cpu().numpy(),
                md.load(f"./data/{self.molecule}/{self.start_state}.pdb").topology,
            )
            feature = feat.transform(target_position)
            target_tica = tica_model.transform(feature)
            ax.scatter(
                target_tica[:, 0],
                target_tica[:, 1],
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=circle_size,
            )
            plt.xlabel("TIC 1", fontsize=35, fontweight="medium")
            plt.ylabel("TIC 2", fontsize=35, fontweight="medium")

            plt.xlim(xs.min(), xs.max())
            plt.ylim(ys.min(), ys.max())

        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/paths.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        return fig
