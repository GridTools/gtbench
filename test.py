import numpy as np
import sys
import math
import matplotlib.pyplot as plt

import scipy.sparse as scipy_sparse

from advdiff import (calculate_global_domain, add_boundary, compute_domain_slice, step)


class Benchmark:
    def __init__(self, compute_domain):
        self.compute_domain = compute_domain
        self.boundaries = [(3, 3), (3, 3), (0, 0)]
        self.xs = np.arange(self.compute_domain[0])
        self.ys = np.arange(self.compute_domain[1])
        self.zs = np.arange(self.compute_domain[2])

        self.dx = 2 * np.pi / self.compute_domain[0]
        self.dy = 2 * np.pi / self.compute_domain[1]
        self.dz = 2 * np.pi / self.compute_domain[2]
        self.dt = 1e-2
        self.timestep = 0
        self.D = 1e-1

        self.data = np.fromfunction(
            lambda i, j, k: (
                np.sin(2 * math.pi / self.compute_domain[0] * i)
                * np.sin(math.pi / self.compute_domain[1] * j)
                * (k <= 8)
                * (k >= 3)
            ),
            self.compute_domain,
            dtype=np.float32,
        )
        self.u = np.ones(self.compute_domain, dtype=np.float32) * 0.5
        self.u = np.full(self.compute_domain, 1, dtype=np.float32)
        self.v = np.ones(self.compute_domain, dtype=np.float32) * 0.5
        self.v = np.full(self.compute_domain, -1, dtype=np.float32)
        # w is staggered
        self.w = np.fromfunction(
            lambda i, j, k: (
                (
                    k
                    / (self.compute_domain[2] + 1)
                    * (1 - k / (self.compute_domain[2] + 1))
                    * 4
                )
                ** 4
                * np.sin(2 * math.pi / self.compute_domain[0] * i)
                * np.sin(2 * math.pi / self.compute_domain[1] * j)
            ),
            map(sum, zip(self.compute_domain, [0, 0, 1])),
            dtype=np.float32,
        )
        self.w = (
            np.ones(
                list(map(sum, zip(self.compute_domain, [0, 0, 1]))), dtype=np.float32
            )
            * 0.5
        )
        self.w = np.full(tuple(map(sum, zip(self.compute_domain, (0, 0, 1)))), 1, dtype=np.float32)
        self.data = np.fromfunction(
            lambda i, j, k: (i > 3)
            * (i < 8)
            * (j > 3)
            * (j < 8)
            * (k > 13)
            * (k < 18)
            * 1,
            self.compute_domain,
            dtype=np.float32,
        )
        self.data = self.exact_solution()
        #  plt.pcolor(self.w[5, :, :].transpose())
        #  plt.colorbar()
        #  plt.figure()
        #  plt.pcolor(self.data[5, :, :].transpose())
        #  plt.colorbar()
        #  plt.show()

        self.u = add_boundary(self.u, self.boundaries)
        self.v = add_boundary(self.v, self.boundaries)
        self.w = add_boundary(self.w, self.boundaries)
        self.data = add_boundary(self.data, self.boundaries)

    def exact_solution(self):
        i, j, k = np.indices(self.compute_domain)
        x, y, z, t = i * self.dx, j * self.dy, k * self.dz, self.timestep * self.dt
        return (np.sin(x - t) * np.sin(y + t) * np.cos(z - t) * np.exp(-t * self.D * 3)).astype(np.float32)

    def step(self):
        self.timestep += 1
        print(self.timestep)

        self.data = step(self.data, self.u, self.v, self.w, self.D, self.dx, self.dy, self.dz, self.dt, self.boundaries)

        print('data L2 norm: {:12.6g}'.format(np.sqrt(np.mean(self.data[self.cds]**2))))
        print('data max norm: {:12.6g}'.format(np.amax(np.abs(self.data[self.cds]))))

        error = self.data[self.cds] - self.exact_solution()
        print('error L2 norm: {:12.6g}'.format(np.sqrt(np.mean(error**2))))
        print('error max norm: {:12.6g}'.format(np.amax(np.abs(error))))

    @property
    def global_domain(self):
        return calculate_global_domain(self.compute_domain, self.boundaries)

    @property
    def cds(self):
        return compute_domain_slice(self.boundaries)

    def save_img(self):
        #  plt.contourf(self.u[:, :, 0], levels=100)
        #  plt.colorbar()
        u = self.u[0, 0, 0]
        v = self.v[0, 0, 0]
        w = self.w[0, 0, 0]
        x_from = 3 + int(self.dt * self.timestep / self.dx * u) % self.compute_domain[0]
        x_to = 5 + x_from + 1
        y_from = 3 + int(self.dt * self.timestep / self.dy * v) % self.compute_domain[1]
        y_to = 5 + y_from + 1
        z_from = 13 + int(self.dt * self.timestep / self.dz * w)
        z_to = 5 + z_from + 1
        expected = np.fromfunction(
            lambda i, j, k: (
                (
                    (i > x_from) * (i < x_to)
                    + (i > x_from + self.compute_domain[0])
                    * (i < x_to + self.compute_domain[0])
                    + (i > x_from - self.compute_domain[0])
                    * (i < x_to - self.compute_domain[0])
                )
                * (
                    (j > y_from) * (j < y_to)
                    + (j > y_from + self.compute_domain[1])
                    * (j < y_to + self.compute_domain[1])
                    + (j > y_from - self.compute_domain[1])
                    * (j < y_to - self.compute_domain[1])
                )
                * (
                    (k > z_from) * (k < z_to)
                    + (k > z_from - self.compute_domain[2])
                    * (k < z_to - self.compute_domain[2])
                )
            ),
            self.compute_domain,
            dtype=np.float32,
        )
        plot = 4
        if plot == 1:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.pcolor(self.data[3:-3, 3:-3, 5].transpose(), vmin=0, vmax=1)
            ax2.pcolor(expected[:, :, 0].transpose(), vmin=0, vmax=1)
            ax3.pcolor(
                self.data[3:-3, 3:-3, 0].transpose() - expected[:, :, 0].transpose(),
                vmin=0,
                vmax=1,
            )
            plt.savefig("u" + str(self.timestep) + ".png")
            plt.close(f)
        if plot == 2:
            f, ax1 = plt.subplots(1, 1)
            ax1.plot(
                self.zs,
                self.data[15, 5, :],
                lw=3,
                color="#dddddd",
                markerfacecolor="r",
                ls="-",
                marker=".",
            )
            ax1.plot(self.zs, expected[15, 5, :])
            plt.savefig("u" + str(self.timestep) + ".png")
            #  plt.show()
            plt.close(f)
        if plot == 3:
            #  ax1.pcolor(self.data[3:-3, 3:-3, 5].transpose(), vmin=0, vmax=1)
            f, (ax1, ax2) = plt.subplots(1, 2)
            position_x = (
                int(5.5 + self.dt * self.timestep / self.dx * self.u[0, 0, 0])
            ) % (self.compute_domain[0])
            position_y = (
                int(5.5 + self.dt * self.timestep / self.dy * self.v[0, 0, 0])
            ) % (self.compute_domain[1])

            ax1.set_title("[" + str(position_x) + ", :, :]")
            ax1.pcolor(
                self.data[position_x + 3, 3:-3, :].transpose(),
                vmin=-1,
                vmax=1,
                cmap="seismic",
            )
            ax1.contour(
                expected[position_x, :, :].transpose(), levels=[0.99], corner_mask=True
            )

            ax2.set_title("[:, " + str(position_y) + ", :]")
            ax2.pcolor(
                self.data[3:-3, position_y + 3, :].transpose(),
                vmin=-1,
                vmax=1,
                cmap="seismic",
            )
            ax2.contour(
                expected[:, position_y, :].transpose(), levels=[0.99], corner_mask=True
            )
            plt.savefig("u" + str(self.timestep) + ".png")
            #  plt.show()
            plt.close("all")
        if plot == 4:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            position_y = self.compute_domain[1] // 4
            data = self.data[3:-3, 3 + position_y, :]
            exact = self.exact_solution()[:, position_y, :]
            f.colorbar(ax1.contourf(data, vmin=-1, vmax=1, levels=np.linspace(-1, 1, 11), cmap='seismic'), ax=ax1)
            f.colorbar(ax2.contourf(exact, vmin=-1, vmax=1, levels=np.linspace(-1, 1, 11), cmap='seismic'), ax=ax2)
            f.colorbar(ax3.contourf(np.abs(data - exact), cmap='seismic'), ax=ax3)
            plt.savefig("u" + str(self.timestep) + ".png")
            plt.close("all")


b = Benchmark([16, 16, 60])
b.save_img()
sums_abs = []
sums = []
for i in range(15):
    for i in range(10):
        b.step()
    sums_abs.append(np.sum(np.abs((b.data[3:-3, 3:-3, :]))))
    sums.append(np.sum((b.data[3:-3, 3:-3, :])))

    b.save_img()

plt.plot(sums)
plt.plot(sums_abs)
plt.savefig("numbers.png")
plt.show()
