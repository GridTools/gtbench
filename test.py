import numpy as np
import sys
import math
import matplotlib.pyplot as plt

import scipy.sparse as scipy_sparse


def tridiagonal_solve(a, b, c, d):
    d = d.copy()
    c[0] /= b[0]
    d[0] /= b[0]
    for k in range(1, len(a) - 1):
        div = b[k] - c[k - 1] * a[k]
        c[k] /= div
        d[k] = (d[k] - d[k - 1] * a[k]) / div
    d[-1] = (d[-1] - d[-2] * a[-1]) / (b[-1] - c[-2] * a[-1])

    for k in range(len(a) - 2, -1, -1):
        d[k] = d[k] - c[k] * d[k + 1]
    return d


assert np.all(
    np.rint(
        tridiagonal_solve(
            np.array([0, 1, 1, 7, 6, 3, 8, 6, 5, 4], dtype=np.float32),
            np.array([2, 3, 3, 2, 2, 4, 1, 2, 4, 5], dtype=np.float32),
            np.array([1, 2, 1, 6, 1, 3, 5, 7, 3, 0], dtype=np.float32),
            np.array([1, 2, 6, 34, 10, 1, 4, 22, 25, 3], dtype=np.float32),
        )
    )
    == np.array([1, -1, 2, 1, 3, -2, 0, 4, 2, -1], dtype=np.float32)
)


def laplacian(data):
    return (
        -4 * data[1:-1, 1:-1, :]
        + data[:-2, 1:-1, :]
        + data[2:, 1:-1, :]
        + data[1:-1, :-2, :]
        + data[1:-1, 2:, :]
    ) * 0.25


def horizontal_diffusion_fancy(data):
    K = 0.1
    lap = laplacian(data[1:-1, 1:-1, :])

    flx_x = lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]
    flx_x *= flx_x[:, :, :] * (data[3:-2, 3:-3, :] - data[2:-3, 3:-3, :]) < 0

    flx_y = lap[1:-1, 1:, :] - lap[1:-1, :-1, :]
    flx_y *= flx_y[:, :, :] * (data[3:-3, 3:-2, :] - data[3:-3, 2:-3, :]) < 0

    return data[3:-3, 3:-3, :] - K * (
        flx_x[1:, :, :] - flx_x[:-1, :, :] + flx_y[:, 1:, :] - flx_y[:, :-1, :]
    )


def horizontal_diffusion(data, dx, dy, dt):
    K = 0.1
    flx_x = (data[3:-2, 3:-3, :] - data[2:-3, 3:-3, :]) / dx
    flx_y = (data[3:-3, 3:-2, :] - data[3:-3, 2:-3, :]) / dy
    return data[3:-3, 3:-3, :] + K * dt * (
            (flx_x[1:, :, :] - flx_x[:-1, :, :]) / dx +
            (flx_y[:, 1:, :] - flx_y[:, :-1, :]) / dy)


def advection_flux_v(v, data0, data, dy):
    weights = [1.0 / 30, -1.0 / 4, 1, -1.0 / 3, -1.0 / 2, 0]
    weights[-1] = -sum(weights[:-1])

    negative_mask = v[3:-3, 3:-3, :] < 0
    positive_mask = v[3:-3, 3:-3, :] > 0

    return -v[3:-3, 3:-3, :] * (
        positive_mask
        * -(
            weights[0] * data[3:-3, :-6, :]
            + weights[1] * data[3:-3, 1:-5, :]
            + weights[2] * data[3:-3, 2:-4, :]
            + weights[3] * data[3:-3, 3:-3, :]
            + weights[4] * data[3:-3, 4:-2, :]
            + weights[5] * data[3:-3, 5:-1, :]
        )
        / dy
        + negative_mask
        * (
            weights[5] * data[3:-3, 1:-5, :]
            + weights[4] * data[3:-3, 2:-4, :]
            + weights[3] * data[3:-3, 3:-3, :]
            + weights[2] * data[3:-3, 4:-2, :]
            + weights[1] * data[3:-3, 5:-1, :]
            + weights[0] * data[3:-3, 6:, :]
        )
        / dy
    )


def advection_flux_u(u, data0, data, dx):
    weights = [1.0 / 30, -1.0 / 4, 1, -1.0 / 3, -1.0 / 2, 0]
    weights[-1] = -sum(weights[:-1])

    negative_mask = u[3:-3, 3:-3, :] < 0
    positive_mask = u[3:-3, 3:-3, :] > 0

    return -u[3:-3, 3:-3, :] * (
        positive_mask
        * -(
            weights[0] * data[:-6, 3:-3, :]
            + weights[1] * data[1:-5, 3:-3, :]
            + weights[2] * data[2:-4, 3:-3, :]
            + weights[3] * data[3:-3, 3:-3, :]
            + weights[4] * data[4:-2, 3:-3, :]
            + weights[5] * data[5:-1, 3:-3, :]
        )
        / dx
        + negative_mask
        * (
            weights[5] * data[1:-5, 3:-3, :]
            + weights[4] * data[2:-4, 3:-3, :]
            + weights[3] * data[3:-3, 3:-3, :]
            + weights[2] * data[4:-2, 3:-3, :]
            + weights[1] * data[5:-1, 3:-3, :]
            + weights[0] * data[6:, 3:-3, :]
        )
        / dx
    )


def advection_w_column(w, data0, data, dz, dt):
    assert len(w) == len(data) + 1
    a = np.zeros(data.shape)
    b = np.zeros(data.shape)
    c = np.zeros(data.shape)
    d = np.zeros(data.shape)
    # assume zero wind outside...
    a[0] = -0.25 * w[0] / dz
    c[0] = 0.25 * w[1] / dz
    b[0] = 1 / dt - a[0] - c[0]
    d[0] = (
        1 / dt * data[0]
        - 0.25 * w[0] * (data[0] - 0) / dz
        - 0.25 * w[0 + 1] * (data[0 + 1] - data[0]) / dz
    )
    for k in range(1, len(data) - 1):
        a[k] = -0.25 * w[k] / dz
        c[k] = 0.25 * w[k + 1] / dz
        b[k] = 1 / dt - a[k] - c[k]
        d[k] = (
            1 / dt * data[k]
            - 0.25 * w[k + 1] * (data[k + 1] - data[k]) / dz
            - 0.25 * w[k] * (data[k] - data[k - 1]) / dz
        )
    a[-1] = -0.25 * w[-2] / dz
    c[-1] = 0.25 * w[-1] / dz
    b[-1] = 1 / dt - a[-1] - c[-1]
    d[-1] = (
        1 / dt * data[-1]
        - 0.25 * w[-1] * (0 - data[-1]) / dz
        - 0.25 * w[-2] * (data[-1] - data[-2]) / dz
    )
    return tridiagonal_solve(a, b, c, d)


def advection_flux_w(w, data0, data, dz, dt):
    advected = np.zeros_like(data[3:-3, 3:-3, :])
    for i in range(3, data.shape[0] - 3):
        for j in range(3, data.shape[1] - 3):
            advected[i - 3, j - 3, :] = advection_w_column(
                w[i, j, :], data0[i, j, :], data[i, j, :], dz, dt
            )

    return (advected - data[3:-3, 3:-3, :]) / dt


def diffusion_w_column(data, dx, dt):
    a = np.zeros(data.shape)
    b = np.zeros(data.shape)
    c = np.zeros(data.shape)
    d = np.zeros(data.shape)
    D = 0.3
    # assume zero wind, and zero data outside...
    a[0] = 0
    c[0] = -D / 2 * dt
    b[0] = 1 / dt - a[0] - c[0]
    d[0] = 1 / dt * data[0] + 0.5 * D * (data[1] - 2 * data[0] + 0) / dx
    for k in range(1, len(data) - 1):
        a[k] = -D / 2 * dt
        c[k] = -D / 2 * dt
        b[k] = 1 / dt - a[k] - c[k]
        d[k] = (
            1 / dt * data[k] + 0.5 * D * (data[k + 1] - 2 * data[k] + data[k - 1]) / dx
        )
    a[-1] = -D / 2 * dt
    c[-1] = 0
    b[-1] = 1 / dt - a[-1] - c[-1]
    d[-1] = 1 / dt * data[-1] + 0.5 * D * (0 - 2 * data[-1] + data[-2]) / dx
    return tridiagonal_solve(a, b, c, d)


def diffusion_flux_w(w, data, dx, dt):
    diffused = np.zeros_like(data[3:-3, 3:-3, :])
    for i in range(3, data.shape[0] - 3):
        for j in range(3, data.shape[1] - 3):
            diffused[i - 3, j - 3, :] = diffusion_w_column(data[i, j, :], dx, dt)

    return (diffused - data[3:-3, 3:-3, :]) / dt


def advection_flux(u, v, w, data0, data, dx, dy, dz, dt):
    return (
        advection_flux_u(u, data0, data, dx)
        + advection_flux_v(v, data0, data, dy)
        + advection_flux_w(w, data0, data, dz, dt)
        #  + diffusion_flux_w(w, data, dx, dt)
    )


def calculate_global_domain(compute_domain, boundaries):
    return [sz + sum(b) for (sz, b) in zip(compute_domain, boundaries)]


def periodic_boundary_condition(data, boundaries):
    ((x1, x2), (y1, y2), (z1, z2)) = boundaries
    assert x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0 and z1 == 0 and z2 == 0

    # edges
    data[:x1, y1:-y2, :] = data[-x1 - x2 : -x2, y1:-y2, :]
    data[-x2:, y1:-y2, :] = data[x1 : x1 + x2, y1:-y2, :]
    data[x1:-x2, :y1, :] = data[x1:-x2, -y1 - y2 : -y2, :]
    data[x1:-x2, -y2:, :] = data[x1:-x2, y1 : y1 + y2, :]

    # corners
    data[:x1, :y1, :] = data[-x1 - x2 : -x2, -y1 - y1 : -y2, :]
    data[-x2:, -y2:, :] = data[x1 : x1 + x2, y1 : y1 + y2, :]
    data[:x1, -y2:, :] = data[-x1 - x2 : -x2, y1 : y1 + y2, :]
    data[-x2:, :y1, :] = data[x1 : y1 + x2, -y1 - y2 : -y2, :]


def add_boundary(data, boundaries):
    ((x1, x2), (y1, y2), (z1, z2)) = boundaries
    assert x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0 and z1 == 0 and z2 == 0
    global_domain = calculate_global_domain(data.shape, boundaries)
    new_data = np.zeros(global_domain)
    new_data[x1:-x2, y1:-y2, :] = data

    periodic_boundary_condition(new_data, boundaries)
    return new_data


class Benchmark:
    def __init__(self, compute_domain):
        self.compute_domain = compute_domain
        self.boundaries = [(3, 3), (3, 3), (0, 0)]
        self.xs = np.arange(self.compute_domain[0])
        self.ys = np.arange(self.compute_domain[1])
        self.zs = np.arange(self.compute_domain[2])

        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.dt = 1

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
        self.v = np.ones(self.compute_domain, dtype=np.float32) * 0.5
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

        self.timestep = 0

    def advect(self):
        # irk_order=3, irunge_kutta=1
        # it is irk_order=3, but not third order... Wicker, Skamarock (2002)
        # y' = y^n + 1/3 * dt * f(t^n, y)
        # y'' = y^n + 1/2 * dt * f(t^n + 1/3 dt, y')
        # y^{n+1} = y^n + dt * f(t^n + 1/2 dt, y'')

        diff_flux = diffusion_flux_w(self.w, self.data, self.dz, self.dt)

        flux = diff_flux + advection_flux(
            self.u,
            self.v,
            self.w,
            self.data,
            self.data,
            self.dx,
            self.dy,
            self.dz,
            self.dt,
        )
        y1 = add_boundary(
            self.data[3:-3, 3:-3, :] + self.dt / 3 * flux, self.boundaries
        )
        flux = diff_flux + advection_flux(
            self.u, self.v, self.w, self.data, y1, self.dx, self.dy, self.dz, self.dt
        )
        y2 = add_boundary(
            self.data[3:-3, 3:-3, :] + self.dt / 2 * flux, self.boundaries
        )
        flux = diff_flux + advection_flux(
            self.u, self.v, self.w, self.data, y2, self.dx, self.dy, self.dz, self.dt
        )
        self.data = add_boundary(
            self.data[3:-3, 3:-3, :] + self.dt * flux, self.boundaries
        )
        print(
            np.sum(
                np.stack(
                    (y1[3:-3, 3:-3, :], y2[3:-3, 3:-3, :], self.data[3:-3, 3:-3, :])
                ),
                axis=(1, 2, 3),
            ),
            np.sum(
                np.abs(
                    np.stack(
                        (y1[3:-3, 3:-3, :], y2[3:-3, 3:-3, :], self.data[3:-3, 3:-3, :])
                    )
                ),
                axis=(1, 2, 3),
            ),
        )

    def step(self):
        self.timestep += 1
        print(self.timestep)
        self.advect()

        periodic_boundary_condition(self.data, self.boundaries)

        self.data[3:-3, 3:-3, :] = horizontal_diffusion(self.data, self.dx, self.dy, self.dt)
        periodic_boundary_condition(self.data, self.boundaries)

    def get_global_domain(self):
        return calculate_global_domain(self.compute_domain, self.boundaries)

    global_domain = property(get_global_domain)

    def save_img(self):
        #  plt.contourf(b.u[:, :, 0], levels=100)
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
                * (k > z_from)
                * (k < z_to)
            ),
            self.compute_domain,
            dtype=np.float32,
        )
        plot = 3
        if plot == 1:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.pcolor(b.data[3:-3, 3:-3, 5].transpose(), vmin=0, vmax=1)
            ax2.pcolor(expected[:, :, 0].transpose(), vmin=0, vmax=1)
            ax3.pcolor(
                b.data[3:-3, 3:-3, 0].transpose() - expected[:, :, 0].transpose(),
                vmin=0,
                vmax=1,
            )
            plt.savefig("u" + str(self.timestep) + ".png")
            plt.close(f)
        if plot == 2:
            f, ax1 = plt.subplots(1, 1)
            ax1.plot(
                self.zs,
                b.data[15, 5, :],
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
            #  ax1.pcolor(b.data[3:-3, 3:-3, 5].transpose(), vmin=0, vmax=1)
            f, (ax1, ax2) = plt.subplots(1, 2)
            position_x = (
                int(5.5 + self.dt * self.timestep / self.dx * self.u[0, 0, 0])
            ) % (self.compute_domain[0])
            position_y = (
                int(5.5 + self.dt * self.timestep / self.dy * self.v[0, 0, 0])
            ) % (self.compute_domain[1])

            ax1.set_title("[" + str(position_x) + ", :, :]")
            ax1.pcolor(
                b.data[position_x + 3, 3:-3, :].transpose(),
                vmin=-1,
                vmax=1,
                cmap="seismic",
            )
            ax1.contour(
                expected[position_x, :, :].transpose(), levels=[0.99], corner_mask=True
            )

            ax2.set_title("[:, " + str(position_y) + ", :]")
            ax2.pcolor(
                b.data[3:-3, position_y + 3, :].transpose(),
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


b = Benchmark([16, 16, 60])
b.save_img()
sums_abs = []
sums = []
for i in range(150):
    for i in range(1):
        b.step()
    sums_abs.append(np.sum(np.abs((b.data[3:-3, 3:-3, :]))))
    sums.append(np.sum((b.data[3:-3, 3:-3, :])))

    b.save_img()

plt.plot(sums)
plt.plot(sums_abs)
plt.savefig("numbers.png")
plt.show()
