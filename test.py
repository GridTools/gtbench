import numpy as np
import sys
import math
import matplotlib.pyplot as plt

import scipy.sparse as scipy_sparse


def tridiagonal_solve(a, b, c, d):
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


def horizontal_diffusion(data):
    lap = laplacian(data[1:-1, 1:-1, :])

    flx_x = lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]
    flx_x *= flx_x[:, :, :] * (data[3:-2, 3:-3, :] - data[2:-3, 3:-3, :]) <= 0

    flx_y = lap[1:-1, 1:, :] - lap[1:-1, :-1, :]
    flx_y *= flx_y[:, :, :] * (data[3:-3, 3:-2, :] - data[3:-3, 2:-3, :]) <= 0

    return data[3:-3, 3:-3, :] - 0.25 * (
        flx_x[1:, :, :] - flx_x[:-1, :, :] + flx_y[:, 1:, :] - flx_y[:, :-1, :]
    )


def advection_flux_u(u, data, dx):
    weights = [1.0 / 30, -1.0 / 4, 1, -1.0 / 3, -1.0 / 2, 0]
    weights[-1] = -sum(weights[:-1])

    negative_mask = u[3:-3, 3:-3, :] < 0
    positive_mask = u[3:-3, 3:-3, :] > 0

    #  return u[3:-3, 3:-3, :] * (
    #  negative_mask * (data[4:-2, 3:-3, :] - data[3:-3, 3:-3, :])
    #  + positive_mask * (data[3:-3, 3:-3, :] - data[2:-4, 3:-3, :])
    #  )
    return -u[3:-3, 3:-3, :] * (
        negative_mask
        * -(
            weights[0] * data[:-6, 3:-3, :]
            + weights[1] * data[1:-5, 3:-3, :]
            + weights[2] * data[2:-4, 3:-3, :]
            + weights[3] * data[3:-3, 3:-3, :]
            + weights[4] * data[4:-2, 3:-3, :]
            + weights[5] * data[5:-1, 3:-3, :]
        )
        / dx
        + positive_mask
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


def advection_w_column(w, data, dt, dx):
    assert len(w) == len(data) + 1
    a = np.zeros(data.shape)
    b = np.zeros(data.shape)
    c = np.zeros(data.shape)
    d = np.zeros(data.shape)
    # assume zero wind outside...
    a[0] = 0
    c[0] = 0.25 * w[0] / dx
    b[0] = 1 / dt - a[0] - c[0]
    d[0] = (
        1 / dt * data[0]
        - 0.25 * w[0] * (data[0] - 0) / dx
        - 0.25 * w[0 + 1] * (data[0 + 1] - data[0]) / dx
    )
    for k in range(1, len(data) - 1):
        a[k] = -0.25 * w[k] / dx
        c[k] = 0.25 * w[k + 1] / dx
        b[k] = 1 / dt - a[k] - c[k]
        d[k] = (
            1 / dt * data[k]
            # - 0.25 * w[k] * data[k + 1] / dx
            # + 0.25 * w[k] * data[k - 1] / dx
            - 0.25 * w[k + 1] * (data[k + 1] - data[k]) / dx
            - 0.25 * w[k] * (data[k] - data[k - 1]) / dx
        )
    a[-1] = -0.25 * w[-2] / dx
    c[-1] = 0
    b[-1] = 1 / dt - a[-1] - c[-1]
    d[-1] = (
        1 / dt * data[-1]
        - 0.25 * w[-1] * (0 - data[-1]) / dx
        - 0.25 * w[-2] * (data[-1] - data[-2]) / dx
    )
    return tridiagonal_solve(a, b, c, d)


def advection_flux_w(w, data, dt, dx):
    advected = np.zeros_like(data[3:-3, 3:-3, :])
    for j in range(3, data.shape[1] - 3):
        for k in range(data.shape[2]):
            advected[:, j - 3, k] = advection_w_column(
                w[3:-3, j, k], data[3:-3, j, k], dt, dx
            )

    return (advected - data[3:-3, 3:-3, :]) / dt


def diffusion_w_column(data, dt, dx):
    a = np.zeros(data.shape)
    b = np.zeros(data.shape)
    c = np.zeros(data.shape)
    d = np.zeros(data.shape)
    D = 0.1
    # assume zero wind, and zero data outside...
    a[0] = 0
    c[0] = -D / 2 * dt
    b[0] = 1 / dt - a[0] - c[0]
    d[0] = (
        1 / dt * data[0]
        - 0.25 * (data[0] - 0) / dx
        - 0.25 * (data[0 + 1] - data[0]) / dx
    )
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
    d[-1] = (
        1 / dt * data[-1]
        - 0.25 * (0 - data[-1]) / dx
        - 0.25 * (data[-1] - data[-2]) / dx
    )
    return tridiagonal_solve(a, b, c, d)


def diffusion_flux_w(w, data, dt, dx):
    diffused = np.zeros_like(data[3:-3, 3:-3, :])
    for j in range(3, data.shape[1] - 3):
        for k in range(data.shape[2]):
            diffused[:, j - 3, k] = diffusion_w_column(data[3:-3, j, k], dt, dx)

    return (diffused - data[3:-3, 3:-3, :]) / dt


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
        self.xs = np.linspace(0, 1, self.compute_domain[0], False)
        self.ys = np.linspace(0, 1, self.compute_domain[1], False)
        self.zs = np.linspace(0, 1, self.compute_domain[2], False)

        self.dx = self.xs[1] - self.xs[0]
        self.dt = self.dx

        self.data = np.fromfunction(
            lambda i, j, k: (i > 10) * (i < 20) * 1,
            #  lambda i, j, k: np.sin(2 * math.pi / self.compute_domain[0] * i),
            self.compute_domain,
            dtype=np.float32,
        )
        self.u = np.ones(self.compute_domain, dtype=np.float32) * 0.5
        self.v = np.ones(self.compute_domain, dtype=np.float32) * 0.5
        # w is staggered
        self.w = (
            np.ones(
                list(map(sum, zip(self.compute_domain, [1, 0, 0]))), dtype=np.float32
            )
            * 0.5
        )
        self.u = self.w

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
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=1, linewidth=200)

        y1 = add_boundary(
            self.data[3:-3, 3:-3, :]
            + self.dt / 3 * diffusion_flux_w(self.u, self.data, self.dx, self.dt / 3),
            self.boundaries,
        )
        y2 = add_boundary(
            self.data[3:-3, 3:-3, :]
            + self.dt / 2 * diffusion_flux_w(self.u, y1, self.dx, self.dt / 2),
            self.boundaries,
        )
        self.data = add_boundary(
            self.data[3:-3, 3:-3, :]
            + self.dt * diffusion_flux_w(self.u, y2, self.dx, self.dt),
            self.boundaries,
        )
        #  print(np.concatenate((scipy_sparse.diags([a[1:], b, c[:-1]], [-1, 0, 1]).toarray(), [[x] for x in d]),
        print(
            sum(
                np.concatenate(
                    (y1[3:-3, 3, :], y2[3:-3, 3, :], self.data[3:-3, 3, :]), axis=1
                )
            )
        )

    def step(self):
        self.timestep += 1
        print(self.timestep)
        self.advect()

        periodic_boundary_condition(self.data, self.boundaries)

        #  self.data[3:-3, 3:-3, :] = horizontal_diffusion(self.data)

        periodic_boundary_condition(self.data, self.boundaries)

    def get_global_domain(self):
        return calculate_global_domain(self.compute_domain, self.boundaries)

    global_domain = property(get_global_domain)

    def save_img(self):
        #  plt.contourf(b.u[:, :, 0], levels=100)
        #  plt.colorbar()
        u = 0.5
        expected = np.fromfunction(
            lambda i, j, k: np.sin(
                (i - self.dt * self.timestep / self.dx * u)
                / self.compute_domain[0]
                * math.pi
                * 2
            ),
            self.compute_domain,
            dtype=np.float32,
        )
        plt.plot(self.xs, b.data[3:-3, 3, 0], lw=3)
        plt.plot(self.xs, expected[:, 0, 0])
        plt.savefig("u" + str(self.timestep) + ".png")

        plt.clf()


b = Benchmark([50, 1, 1])
b.save_img()
for i in range(80):
    for i in range(1):
        b.step()
    b.save_img()
