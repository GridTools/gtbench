import numba
import numpy as np


@numba.jit
def tridiagonal_solve(a, b, c, d):
    d = d.copy()
    c = c.copy()
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
        )) == np.array([1, -1, 2, 1, 3, -2, 0, 4, 2, -1], dtype=np.float32))


# see http://www.phys.lsu.edu/classes/fall2013/phys7412/lecture10.pdf
@numba.jit
def tridiagonal_solve_periodic(a, b, c, d):
    alpha = c[-1]
    beta = a[0]
    gamma = -b[0]

    bb = b.copy()
    bb[0] = b[0] - gamma
    bb[-1] = b[-1] - alpha * beta / gamma
    x = tridiagonal_solve(a, bb, c, d)

    u = np.zeros_like(a)
    u[0] = gamma
    u[-1] = alpha
    z = tridiagonal_solve(a, bb, c, u)

    fact = (x[0] + beta * x[-1] / gamma) / (1 + z[0] + beta * z[-1] / gamma)
    x -= fact * z
    return x


@numba.jit
def laplacian(data, dx, dy):
    return (
        (data[:-2, 1:-1, :] - 2 * data[1:-1, 1:-1, :] + data[2:, 1:-1, :]) / dx
        + (data[1:-1, :-2, :] - 2 * data[1:-1, 1:-1, :] + data[1:-1, 2:, :]) /
        dy) / 4


@numba.jit
def horizontal_diffusion_fancy(data, dx, dy, dt):
    K = 0.1
    lap = laplacian(data[1:-1, 1:-1, :], dx, dy)

    flx_x = (lap[1:, 1:-1, :] - lap[:-1, 1:-1, :]) / dx
    flx_x *= flx_x[:, :, :] * (data[3:-2, 3:-3, :] - data[2:-3, 3:-3, :]) < 0

    flx_y = (lap[1:-1, 1:, :] - lap[1:-1, :-1, :]) / dy
    flx_y *= flx_y[:, :, :] * (data[3:-3, 3:-2, :] - data[3:-3, 2:-3, :]) < 0

    return data[3:-3, 3:-3, :] - K * dt * (
        (flx_x[1:, :, :] - flx_x[:-1, :, :]) / dx +
        (flx_y[:, 1:, :] - flx_y[:, :-1, :]) / dy)


@numba.jit
def horizontal_diffusion(data, D, dx, dy, dt):
    flx_x = (data[3:-2, 3:-3, :] - data[2:-3, 3:-3, :]) / dx
    flx_y = (data[3:-3, 3:-2, :] - data[3:-3, 2:-3, :]) / dy
    return data[3:-3, 3:-3, :] + D * dt * (
        (flx_x[1:, :, :] - flx_x[:-1, :, :]) / dx +
        (flx_y[:, 1:, :] - flx_y[:, :-1, :]) / dy)


@numba.jit
def advection_flux_v(v, data0, data, dy):
    weights = np.array([1.0 / 30, -1.0 / 4, 1, -1.0 / 3, -1.0 / 2, 0])
    weights[-1] = -np.sum(weights[:-1])

    negative_mask = v[3:-3, 3:-3, :] < 0
    positive_mask = v[3:-3, 3:-3, :] > 0

    return -v[3:-3, 3:-3, :] * (
        positive_mask *
        -(weights[0] * data[3:-3, :-6, :] + weights[1] * data[3:-3, 1:-5, :] +
          weights[2] * data[3:-3, 2:-4, :] + weights[3] * data[3:-3, 3:-3, :] +
          weights[4] * data[3:-3, 4:-2, :] + weights[5] * data[3:-3, 5:-1, :])
        / dy + negative_mask *
        (weights[5] * data[3:-3, 1:-5, :] + weights[4] * data[3:-3, 2:-4, :] +
         weights[3] * data[3:-3, 3:-3, :] + weights[2] * data[3:-3, 4:-2, :] +
         weights[1] * data[3:-3, 5:-1, :] + weights[0] * data[3:-3, 6:, :]) /
        dy)


@numba.jit
def advection_flux_u(u, data0, data, dx):
    weights = np.array([1.0 / 30, -1.0 / 4, 1, -1.0 / 3, -1.0 / 2, 0])
    weights[-1] = -np.sum(weights[:-1])

    negative_mask = u[3:-3, 3:-3, :] < 0
    positive_mask = u[3:-3, 3:-3, :] > 0

    return -u[3:-3, 3:-3, :] * (
        positive_mask *
        -(weights[0] * data[:-6, 3:-3, :] + weights[1] * data[1:-5, 3:-3, :] +
          weights[2] * data[2:-4, 3:-3, :] + weights[3] * data[3:-3, 3:-3, :] +
          weights[4] * data[4:-2, 3:-3, :] + weights[5] * data[5:-1, 3:-3, :])
        / dx + negative_mask *
        (weights[5] * data[1:-5, 3:-3, :] + weights[4] * data[2:-4, 3:-3, :] +
         weights[3] * data[3:-3, 3:-3, :] + weights[2] * data[4:-2, 3:-3, :] +
         weights[1] * data[5:-1, 3:-3, :] + weights[0] * data[6:, 3:-3, :]) /
        dx)


@numba.jit
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
    d[0] = (1 / dt * data[0] - 0.25 * w[1] * (data[1] - data[0]) / dz -
            0.25 * w[0] * (data[0] - data[-1]) / dz)
    for k in range(1, len(data) - 1):
        a[k] = -0.25 * w[k] / dz
        c[k] = 0.25 * w[k + 1] / dz
        b[k] = 1 / dt - a[k] - c[k]
        d[k] = (1 / dt * data[k] - 0.25 * w[k + 1] *
                (data[k + 1] - data[k]) / dz - 0.25 * w[k] *
                (data[k] - data[k - 1]) / dz)
    a[-1] = -0.25 * w[-2] / dz
    c[-1] = 0.25 * w[-1] / dz
    b[-1] = 1 / dt - a[-1] - c[-1]
    d[-1] = (1 / dt * data[-1] - 0.25 * w[0] * (data[0] - data[-1]) / dz -
             0.25 * w[-1] * (data[-1] - data[-2]) / dz)
    return tridiagonal_solve_periodic(a, b, c, d)


@numba.jit
def advection_flux_w(w, data0, data, dz, dt):
    advected = np.zeros_like(data[3:-3, 3:-3, :])
    for i in range(3, data.shape[0] - 3):
        for j in range(3, data.shape[1] - 3):
            advected[i - 3, j - 3, :] = advection_w_column(
                w[i, j, :], data0[i, j, :], data[i, j, :], dz, dt)

    return (advected - data[3:-3, 3:-3, :]) / dt


@numba.jit
def diffusion_w_column(data, D, dx, dt):
    a = np.zeros(data.shape)
    b = np.zeros(data.shape)
    c = np.zeros(data.shape)
    d = np.zeros(data.shape)
    # assume zero wind, and zero data outside...
    a[0] = -D / 2 * dt
    c[0] = -D / 2 * dt
    b[0] = 1 / dt - a[0] - c[0]
    d[0] = 1 / dt * data[0] + 0.5 * D * (data[1] - 2 * data[0] + data[-1]) / dx
    for k in range(1, len(data) - 1):
        a[k] = -D / 2 * dt
        c[k] = -D / 2 * dt
        b[k] = 1 / dt - a[k] - c[k]
        d[k] = (1 / dt * data[k] + 0.5 * D *
                (data[k + 1] - 2 * data[k] + data[k - 1]) / dx)
    a[-1] = -D / 2 * dt
    c[-1] = -D / 2 * dt
    b[-1] = 1 / dt - a[-1] - c[-1]
    d[-1] = 1 / dt * data[-1] + 0.5 * D * (data[0] - 2 * data[-1] +
                                           data[-2]) / dx
    return tridiagonal_solve_periodic(a, b, c, d)


@numba.jit
def diffusion_flux_w(w, data, D, dx, dt):
    diffused = np.zeros_like(data[3:-3, 3:-3, :])
    for i in range(3, data.shape[0] - 3):
        for j in range(3, data.shape[1] - 3):
            diffused[i - 3, j - 3, :] = diffusion_w_column(
                data[i, j, :], D, dx, dt)

    return (diffused - data[3:-3, 3:-3, :]) / dt


def advection_flux(u, v, w, data0, data, dx, dy, dz, dt):
    return (advection_flux_u(u, data0, data, dx) +
            advection_flux_v(v, data0, data, dy) +
            advection_flux_w(w, data0, data, dz, dt)
            #  + diffusion_flux_w(w, data, dx, dt)
            )


def periodic_boundary_condition(data, boundaries):
    ((x1, x2), (y1, y2), (z1, z2)) = boundaries
    assert x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0 and z1 == 0 and z2 == 0

    # edges
    data[:x1, y1:-y2, :] = data[-x1 - x2:-x2, y1:-y2, :]
    data[-x2:, y1:-y2, :] = data[x1:x1 + x2, y1:-y2, :]
    data[x1:-x2, :y1, :] = data[x1:-x2, -y1 - y2:-y2, :]
    data[x1:-x2, -y2:, :] = data[x1:-x2, y1:y1 + y2, :]

    # corners
    data[:x1, :y1, :] = data[-x1 - x2:-x2, -y1 - y1:-y2, :]
    data[-x2:, -y2:, :] = data[x1:x1 + x2, y1:y1 + y2, :]
    data[:x1, -y2:, :] = data[-x1 - x2:-x2, y1:y1 + y2, :]
    data[-x2:, :y1, :] = data[x1:y1 + x2, -y1 - y2:-y2, :]


def calculate_global_domain(compute_domain, boundaries):
    return tuple(c + sum(b) for c, b in zip(compute_domain, boundaries))


def compute_domain_slice(boundaries):
    return tuple(
        slice(a if a != 0 else None, -b if b != 0 else None)
        for a, b in boundaries)


def add_boundary(data, boundaries):
    ((x1, x2), (y1, y2), (z1, z2)) = boundaries
    global_domain = calculate_global_domain(data.shape, boundaries)
    new_data = np.empty(global_domain)
    new_data[compute_domain_slice(boundaries)] = data

    periodic_boundary_condition(new_data, boundaries)
    return new_data


def step(data, u, v, w, D, dx, dy, dz, dt, boundaries):
    # irk_order=3, irunge_kutta=1
    # it is irk_order=3, but not third order... Wicker, Skamarock (2002)
    # y' = y^n + 1/3 * dt * f(t^n, y)
    # y'' = y^n + 1/2 * dt * f(t^n + 1/3 dt, y')
    # y^{n+1} = y^n + dt * f(t^n + 1/2 dt, y'')

    diff_flux = diffusion_flux_w(w, data, D, dz, dt)

    flux = diff_flux + advection_flux(
        u,
        v,
        w,
        data,
        data,
        dx,
        dy,
        dz,
        dt,
    )
    y1 = add_boundary(data[3:-3, 3:-3, :] + dt / 3 * flux, boundaries)
    flux = diff_flux + advection_flux(u, v, w, data, y1, dx, dy, dz, dt)
    y2 = add_boundary(data[3:-3, 3:-3, :] + dt / 2 * flux, boundaries)
    flux = diff_flux + advection_flux(u, v, w, data, y2, dx, dy, dz, dt)
    data = add_boundary(data[3:-3, 3:-3, :] + dt * flux, boundaries)

    periodic_boundary_condition(data, boundaries)
    data[3:-3, 3:-3, :] = horizontal_diffusion(data, D, dx, dy, dt)
    periodic_boundary_condition(data, boundaries)

    return data
