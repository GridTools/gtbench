import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import advdiff


def order_verification(func, nmin, nmax):
    ns, errors = [], []
    n = nmin
    while n <= nmax:
        ns.append(n)
        errors.append(func(n))
        n *= 2
    ns = np.array(ns)
    errors = np.array(errors)

    orders = np.log2(errors[:-1] / errors[1:])
    print('errors: ', *errors)
    print('orders: ', *orders)

    plt.figure()
    plt.title(func.__name__.replace('_', ' ').title())
    plt.plot(ns, errors, label='Measured')
    for order in range(1, 3):
        scaling = curve_fit(lambda n, s: s * n**-order, ns, errors)[0][0]
        plt.plot(ns, scaling * ns**-float(order), label=f'O({order})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()


def grid(n, domain, staggered_z=False):
    i, j, k = np.indices((n, n, n + 1 if staggered_z else n))
    dx, dy, dz = np.asarray(domain) / n
    x, y, z = (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz
    if staggered_z:
        z -= 0.5 * dz
    return (x, y, z), (dx, dy, dz)


def exact_constant_velocity(n, D, t):
    (x, y, z), d = grid(n, (2 * np.pi, ) * 3)
    data = np.sin(x - t) * np.sin(y + t) * np.cos(z - t) * np.exp(-t * D * 3)
    u = np.ones((n, n, n))
    v = -np.ones((n, n, n))
    w = np.ones((n, n, n + 1))
    return data, (u, v, w), d


def exact_zero_velocity(n, D, t):
    (x, y, z), d = grid(n, (2 * np.pi, ) * 3)
    data = np.sin(x) * np.sin(y) * np.cos(z) * np.exp(-t * D * 3)
    u = np.zeros((n, n, n))
    v = np.zeros((n, n, n))
    w = np.zeros((n, n, n + 1))
    return data, (u, v, w), d


def exact_spatially_varying_velocity_xy(n, D, t):
    (x, y, z), d = grid(n, (2 * np.pi, ) * 3)
    data = np.sin(x) * np.sin(y) * np.exp(-t * 2 * D)
    u = -np.sin(x) * np.cos(y)
    v = np.cos(x) * np.sin(y)
    w = np.zeros((n, n, n + 1))
    return data, (u, v, w), d


def exact_spatially_varying_velocity_xz(n, D, t):
    domain = (2 * np.pi, ) * 3
    (x, y, z), d = grid(n, domain)
    data = np.sin(x) * np.sin(z) * np.exp(-t * 2 * D)
    u = -np.sin(x) * np.cos(z)
    v = np.zeros((n, n, n))
    (x, y, z), _ = grid(n, domain, staggered_z=True)
    w = np.cos(x) * np.sin(z)
    return data, (u, v, w), d


def exact_spatially_varying_velocity_fancy(n, D, t):
    phi = np.pi / 4
    domain = (2 * np.pi, np.sqrt(2) * 2 * np.pi, np.sqrt(2) * 2 * np.pi)
    (x, y, z), d = grid(n, domain)
    data = -np.exp(-2 * D * t) * np.sin(x) * np.sin(y * np.sin(phi) -
                                                    z * np.cos(phi))
    u = -np.sin(x) * np.cos(y * np.sin(phi) - z * np.cos(phi))
    v = np.sin(phi) * np.sin(y * np.sin(phi) - z * np.cos(phi)) * np.cos(x)
    (x, y, z), _ = grid(n, domain, staggered_z=True)
    w = -np.sin(y * np.sin(phi) - z * np.cos(phi)) * np.cos(x) * np.cos(phi)
    return data, (u, v, w), d


def run(D, n, dt, tmax, exact):
    compute_domain = n, n, n
    boundaries = (3, 3), (3, 3), (0, 0)
    data, (u, v, w), (dx, dy, dz) = exact(n, D, 0)

    def prepare(field):
        return advdiff.periodic_boundary_condition(
            advdiff.add_boundary(field, boundaries), boundaries)

    data = prepare(data)
    u = prepare(u)
    v = prepare(v)
    w = prepare(w)

    t = 0.0
    while t < tmax:
        data = advdiff.step(data, u, v, w, D, dx, dy, dz, dt, boundaries)
        t += dt

    cds = advdiff.compute_domain_slice(boundaries)
    return np.sqrt(np.sum((data[cds] - exact(n, D, t)[0])**2) * dx * dy * dz)


def space_dependent(D, dt, tmax, exact):
    def compute(n):
        print(f'computing for n = {n}')
        return run(D, n, dt, tmax, exact)

    return compute


def time_dependent(D, n, tmax, exact):
    def compute(steps):
        print(f'computing {steps} steps')
        return run(D, n, tmax / steps, tmax, exact)

    return compute


def space_time_dependent(D, tmax, exact):
    def compute(n):
        print(f'computing n = {n} with {n} steps')
        return run(D, n, tmax / n, tmax, exact)

    return compute


order_verification(
    space_dependent(0.1, 1e-4, 0.001, exact_spatially_varying_velocity_fancy),
    8, 128)

order_verification(
    time_dependent(0.1, 128, 0.1, exact_spatially_varying_velocity_fancy), 4,
    64)

order_verification(
    space_time_dependent(0.1, 0.01, exact_spatially_varying_velocity_fancy), 4,
    64)

plt.show()
