#!/usr/bin/env python3

# for f in out?; do echo $f; python ../plot.py $f $f.png; done

import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot image')
parser.add_argument('input', action='store')
parser.add_argument('output', action='store')
args = parser.parse_args()

assert len(sys.argv) == 3

with open(args.input) as f:
    sz = f.readline().strip()
    compute_domain = list(map(int, sz.split(" ")))

    data = np.empty(compute_domain, dtype=np.float32)
    for k in range(compute_domain[2]):
        for j in range(compute_domain[1]):
            line = f.readline().strip()
            numbers = list(map(float, line.split(" ")))
            assert len(numbers) == compute_domain[0]
            for i in range(compute_domain[0]):
                data[i, j, k] = numbers[i]
        f.readline()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
#  position_x = (
    #  int(5.5 + self.dt * self.timestep / self.dx * self.u[0, 0, 0])
#  ) % (compute_domain[0])
#  position_y = (
    #  int(5.5 + self.dt * self.timestep / self.dy * self.v[0, 0, 0])
#  ) % (compute_domain[1])
position_x = 5
position_y = 7
position_z = 5

ax1.set_title("[" + str(position_x) + ", :, :]")
ax1.pcolor(
    data[position_x, :, :].transpose(),
    vmin=-1,
    vmax=1,
    cmap="seismic",
)
#  ax1.contour(
    #  expected[position_x, :, :].transpose(), levels=[0.99], corner_mask=True
#  )

ax2.set_title("[:, " + str(position_y) + ", :]")
ax2.pcolor(
    data[:, position_y, :].transpose(),
    vmin=-1,
    vmax=1,
    cmap="seismic",
)

ax3.set_title("[:, :, " + str(position_z) + "]")
ax3.pcolor(
    data[:, :, position_z],
    vmin=-1,
    vmax=1,
    cmap="seismic",
)
#  ax2.contour(
    #  expected[:, position_y, :].transpose(), levels=[0.99], corner_mask=True
#  )
plt.savefig(args.output)
#  plt.show()
plt.close("all")

