#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import numpy as np

args = sys.stdin.readline().strip()
compute_domain = list(map(int, args.split(" ")))

data = np.empty(compute_domain, dtype=np.float32)
for i in range(compute_domain[0]):
    for j in range(compute_domain[1]):
        l = list(map(int, sys.stdin.readline().strip().split(" ")))
        assert len(l) == compute_domain[2]
        for k in range(compute_domain[2]):
            data[i, j, k] = l[k]
    sys.stdin.readline()


f, (ax1, ax2) = plt.subplots(1, 2)
#  position_x = (
    #  int(5.5 + self.dt * self.timestep / self.dx * self.u[0, 0, 0])
#  ) % (compute_domain[0])
#  position_y = (
    #  int(5.5 + self.dt * self.timestep / self.dy * self.v[0, 0, 0])
#  ) % (compute_domain[1])
position_x = 5
position_y = 5

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
#  ax2.contour(
    #  expected[:, position_y, :].transpose(), levels=[0.99], corner_mask=True
#  )
#  plt.savefig("u" + str(self.timestep) + ".png")
plt.show()
plt.close("all")

