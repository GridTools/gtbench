#!/usr/bin/env python
#
# gtbench
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

from dataclasses import dataclass
import typing

import gtbench
import numpy as np
import pytest


@dataclass
class State:
    resolution: typing.Tuple[int, int, int]
    delta: typing.Tuple[float, float, float]
    u: gtbench.Storage
    v: gtbench.Storage
    w: gtbench.Storage
    data: gtbench.Storage
    data1: gtbench.Storage
    data2: gtbench.Storage

    @classmethod
    def random(cls, resolution, delta):
        size = (resolution[0] + 2 * gtbench.halo,
                resolution[1] + 2 * gtbench.halo, resolution[2] + 1)

        def field():
            return gtbench.storage_from_array(np.random.uniform(size=size))

        return cls(resolution, delta, field(), field(), field(), field(),
                   field(), field())


def exchange(field):
    halo = gtbench.halo
    field[:halo, :, :] = field[-2 * halo:-halo, :, :]
    field[-halo:, :, :] = field[halo:2 * halo, :, :]
    field[:, :halo, :] = field[:, -2 * halo:-halo, :]
    field[:, -halo:, :] = field[:, halo:2 * halo, :]


def test_storage():
    x = np.random.uniform(size=(7, 9, 11)).astype(gtbench.dtype)
    y = gtbench.storage_from_array(x)
    y = gtbench.array_from_storage(y)
    assert np.all(x == y)


def test_storage_access():
    x = np.random.uniform(size=(7, 9, 11)).astype(gtbench.dtype)
    y = gtbench.storage_from_array(x)
    exchange(x)
    exchange(y)
    y = gtbench.array_from_storage(y)
    assert np.all(x == y)


@pytest.mark.parametrize("stepper", [
    gtbench.hdiff_stepper(0.05),
    gtbench.vdiff_stepper(0.05),
    gtbench.diff_stepper(0.05),
    gtbench.hadv_stepper(),
    gtbench.vadv_stepper(),
    gtbench.rkadv_stepper(),
    gtbench.advdiff_stepper(0.05)
])
def test_stepper(stepper):
    state = State.random((11, 7, 9), (0.1, 0.05, 0.02))
    step = stepper(state, exchange)
    initial = np.copy(gtbench.array_from_storage(state.data))
    step(state, 0.01)
    assert np.any(initial != gtbench.array_from_storage(state.data))
