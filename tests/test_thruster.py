# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.thruster."""

import torch

from marinelab.core.thruster import ThrusterModel


class _Cfg:
    num_thrusters = 6
    max_thrust = 50.0
    thrust_coefficient = 40.0
    time_constant_up = 0.1
    time_constant_down = 0.05
    allocation_matrix = (
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.1, -0.1),
        (0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
        (0.19, -0.19, 0.19, -0.19, 0.0, 0.0),
    )


def test_first_order_step_up():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m.apply_dynamics(torch.ones(1, 6), dt=0.1)
    # alpha = dt/tau_up = 0.1/0.1 = 1.0 -> state snaps to target
    assert torch.allclose(m.state, torch.ones(1, 6), atol=1e-5)


def test_first_order_partial():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m.apply_dynamics(torch.ones(1, 6), dt=0.005)
    # alpha = 0.005/0.1 = 0.05
    assert torch.allclose(m.state, torch.full((1, 6), 0.05), atol=1e-5)


def test_compute_wrench_clamps_thrust():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m._state = torch.ones(1, 6) * 2.0  # 2*40=80 N -> clamps to 50
    f, t = m.compute_wrench()
    # heave row: VL+VR = 50+50 = 100 N
    assert abs(t.shape[-1] - 3) == 0
    assert abs(f[0, 2].item() - 100.0) < 1e-3


def test_reset_zeros_state():
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu")
    m.apply_dynamics(torch.ones(4, 6), dt=0.1)
    m.reset(torch.tensor([0, 2]))
    assert torch.all(m.state[0] == 0.0)
    assert torch.all(m.state[1] != 0.0)
