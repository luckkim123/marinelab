# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.hydrodynamics (Isaac-Sim-free)."""

import torch

from marinelab.core.hydrodynamics import HydrodynamicsModel
from marinelab.core.ocean_current import OceanCurrent


class _Cfg:
    added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
    linear_damping = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
    quadratic_damping = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)
    volume = 0.0113459
    body_name = "base_link"
    center_of_buoyancy = (0.0, 0.0, 0.01)
    center_of_gravity = (0.0, 0.0, 0.0)
    water_density = 997.0
    use_full_coriolis = True
    rigid_body_inertia = (0.12, 0.12, 0.12)
    body_mass = 11.5
    apply_added_mass_force = False
    added_mass_stability_factor = 0.8
    damping_cross_coupling = None
    damping_stability_factor = None


class _CurrentCfg:
    max_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _model(n=4):
    return HydrodynamicsModel(num_envs=n, device="cpu", cfg=_Cfg(), current_cfg=_CurrentCfg(), dt=0.005)


def test_base_parameters_immutable_snapshot():
    m = _model()
    base = m.base_parameters
    # base reflects cfg
    assert torch.allclose(base.linear_damping[0], torch.tensor(_Cfg.linear_damping))
    # mutating live params must not change base_parameters
    m.set_parameters(torch.tensor([0]), linear_damping=torch.zeros(1, 6))
    assert torch.allclose(m.base_parameters.linear_damping[0], torch.tensor(_Cfg.linear_damping))


def test_get_parameters_reads_current_values():
    m = _model()
    p = m.get_parameters()
    assert torch.allclose(p.volume, torch.full((4,), 0.0113459))
    assert p.linear_damping.shape == (4, 6)


def test_set_parameters_absolute_write():
    m = _model()
    m.set_parameters(torch.tensor([1, 3]), volume=torch.tensor([0.02, 0.02]))
    v = m.get_parameters().volume
    assert abs(v[1].item() - 0.02) < 1e-6 and abs(v[3].item() - 0.02) < 1e-6
    assert abs(v[0].item() - 0.0113459) < 1e-6  # untouched


def test_set_parameters_updates_buoyancy_force():
    m = _model()
    before = m.buoyancy_force[0].item()
    m.set_parameters(torch.tensor([0]), volume=torch.tensor([0.02]))
    after = m.buoyancy_force[0].item()
    assert after > before  # rho*g*V increased with volume


def test_scale_parameters_multiplies_base():
    torch.manual_seed(0)
    m = _model(64)
    m.scale_parameters(torch.arange(64), linear_damping=(0.5, 0.5))
    ld = m.get_parameters().linear_damping
    expected = torch.tensor(_Cfg.linear_damping) * 0.5
    assert torch.allclose(ld[0], expected, atol=1e-5)


def test_injected_shared_current():
    cur = OceanCurrent(4, "cpu", _CurrentCfg())
    m = HydrodynamicsModel(4, "cpu", _Cfg(), current=cur, dt=0.005)
    assert m.current is cur


def test_buoyancy_neutral_magnitude():
    m = _model()
    # F_b = rho * g * V ~ 110.97 N for V=0.0113459
    fb = m.buoyancy_force[0].item()
    assert abs(fb - 997.0 * 9.81 * 0.0113459) < 1e-3


def test_set_ocean_current_delegates_to_component():
    m = _model()
    m.set_ocean_current(torch.tensor([0]), velocity=torch.ones(1, 6))
    assert torch.all(m.current.velocity_w[0] == 1.0)
