# marinelab Framework-ization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `marinelab` into a general-purpose UUV framework with a clean `core/` API (setter-based DR, injectable `OceanCurrent`), fix diagnosed physics/bugs, restructure bluerov as a reference example, add a pytest suite, and document the framework workflow — all while keeping `marinelab.physics` imports working via a shim so the concurrently-edited constrained-albc stays unbroken this phase.

**Architecture:** New `marinelab/core/` holds the framework (hydrodynamics, thruster, ocean_current, parameters, volume). `marinelab/physics` becomes a deprecated re-export shim. DR moves from leaky in-place buffer index-writes to an explicit `set_parameters`/`scale_parameters`/`get_parameters` API backed by a framework-owned base-cache. Ocean current splits into an injectable `OceanCurrent` component supporting a shared-source pattern. `marinelab/tasks/bluerov` is restructured to consume the new API as the reference example.

**Tech Stack:** Python 3.10+, PyTorch (GPU tensors), Isaac Lab (`isaaclab.utils.math`), gymnasium, pytest. Tests run Isaac-Sim-free via the `_MockModule` + `importlib.util` pattern from `/workspace/constrained-albc/tests/test_tdc_controller.py`.

**Spec:** `/workspace/marinelab/docs/superpowers/specs/2026-05-25-marinelab-framework-design.md`

**Conventions:** English code/headings, Korean only in user-facing prose, no emojis, no AI attribution. Commit after each task. `cd /workspace/marinelab` for all commands. Do NOT touch isaaclab. Do NOT edit constrained-albc this phase (§7 of spec — separate later phase).

---

## File Structure

```
marinelab/
├── core/
│   ├── __init__.py            # exports HydrodynamicsModel, ThrusterModel, OceanCurrent, HydroParams, cfgs
│   ├── parameters.py          # HydroParams dataclass, default_rigid_inertia(), HydroBaseCache
│   ├── ocean_current.py       # OceanCurrent component
│   ├── hydrodynamics.py       # HydrodynamicsModel (moved from physics/, + new API + fixes)
│   ├── thruster.py            # ThrusterModel (moved from physics/, + tau caching)
│   └── volume.py              # moved from utils/volume.py
├── physics/
│   └── __init__.py            # deprecated shim: re-export from marinelab.core
├── assets/                    # bluerov.py: A1 + A2 fixes
├── tasks/bluerov/             # restructured to use core API; dead code removed
└── tests/
    ├── conftest.py            # shared Isaac-Sim-free mocks (real quat math)
    ├── test_parameters.py
    ├── test_ocean_current.py
    ├── test_hydrodynamics.py
    ├── test_thruster.py
    └── test_bluerov_smoke.py
```

**Execution order rationale:** Build the new `core` primitives bottom-up (parameters → ocean_current → hydrodynamics → thruster) with tests, each as a *new* module so nothing breaks mid-flight. Then flip `physics` to a shim. Then fix asset-level physics bugs. Then restructure bluerov. README last.

---

## Phase 0 — Test harness

### Task 0: Isaac-Sim-free test harness with REAL quaternion math

**Why this matters:** marinelab physics calls `isaaclab.utils.math.quat_apply_inverse`, `quat_from_euler_xyz`, `quat_mul`, `subtract_frame_transforms`. The generic `_MockModule.__call__` returns `args[0]` — wrong for these (would return the quaternion, not the rotated vector), making physics tests meaningless. The harness must inject a *correct* `quat_apply_inverse`.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create `tests/__init__.py`** (empty file).

- [ ] **Step 2: Write `tests/conftest.py`** — mocks `isaaclab` packages but provides a real `quat_apply_inverse` and `configclass`.

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures: run marinelab.core tests without Isaac Sim.

The generic mock-everything approach breaks physics tests because
quat_apply_inverse must actually rotate vectors. This conftest installs a
correct quaternion implementation while stubbing the rest of isaaclab.
"""

import sys
import types

import torch


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vec by the inverse of quat (w, x, y, z). Matches isaaclab.utils.math."""
    q = quat
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # inverse rotation = conjugate for unit quat: (w, -x, -y, -z)
    # v' = v + 2 * q_conj_vec x (q_conj_vec x v + w_conj * v), w_conj = w
    qvec = torch.stack([-x, -y, -z], dim=-1)
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _install_isaaclab_mocks():
    if "isaaclab" in sys.modules:
        return
    for name in ["isaaclab", "isaaclab.utils", "isaaclab.utils.math"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["isaaclab.utils"].configclass = _configclass
    math_mod = sys.modules["isaaclab.utils.math"]
    math_mod.quat_apply_inverse = _quat_apply_inverse
    math_mod.quat_from_euler_xyz = _quat_from_euler_xyz


def _configclass(cls):
    """Minimal stand-in for isaaclab.utils.configclass.

    The real one auto-generates __init__ from annotated class attributes with
    defaults. Our cfg classes only use simple typed defaults, so dataclass
    semantics suffice for tests.
    """
    import dataclasses

    return dataclasses.dataclass(cls)


_install_isaaclab_mocks()
```

- [ ] **Step 3: Verify the quat math is correct** with a quick check.

Run: `cd /workspace/marinelab && python -c "
import tests.conftest as c, torch
q = c._quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.5707963]))
v = torch.tensor([[1.0,0.0,0.0]])
print(c._quat_apply_inverse(q, v))
"`
Expected: approximately `tensor([[ 0., -1.,  0.]])` (rotating world-x into a body yawed +90deg gives body -y).

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add Isaac-Sim-free harness with real quaternion math"
```

---

## Phase 1 — core/parameters.py

### Task 1: `HydroParams` dataclass + `default_rigid_inertia` helper

**Files:**
- Create: `marinelab/core/__init__.py` (start minimal, grow later)
- Create: `marinelab/core/parameters.py`
- Create: `tests/test_parameters.py`

- [ ] **Step 1: Write the failing test** in `tests/test_parameters.py`.

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.parameters."""

import torch

from marinelab.core.parameters import HydroParams, default_rigid_inertia


def test_default_rigid_inertia_uses_added_mass_fallback():
    # When rigid_body_inertia is None, fallback = added_mass[3:6] * 0.5
    class Cfg:
        added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
        rigid_body_inertia = None

    assert default_rigid_inertia(Cfg()) == [0.06, 0.06, 0.06]


def test_default_rigid_inertia_prefers_explicit():
    class Cfg:
        added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
        rigid_body_inertia = (0.2, 0.3, 0.4)

    assert default_rigid_inertia(Cfg()) == [0.2, 0.3, 0.4]


def test_hydroparams_holds_optional_tensor_fields():
    p = HydroParams(volume=torch.ones(4))
    assert p.volume.shape == (4,)
    assert p.added_mass is None  # unset fields default to None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/marinelab && python -m pytest tests/test_parameters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marinelab.core'`.

- [ ] **Step 3: Create `marinelab/core/__init__.py`** (minimal for now).

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .parameters import HydroParams, default_rigid_inertia

__all__ = ["HydroParams", "default_rigid_inertia"]
```

- [ ] **Step 4: Write `marinelab/core/parameters.py`.**

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Hydrodynamic parameter bundle and base-parameter cache.

HydroParams is a named bundle of per-env parameter tensors used by both the
read API (get_parameters, for privileged observations) and the write API
(set_parameters, for domain randomization). default_rigid_inertia centralizes
the added_mass[3:6]*0.5 fallback that was previously duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HydroParams:
    """Named bundle of per-env hydrodynamic parameter tensors.

    All fields are optional: a None field means "not provided / leave unchanged"
    on the write path, and is simply absent on the read path if not requested.
    """

    added_mass: torch.Tensor | None = None  # (N, 6)
    linear_damping: torch.Tensor | None = None  # (N, 6)
    quadratic_damping: torch.Tensor | None = None  # (N, 6)
    volume: torch.Tensor | None = None  # (N,)
    water_density: torch.Tensor | None = None  # (N,)
    center_of_buoyancy: torch.Tensor | None = None  # (N, 3)
    center_of_gravity: torch.Tensor | None = None  # (N, 3)
    rigid_body_inertia: torch.Tensor | None = None  # (N, 3)
    body_mass: torch.Tensor | None = None  # (N,)


def default_rigid_inertia(cfg) -> list[float]:
    """Resolve rigid-body inertia, falling back to added_mass[3:6] * 0.5.

    Single source of the heuristic that was duplicated across the model init,
    stability validation, and randomization code.

    Args:
        cfg: A config with `added_mass` (len-6) and optional `rigid_body_inertia`.

    Returns:
        Three-element inertia diagonal [Ixx, Iyy, Izz].
    """
    if getattr(cfg, "rigid_body_inertia", None) is not None:
        return list(cfg.rigid_body_inertia)
    return [x * 0.5 for x in cfg.added_mass[3:6]]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/marinelab && python -m pytest tests/test_parameters.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add marinelab/core/__init__.py marinelab/core/parameters.py tests/test_parameters.py
git commit -m "feat(core): add HydroParams bundle and default_rigid_inertia helper"
```

---

## Phase 2 — core/ocean_current.py

### Task 2: `OceanCurrent` component (split from hydro)

**Files:**
- Create: `marinelab/core/ocean_current.py`
- Modify: `marinelab/core/__init__.py`
- Create: `tests/test_ocean_current.py`

**Note on cfg import:** `OceanCurrentCfg` currently lives in `marinelab/assets/uuv_cfg.py`. To avoid a circular dependency (`core` → `assets` → ...), `OceanCurrent.__init__` takes the cfg object by duck-typing (`cfg.max_velocity`, `cfg.noise_scale`), not by importing the class. The cfg class itself stays in `assets/uuv_cfg.py`.

- [ ] **Step 1: Write the failing test** in `tests/test_ocean_current.py`.

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.ocean_current."""

import torch

from marinelab.core.ocean_current import OceanCurrent


class _Cfg:
    max_velocity = (0.5, 0.5, 0.25, 0.0, 0.0, 0.0)
    noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_init_zero_velocity():
    oc = OceanCurrent(num_envs=4, device="cpu", cfg=_Cfg())
    assert oc.velocity_w.shape == (4, 6)
    assert torch.all(oc.velocity_w == 0.0)
    assert oc.max_velocity.shape == (6,)


def test_set_explicit_velocity():
    oc = OceanCurrent(4, "cpu", _Cfg())
    vel = torch.ones(2, 6)
    oc.set(torch.tensor([0, 2]), velocity=vel)
    assert torch.all(oc.velocity_w[0] == 1.0)
    assert torch.all(oc.velocity_w[1] == 0.0)  # untouched
    assert torch.all(oc.velocity_w[2] == 1.0)


def test_set_sampled_within_bounds():
    torch.manual_seed(0)
    oc = OceanCurrent(64, "cpu", _Cfg())
    oc.set(torch.arange(64))
    maxv = oc.max_velocity
    assert torch.all(oc.velocity_w.abs() <= maxv + 1e-6)


def test_set_strength_scales_sample():
    torch.manual_seed(0)
    oc = OceanCurrent(64, "cpu", _Cfg())
    oc.set(torch.arange(64), strength=torch.zeros(64))
    assert torch.allclose(oc.velocity_w, torch.zeros(64, 6))


def test_add_drift_and_reset():
    oc = OceanCurrent(4, "cpu", _Cfg())
    oc.add_drift(torch.full((4, 6), 0.1))
    assert torch.allclose(oc.velocity_w, torch.full((4, 6), 0.1))
    oc.reset(torch.tensor([0, 1]))
    assert torch.all(oc.velocity_w[0] == 0.0)
    assert torch.allclose(oc.velocity_w[2], torch.full((6,), 0.1))
    oc.reset()  # all
    assert torch.all(oc.velocity_w == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/marinelab && python -m pytest tests/test_ocean_current.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marinelab.core.ocean_current'`.

- [ ] **Step 3: Write `marinelab/core/ocean_current.py`.**

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Ocean current as a standalone, injectable component.

A single OceanCurrent instance can be shared by multiple HydrodynamicsModel
instances (e.g. a vehicle body and a buoyancy body) so they see the same flow
field without manual buffer copying. Time-varying current (e.g. OU drift) is
injected by the caller via set() / add_drift(); the framework stores the field
and exposes it for relative-velocity computation and clamping.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class OceanCurrent:
    """World-frame 6-DOF ocean current field shared across hydrodynamics models.

    Attributes:
        num_envs: Number of parallel environments.
        device: Computation device.
    """

    def __init__(self, num_envs: int, device: str, cfg) -> None:
        """Initialize the current field to zero.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Object exposing `max_velocity` (len-6) and `noise_scale` (len-6).
        """
        self.num_envs = num_envs
        self.device = device
        kw = {"dtype": torch.float32, "device": device}
        self._velocity_w = torch.zeros(num_envs, 6, **kw)
        self._max_velocity = torch.tensor(cfg.max_velocity, **kw)
        self._noise_scale = torch.tensor(cfg.noise_scale, **kw)

    @property
    def velocity_w(self) -> torch.Tensor:
        """Current velocity in world frame. Shape: (num_envs, 6)."""
        return self._velocity_w

    @property
    def max_velocity(self) -> torch.Tensor:
        """Configured maximum current velocity. Shape: (6,)."""
        return self._max_velocity

    def set(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
        strength: torch.Tensor | None = None,
    ) -> None:
        """Set current velocity for specified environments.

        Args:
            env_ids: Environment indices to update.
            velocity: Explicit per-env (n, 6) velocity. If None, sampled uniformly
                from [-max, +max] plus optional gaussian noise.
            strength: Optional per-env scale in [0, 1] applied to the sampled
                velocity (ignored when `velocity` is given).
        """
        env_ids = self._as_index(env_ids)
        if velocity is None:
            n = len(env_ids)
            rand = torch.rand(n, 6, device=self.device)
            velocity = rand * self._max_velocity * 2 - self._max_velocity
            if self._noise_scale.any():
                velocity = velocity + torch.randn(n, 6, device=self.device) * self._noise_scale
            if strength is not None:
                velocity = velocity * strength.unsqueeze(-1)
        self._velocity_w[env_ids] = velocity

    def add_drift(self, delta: torch.Tensor) -> None:
        """Add an increment to the current field (for time-varying current).

        Args:
            delta: (num_envs, 6) increment to add to velocity_w.
        """
        self._velocity_w = self._velocity_w + delta

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Zero the current for specified environments (all if None)."""
        if env_ids is None:
            self._velocity_w.zero_()
        else:
            self._velocity_w[self._as_index(env_ids)] = 0.0

    def _as_index(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            return torch.tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids
```

- [ ] **Step 4: Export from `core/__init__.py`** — update the imports and `__all__`.

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia

__all__ = ["HydroParams", "default_rigid_inertia", "OceanCurrent"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/marinelab && python -m pytest tests/test_ocean_current.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add marinelab/core/ocean_current.py marinelab/core/__init__.py tests/test_ocean_current.py
git commit -m "feat(core): add injectable OceanCurrent component with shared-source support"
```

---

## Phase 3 — core/hydrodynamics.py (move + new API + A1 fix)

### Task 3: Move HydrodynamicsModel into core, inject OceanCurrent, add setter/get API

**Files:**
- Create: `marinelab/core/hydrodynamics.py` (moved from `marinelab/physics/hydrodynamics.py`, modified)
- Modify: `marinelab/core/__init__.py`
- Create: `tests/test_hydrodynamics.py`

**Changes vs the original `physics/hydrodynamics.py`:**
1. Import `OceanCurrent` from `.ocean_current` and `HydroParams`, `default_rigid_inertia` from `.parameters`.
2. Constructor: accept `current: OceanCurrent | None = None`. If None, construct one internally from `current_cfg` (backward compatible). Store as `self._current`. Replace internal `self._current_velocity` / `self._max_current_vel` usage with `self._current.velocity_w` / `self._current.max_velocity`.
3. Add `current` property returning `self._current`.
4. Add `base_parameters`, `get_parameters(env_ids=None)`, `set_parameters(env_ids, **fields)`, `scale_parameters(env_ids, **ranges)`.
5. Use `default_rigid_inertia(cfg)` in `_init_hydrodynamic_matrices` and `_validate_added_mass_stability` instead of the inline `*0.5`.
6. Cache `self._up_dir_w` buffer in `_init_state_buffers` (efficiency C2); use it in `_compute_buoyancy_quat`.
7. Keep `set_ocean_current` as a thin delegator to `self._current.set(...)` (backward compat). Keep all existing public properties as read getters.

- [ ] **Step 1: Write the failing test** in `tests/test_hydrodynamics.py`.

```python
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
    assert v[1].item() == 0.02 and v[3].item() == 0.02
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/marinelab && python -m pytest tests/test_hydrodynamics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marinelab.core.hydrodynamics'`.

- [ ] **Step 3: Create `marinelab/core/hydrodynamics.py`** by copying `marinelab/physics/hydrodynamics.py` verbatim, then applying the changes below.

Run first: `cp marinelab/physics/hydrodynamics.py marinelab/core/hydrodynamics.py`

Then edit `marinelab/core/hydrodynamics.py`:

(3a) Replace the import block. Change:
```python
from marinelab.assets import HydrodynamicsCfg, OceanCurrentCfg
```
to:
```python
from marinelab.assets import HydrodynamicsCfg, OceanCurrentCfg

from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia
```

(3b) Change the constructor signature to accept an injected current. Replace:
```python
    def __init__(
        self,
        num_envs: int,
        device: str,
        cfg: HydrodynamicsCfg,
        current_cfg: OceanCurrentCfg | None = None,
        dt: float = 0.01,
        articulation_prim_path: str | None = None,
    ) -> None:
```
(keep the docstring) and at the end of `__init__` body, replace the call order. The original ends with:
```python
        self._init_hydrodynamic_matrices(cfg)
        self._init_buoyancy_params(cfg, articulation_prim_path)
        self._init_state_buffers()
```
Insert current setup BEFORE `_init_state_buffers`, and add `current` param:
```python
    def __init__(
        self,
        num_envs: int,
        device: str,
        cfg: HydrodynamicsCfg,
        current_cfg: OceanCurrentCfg | None = None,
        dt: float = 0.01,
        articulation_prim_path: str | None = None,
        current: OceanCurrent | None = None,
    ) -> None:
```
and in the body:
```python
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.dt = dt
        self._tensor_kwargs = {"dtype": torch.float32, "device": device}

        # Ocean current: use injected shared component, or build one from cfg.
        if current is not None:
            self._current = current
        else:
            self._current = OceanCurrent(num_envs, device, current_cfg or OceanCurrentCfg())

        self._init_hydrodynamic_matrices(cfg)
        self._init_buoyancy_params(cfg, articulation_prim_path)
        self._init_state_buffers()
```
(Remove the old `self.current_cfg = current_cfg or OceanCurrentCfg()` line.)

(3c) In `_init_hydrodynamic_matrices`, replace the inertia fallback. Change:
```python
        if cfg.rigid_body_inertia is not None:
            inertia = torch.tensor(cfg.rigid_body_inertia, **self._tensor_kwargs)
        else:
            # Fallback: estimate from added mass rotational terms (heuristic)
            inertia = torch.tensor(cfg.added_mass[3:6], **self._tensor_kwargs) * 0.5
```
to:
```python
        inertia = torch.tensor(default_rigid_inertia(cfg), **self._tensor_kwargs)
```

(3d) In `_validate_added_mass_stability`, replace:
```python
            rot_inertia = (
                list(cfg.rigid_body_inertia)
                if cfg.rigid_body_inertia is not None
                else [x * 0.5 for x in cfg.added_mass[3:6]]
            )
```
to:
```python
            rot_inertia = default_rigid_inertia(cfg)
```

(3e) In `_init_state_buffers`, replace the body. Change:
```python
    def _init_state_buffers(self) -> None:
        """Initialize state buffers and ocean current."""
        # Ocean current state (world frame, 6-DOF)
        self._current_velocity = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)
        self._max_current_vel = torch.tensor(self.current_cfg.max_velocity, **self._tensor_kwargs)
        self._current_noise_scale = torch.tensor(self.current_cfg.noise_scale, **self._tensor_kwargs)

        # PhysX acceleration cache (body frame, updated via update_physx_state)
        self._physx_acc_b = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)
```
to:
```python
    def _init_state_buffers(self) -> None:
        """Initialize cached buffers (current lives in the OceanCurrent component)."""
        # PhysX acceleration cache (body frame, updated via update_physx_state)
        self._physx_acc_b = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)

        # Constant world-up direction, cached to avoid per-step allocation.
        self._up_dir_w = torch.zeros(self.num_envs, 3, **self._tensor_kwargs)
        self._up_dir_w[:, 2] = 1.0
```

(3f) In `compute_forces`, replace the current read. Change:
```python
        # Transform ocean current to body frame
        current_lin_b = quat_apply_inverse(root_quat_w, self._current_velocity[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, self._current_velocity[:, 3:])
```
to:
```python
        # Transform ocean current to body frame
        current_w = self._current.velocity_w
        current_lin_b = quat_apply_inverse(root_quat_w, current_w[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, current_w[:, 3:])
```

(3g) In `_compute_buoyancy_quat`, use the cached up vector. Change:
```python
        # World up direction
        up_dir_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        up_dir_w[:, 2] = 1.0

        # Transform to body frame
        up_dir_b = quat_apply_inverse(root_quat_w, up_dir_w)
```
to:
```python
        # Transform cached world-up to body frame
        up_dir_b = quat_apply_inverse(root_quat_w, self._up_dir_w)
```

(3h) Replace `set_ocean_current` and `reset` to delegate to the component. Change the whole `set_ocean_current` method body to:
```python
    def set_ocean_current(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
        strength: torch.Tensor | None = None,
    ) -> None:
        """Set ocean current (delegates to the OceanCurrent component)."""
        self._current.set(env_ids, velocity=velocity, strength=strength)
```
And in `reset`, change the current-zeroing lines:
```python
        self._current_velocity[idx] = 0.0
        self._physx_acc_b[idx] = 0.0
        self._water_density[idx] = self.cfg.water_density
```
to:
```python
        self._current.reset(env_ids)
        self._physx_acc_b[idx] = 0.0
        self._water_density[idx] = self.cfg.water_density
```

(3i) Add the new API methods and a `current` property. Insert after the existing `apply_added_mass` property (near the other `@property` definitions), before `update_buoyancy_force`:
```python
    @property
    def current(self) -> OceanCurrent:
        """The ocean current component (shareable across models)."""
        return self._current

    @property
    def base_parameters(self) -> HydroParams:
        """Immutable snapshot of nominal parameters from cfg (computed once)."""
        return self._base_parameters

    def get_parameters(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> HydroParams:
        """Read current per-env hydrodynamic parameters.

        Args:
            env_ids: Indices to read (all if None).

        Returns:
            HydroParams with per-env tensors for the requested environments.
        """
        idx = slice(None) if env_ids is None else self._to_env_ids(env_ids)
        am_diag = torch.diagonal(self._added_mass_matrix[idx], dim1=-2, dim2=-1)
        return HydroParams(
            added_mass=am_diag,
            linear_damping=self._linear_damping_diag[idx],
            quadratic_damping=self._quadratic_damping_diag[idx],
            volume=self._volume[idx],
            water_density=self._water_density[idx],
            center_of_buoyancy=self._r_cb[idx],
            center_of_gravity=self._r_cg[idx],
            rigid_body_inertia=self._rigid_body_inertia[idx],
            body_mass=None if self._body_mass is None else self._body_mass[idx],
        )

    def set_parameters(self, env_ids: torch.Tensor | Sequence[int], **fields) -> None:
        """Write absolute per-env parameter values. None/absent fields untouched.

        Accepted keyword fields match HydroParams: added_mass, linear_damping,
        quadratic_damping, volume, water_density, center_of_buoyancy,
        center_of_gravity, rigid_body_inertia, body_mass. Automatically refreshes
        buoyancy force when volume or water_density changes.

        Args:
            env_ids: Environment indices to write.
            **fields: Per-env tensors keyed by HydroParams field name.
        """
        idx = self._to_env_ids(env_ids)
        if (am := fields.get("added_mass")) is not None:
            self._added_mass_matrix[idx] = torch.diag_embed(am)
        if (ld := fields.get("linear_damping")) is not None:
            self._linear_damping_diag[idx] = ld
        if (qd := fields.get("quadratic_damping")) is not None:
            self._quadratic_damping_diag[idx] = qd
        if (cb := fields.get("center_of_buoyancy")) is not None:
            self._r_cb[idx] = cb
        if (cg := fields.get("center_of_gravity")) is not None:
            self._r_cg[idx] = cg
        if (rbi := fields.get("rigid_body_inertia")) is not None:
            self._rigid_body_inertia[idx] = rbi
        if (bm := fields.get("body_mass")) is not None and self._body_mass is not None:
            self._body_mass[idx] = bm
        vol_changed = (vol := fields.get("volume")) is not None
        den_changed = (den := fields.get("water_density")) is not None
        if vol_changed:
            self._volume[idx] = vol
        if den_changed:
            self._water_density[idx] = den
        if vol_changed or den_changed:
            self.update_buoyancy_force(env_ids)

    def scale_parameters(self, env_ids: torch.Tensor | Sequence[int], **ranges) -> None:
        """Convenience DR: set each named field to base * uniform(lo, hi) per env.

        Args:
            env_ids: Environment indices to randomize.
            **ranges: field_name=(lo, hi) for any HydroParams scalar/vector field.
                e.g. added_mass=(0.8, 1.2), linear_damping=(0.5, 1.0).
        """
        idx = self._to_env_ids(env_ids)
        n = len(idx)
        base = self._base_parameters
        out: dict[str, torch.Tensor] = {}
        for name, (lo, hi) in ranges.items():
            base_val = getattr(base, name)
            scale = torch.rand((n,) + base_val.shape[1:], device=self.device) * (hi - lo) + lo
            out[name] = base_val[idx] * scale
        self.set_parameters(env_ids, **out)
```

(3j) Build the base-parameter snapshot in `__init__`. At the very end of `__init__` (after `_init_state_buffers()`), add:
```python
        self._base_parameters = self._snapshot_parameters()
```
and add the helper method (place near `get_parameters`):
```python
    def _snapshot_parameters(self) -> HydroParams:
        """Capture nominal parameters as an immutable (cloned) snapshot."""
        am_diag = torch.diagonal(self._added_mass_matrix, dim1=-2, dim2=-1)
        return HydroParams(
            added_mass=am_diag.clone(),
            linear_damping=self._linear_damping_diag.clone(),
            quadratic_damping=self._quadratic_damping_diag.clone(),
            volume=self._volume.clone(),
            water_density=self._water_density.clone(),
            center_of_buoyancy=self._r_cb.clone(),
            center_of_gravity=self._r_cg.clone(),
            rigid_body_inertia=self._rigid_body_inertia.clone(),
            body_mass=None if self._body_mass is None else self._body_mass.clone(),
        )
```

- [ ] **Step 4: Export from `core/__init__.py`.**

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .hydrodynamics import HydrodynamicsModel
from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia

__all__ = ["HydrodynamicsModel", "HydroParams", "default_rigid_inertia", "OceanCurrent"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/marinelab && python -m pytest tests/test_hydrodynamics.py -v`
Expected: PASS (9 tests). If `test_buoyancy_neutral_magnitude` fails, check `_buoyancy_force_base = rho*g*V` is computed after volume init.

- [ ] **Step 6: Commit**

```bash
git add marinelab/core/hydrodynamics.py marinelab/core/__init__.py tests/test_hydrodynamics.py
git commit -m "feat(core): move HydrodynamicsModel into core with setter/get DR API, injectable current, cached up-vector"
```

---

### Task 4: Physics correctness regression tests (C_A sign, damping clamp, restoring)

These lock in the physics so later refactors stay behavior-identical, and document the A1 decision.

**Files:**
- Modify: `tests/test_hydrodynamics.py` (append)

- [ ] **Step 1: Append regression tests** to `tests/test_hydrodynamics.py`.

```python
def test_coriolis_added_mass_sign_known_value():
    # C_A(v) v for diagonal added mass, hand-computed reference.
    m = _model(1)
    lin = torch.tensor([[1.0, 0.5, -0.5]])
    ang = torch.tensor([[0.1, -0.2, 0.3]])
    vel6 = torch.cat([lin, ang], dim=-1)
    force, torque = m._compute_coriolis_added_mass(vel6)
    # a1 = M_A[:3] * lin = [5.5, 6.35, -7.285]; force = -(a1 x ang)
    a1 = torch.tensor([5.5 * 1.0, 12.7 * 0.5, 14.57 * -0.5])
    expected_force = -torch.cross(a1, ang[0], dim=-1)
    assert torch.allclose(force[0], expected_force, atol=1e-4)


def test_damping_quadratic_form():
    m = _model(1)
    vel = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d = m._compute_damping(vel)
    # surge: D_l*v + D_q*|v|*v = 4.03*2 + 18.18*2*2 = 8.06 + 72.72
    assert abs(d[0, 0].item() - (4.03 * 2 + 18.18 * 4)) < 1e-3


def test_buoyancy_restoring_zero_when_upright_and_cob_centered():
    # CoB at (0,0,0.01), upright: moment = r_cb x F_b. F_b along +z body => x,y moment only.
    m = _model(1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity
    wrench = m._compute_buoyancy_quat(quat)
    # r_cb=(0,0,0.01) x F=(0,0,Fb) => cross is zero (parallel)
    assert torch.allclose(wrench[0, 3:], torch.zeros(3), atol=1e-5)
```

- [ ] **Step 2: Run**

Run: `cd /workspace/marinelab && python -m pytest tests/test_hydrodynamics.py -v`
Expected: PASS (12 tests total).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hydrodynamics.py
git commit -m "test(core): lock in C_A sign, damping form, restoring-moment regression"
```

---

## Phase 4 — core/thruster.py (move + tau caching) + volume move

### Task 5: Move ThrusterModel into core, precompute tau (efficiency C3)

**Files:**
- Create: `marinelab/core/thruster.py` (moved from `marinelab/physics/thruster.py`, modified)
- Modify: `marinelab/core/__init__.py`
- Create: `tests/test_thruster.py`

**Changes vs original:** precompute scalar tau tensors in `__init__` so `apply_dynamics` does not build them every step; behavior identical.

- [ ] **Step 1: Write the failing test** in `tests/test_thruster.py`.

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /workspace/marinelab && python -m pytest tests/test_thruster.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marinelab.core.thruster'`.

- [ ] **Step 3: Create `marinelab/core/thruster.py`** by copying then editing.

Run first: `cp marinelab/physics/thruster.py marinelab/core/thruster.py`

Then edit `marinelab/core/thruster.py`:

(5a) Change the import of the cfg. Change:
```python
from marinelab.assets import ThrusterCfg
```
to:
```python
from marinelab.assets import ThrusterCfg
```
(unchanged — kept for clarity; the alias line below is removed).

(5b) Remove the backward-compat alias (dead per audit). Delete:
```python
# Alias for backward compatibility (ThrusterModelCfg was the old name)
ThrusterModelCfg = ThrusterCfg

__all__ = ["ThrusterModel", "ThrusterCfg", "ThrusterModelCfg"]
```
and replace with:
```python
__all__ = ["ThrusterModel", "ThrusterCfg"]
```

(5c) Precompute tau in `__init__`. After the `self._allocation_matrix = ...` assignment, add:
```python
        # Precompute time-constant tensors once (avoid per-step allocation).
        self._tau_up_scalar = torch.tensor(cfg.time_constant_up, dtype=torch.float32, device=device)
        self._tau_down_scalar = torch.tensor(cfg.time_constant_down, dtype=torch.float32, device=device)
```

(5d) Use cached tau in `apply_dynamics`. Replace:
```python
        if self._thrust_coeff is not None:
            tau_up = self._time_constant_up.unsqueeze(-1)
            tau_down = self._time_constant_down.unsqueeze(-1)
        else:
            tau_up = self.cfg.time_constant_up
            tau_down = self.cfg.time_constant_down

        tau = torch.where(
            target_state > self._state,
            tau_up if isinstance(tau_up, torch.Tensor) else torch.tensor(tau_up, device=self.device),
            tau_down if isinstance(tau_down, torch.Tensor) else torch.tensor(tau_down, device=self.device),
        )
```
with:
```python
        if self._thrust_coeff is not None:
            tau_up = self._time_constant_up.unsqueeze(-1)
            tau_down = self._time_constant_down.unsqueeze(-1)
        else:
            tau_up = self._tau_up_scalar
            tau_down = self._tau_down_scalar

        tau = torch.where(target_state > self._state, tau_up, tau_down)
```

- [ ] **Step 4: Export from `core/__init__.py`.**

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .hydrodynamics import HydrodynamicsModel
from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia
from .thruster import ThrusterCfg, ThrusterModel

__all__ = [
    "HydrodynamicsModel",
    "HydroParams",
    "default_rigid_inertia",
    "OceanCurrent",
    "ThrusterModel",
    "ThrusterCfg",
]
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /workspace/marinelab && python -m pytest tests/test_thruster.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add marinelab/core/thruster.py marinelab/core/__init__.py tests/test_thruster.py
git commit -m "feat(core): move ThrusterModel into core, precompute tau, drop dead alias"
```

---

### Task 6: Move volume.py into core, update internal import

**Files:**
- Create: `marinelab/core/volume.py` (moved from `marinelab/utils/volume.py`)
- Modify: `marinelab/core/hydrodynamics.py:_resolve_volume` import path

- [ ] **Step 1: Move the file.**

Run: `git mv marinelab/utils/volume.py marinelab/core/volume.py`

- [ ] **Step 2: Update the import inside `core/hydrodynamics.py`.** In `_resolve_volume`, change:
```python
            from marinelab.utils.volume import compute_collision_volume
```
to:
```python
            from marinelab.core.volume import compute_collision_volume
```

- [ ] **Step 3: Keep `marinelab/utils/` working** — make `marinelab/utils/volume.py` a shim (some code/tests may import it). Create `marinelab/utils/volume.py`:
```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Deprecated: moved to marinelab.core.volume. Re-export for compatibility."""

from marinelab.core.volume import *  # noqa: F401,F403
from marinelab.core.volume import (  # noqa: F401
    VolumeInfo,
    compute_articulation_body_volumes,
    compute_collision_volume,
    compute_prim_volume,
)
```

- [ ] **Step 4: Verify import resolves (no Isaac Sim needed for the import line itself).**

Run: `cd /workspace/marinelab && python -c "import ast; ast.parse(open('marinelab/core/volume.py').read()); ast.parse(open('marinelab/utils/volume.py').read()); print('OK')"`
Expected: `OK` (syntax valid; runtime import needs pxr, skipped).

- [ ] **Step 5: Commit**

```bash
git add marinelab/core/volume.py marinelab/utils/volume.py marinelab/core/hydrodynamics.py
git commit -m "refactor(core): move volume util into core, leave utils shim"
```

---

## Phase 5 — physics/ shim

### Task 7: Convert `marinelab/physics/` to a deprecated re-export shim

This is what keeps constrained-albc's `from marinelab.physics import HydrodynamicsModel, ThrusterModel` working this phase.

**Files:**
- Delete: `marinelab/physics/hydrodynamics.py`, `marinelab/physics/thruster.py`
- Modify: `marinelab/physics/__init__.py`

- [ ] **Step 1: Delete the moved originals.**

Run: `git rm marinelab/physics/hydrodynamics.py marinelab/physics/thruster.py`

- [ ] **Step 2: Rewrite `marinelab/physics/__init__.py` as a shim.**

```python
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Deprecated shim. Import from marinelab.core instead.

This module re-exports the framework physics classes from marinelab.core so
existing `from marinelab.physics import ...` imports keep working. New code
should import from marinelab.core directly.
"""

from marinelab.assets import HydrodynamicsCfg, OceanCurrentCfg
from marinelab.core import HydrodynamicsModel, OceanCurrent, ThrusterCfg, ThrusterModel

__all__ = [
    "HydrodynamicsModel",
    "HydrodynamicsCfg",
    "OceanCurrentCfg",
    "OceanCurrent",
    "ThrusterModel",
    "ThrusterCfg",
]
```

- [ ] **Step 3: Verify the shim re-exports the same symbols** (syntax + symbol presence, mocking isaaclab/assets minimally).

Run: `cd /workspace/marinelab && python -c "
import sys, types, torch
sys.path.insert(0, 'tests')
import conftest  # installs isaaclab mocks
# minimal assets mock
a = types.ModuleType('marinelab.assets')
a.HydrodynamicsCfg = type('HydrodynamicsCfg', (), {})
a.OceanCurrentCfg = type('OceanCurrentCfg', (), {'max_velocity': (0,)*6, 'noise_scale': (0,)*6})
a.ThrusterCfg = type('ThrusterCfg', (), {})
sys.modules['marinelab.assets'] = a
import marinelab.core as core
import marinelab.physics as physics
assert physics.HydrodynamicsModel is core.HydrodynamicsModel
assert physics.ThrusterModel is core.ThrusterModel
print('shim OK')
"`
Expected: `shim OK`.

- [ ] **Step 4: Commit**

```bash
git add marinelab/physics/
git commit -m "refactor: make marinelab.physics a deprecated re-export shim over core"
```

---

## Phase 6 — Asset-level physics fixes (A1, A2)

### Task 8: Fix A1 gyroscopic double-count

**Files:**
- Modify: `marinelab/assets/bluerov/bluerov.py:123`
- Check: `marinelab/assets/hero_agent/hero_agent.py` for the same flag

- [ ] **Step 1: Read the hero_agent cfg** to see if it sets `enable_gyroscopic_forces` with `use_full_coriolis=True`.

Run: `cd /workspace/marinelab && grep -n "enable_gyroscopic_forces\|use_full_coriolis" marinelab/assets/hero_agent/hero_agent.py marinelab/assets/bluerov/bluerov.py`

- [ ] **Step 2: Fix bluerov.py.** In `BLUEROV_CFG` → `rigid_props`, change:
```python
            enable_gyroscopic_forces=True,
```
to:
```python
            # use_full_coriolis=True in HydrodynamicsCfg already applies the
            # rigid-body gyroscopic term omega x (I omega). Disable PhysX's own
            # to avoid double-counting (see hydrodynamics.py warning).
            enable_gyroscopic_forces=False,
```

- [ ] **Step 3: If hero_agent.py has the same `enable_gyroscopic_forces=True` with full coriolis,** apply the identical fix there. If it does not set the flag or uses `use_full_coriolis=False`, leave it and note why in the commit message.

- [ ] **Step 4: Commit**

```bash
git add marinelab/assets/bluerov/bluerov.py marinelab/assets/hero_agent/hero_agent.py
git commit -m "fix(physics): disable PhysX gyroscopic forces to stop double-count with full Coriolis (A1)"
```

---

### Task 9: Investigate + fix/escalate A2 allocation pitch/roll geometry

**Files:**
- Read: `marinelab/assets/bluerov/bluerov.py:100-113`, any BlueROV thruster-geometry reference in repo/docs

- [ ] **Step 1: Search the repo for BlueROV2-Heavy thruster geometry** (mesh/USD frame positions, comments, references) to determine the actual vertical-thruster (VL/VR) x and y offsets.

Run: `cd /workspace/marinelab && grep -rn "vert\|VL\|VR\|thruster.*pos\|moment arm\|0.12\|0.19" marinelab/assets/ docs/ 2>/dev/null | grep -v __pycache__ | head -30`

- [ ] **Step 2: Decide.** The roll row `(+0.1,-0.1)` implies VL/VR are separated in y. The pitch row `(+0.12,+0.12)` (same sign) then makes pitch a disguised heave (not independently controllable from the two verticals). Two outcomes:
  - **If evidence shows verticals share x (only y-separated):** pitch authority must come from the 4 horizontal thrusters or is intentionally absent. Document this as a known modeling limitation in a code comment; do NOT silently flip a sign. Leave the matrix and add a comment explaining pitch is not independently actuated by the verticals.
  - **If evidence shows verticals are fore/aft (x-separated):** then the roll row is the inconsistent one. This is a genuine ambiguity.

- [ ] **Step 3: ESCALATE if unresolved.** Per spec §6a, if the geometry cannot be confirmed from repo evidence, STOP and ask the user (do not guess). Report the two possibilities and the evidence found. Only proceed to a matrix change with user confirmation.

- [ ] **Step 4: Commit** (whatever the resolution — comment-only or confirmed fix).

```bash
git add marinelab/assets/bluerov/bluerov.py
git commit -m "docs/fix(physics): clarify BlueROV vertical-thruster pitch/roll allocation (A2)"
```

---

## Phase 7 — bluerov restructure (B1, B2, B4, B6 + dead code + vectorize)

### Task 10: Fix B2 attitude observation_space and B4 sign(w)

**Files:**
- Modify: `marinelab/tasks/bluerov/attitude_env_cfg.py:53`
- Modify: `marinelab/tasks/bluerov/tasks/attitude_task.py:113-119`
- Create: `tests/test_attitude_task.py`

- [ ] **Step 1: Write the failing test** in `tests/test_attitude_task.py`. This tests the math of the attitude-error helper in isolation (no Isaac Sim).

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for the attitude-error sign-correction (B4)."""

import torch


def attitude_error_signed(error_quat: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the corrected attitude error."""
    w = error_quat[:, 0:1]
    vec = error_quat[:, 1:4]
    return 2.0 * torch.sign(w) * vec


def test_sign_correction_keeps_error_continuous_past_180():
    # q and -q are the same rotation; corrected error must agree.
    q = torch.tensor([[0.2, 0.6, 0.0, 0.0]])
    q_neg = -q
    e1 = attitude_error_signed(q)
    e2 = attitude_error_signed(q_neg)
    assert torch.allclose(e1, e2, atol=1e-6)


def test_zero_error_for_identity():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(attitude_error_signed(q), torch.zeros(1, 3))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /workspace/marinelab && python -m pytest tests/test_attitude_task.py -v`
Expected: PASS actually (the reference fn is self-contained) — this test documents the intended behavior. If it fails, fix the reference. Then proceed to apply the same logic in the real file.

- [ ] **Step 3: Fix `attitude_task.py`.** Replace:
```python
        # Convert to axis-angle representation for error
        # For small angles: theta ~= 2 * (x, y, z) components
        attitude_error = 2.0 * error_quat[:, 1:4]  # Approximate for small angles

        return attitude_error
```
with:
```python
        # Convert to axis-angle error. Sign-correct by the scalar part so the
        # error stays continuous across the quaternion double cover (q and -q
        # are the same rotation); without this the signal flips/shrinks near
        # 180 deg where the gradient matters most.
        w = error_quat[:, 0:1]
        vec = error_quat[:, 1:4]
        attitude_error = 2.0 * torch.sign(w) * vec

        return attitude_error
```

- [ ] **Step 4: Fix `attitude_env_cfg.py:53`.** Change:
```python
    observation_space: int = 12
```
to:
```python
    # Env emits a fixed 18-D observation regardless of task (see BlueROVEnv).
    observation_space: int = 18
```
(Also fix the inline comment on that line if it claims 12.)

- [ ] **Step 5: Run**

Run: `cd /workspace/marinelab && python -m pytest tests/test_attitude_task.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add marinelab/tasks/bluerov/attitude_task.py marinelab/tasks/bluerov/tasks/attitude_task.py marinelab/tasks/bluerov/attitude_env_cfg.py tests/test_attitude_task.py
git commit -m "fix(bluerov): attitude obs_space 12->18 (B2) and sign-correct attitude error (B4)"
```

### Task 11: Add `randomize` helpers to bluerov events using core API, fix B1, vectorize

**Files:**
- Modify: `marinelab/tasks/bluerov/mdp/events.py` (`randomize_hydrodynamics` rewritten to use `scale_parameters`/`set_parameters`)
- Modify: `marinelab/tasks/bluerov/bluerov_env.py:388-409` (call events functions, not the missing method)

**Goal:** `bluerov_env._reset_idx` currently calls `self._hydro.randomize_parameters(...)` which does not exist (B1). Route through `events.randomize_hydrodynamics`, and rewrite that function to use the new core `scale_parameters` + `set_parameters` API (replacing the old direct property index-writes).

- [ ] **Step 1: Rewrite `events.py::randomize_hydrodynamics`** to use the core API. Replace the body after the `hydro = env._hydro` / `device` setup (lines ~76-122) with:
```python
    hydro = env._hydro
    num_envs = len(env_ids)
    device = env.device

    # Scale-based randomization via the core API (base * uniform(lo, hi)).
    hydro.scale_parameters(
        env_ids,
        added_mass=added_mass_scale,
        linear_damping=linear_damping_scale,
        quadratic_damping=quadratic_damping_scale,
        volume=volume_scale,
        rigid_body_inertia=inertia_scale,
    )

    # Offset-based randomization for CoB / CoG (base + uniform offset).
    base = hydro.base_parameters

    def _offset(center, ranges):
        out = center[env_ids].clone()
        for axis, (lo, hi) in enumerate(ranges):
            out[:, axis] = out[:, axis] + (torch.rand(num_envs, device=device) * (hi - lo) + lo)
        return out

    hydro.set_parameters(
        env_ids,
        center_of_buoyancy=_offset(base.center_of_buoyancy, (cob_offset_x, cob_offset_y, cob_offset_z)),
        center_of_gravity=_offset(base.center_of_gravity, (cog_offset_x, cog_offset_y, cog_offset_z)),
    )
```
(Leave the function signature and docstring unchanged.)

- [ ] **Step 2: Fix `bluerov_env.py` reset path (B1).** Replace:
```python
        if self.cfg.randomization.enable:
            self._hydro.randomize_parameters(
                env_ids=env_ids,
                added_mass_scale=self.cfg.randomization.added_mass_scale,
                linear_damping_scale=self.cfg.randomization.linear_damping_scale,
                quadratic_damping_scale=self.cfg.randomization.quadratic_damping_scale,
                volume_scale=self.cfg.randomization.volume_scale,
                cob_offset_z=self.cfg.randomization.cob_offset_z,
                inertia_scale=self.cfg.randomization.inertia_scale,
            )
            self._thruster.randomize_parameters(
                env_ids=env_ids,
                thrust_coeff_scale=self.cfg.randomization.thrust_coefficient_scale,
                time_constant_scale=self.cfg.randomization.time_constant_scale,
            )
```
with:
```python
        if self.cfg.randomization.enable:
            from .mdp.events import randomize_hydrodynamics

            r = self.cfg.randomization
            randomize_hydrodynamics(
                self,
                env_ids,
                added_mass_scale=r.added_mass_scale,
                linear_damping_scale=r.linear_damping_scale,
                quadratic_damping_scale=r.quadratic_damping_scale,
                volume_scale=r.volume_scale,
                cob_offset_x=r.cob_offset_x,
                cob_offset_y=r.cob_offset_y,
                cob_offset_z=r.cob_offset_z,
                cog_offset_x=r.cog_offset_x,
                cog_offset_y=r.cog_offset_y,
                cog_offset_z=r.cog_offset_z,
                inertia_scale=r.inertia_scale,
            )
            self._thruster.randomize_parameters(
                env_ids=env_ids,
                thrust_coeff_scale=r.thrust_coefficient_scale,
                time_constant_scale=r.time_constant_scale,
            )
```

- [ ] **Step 3: Commit** (smoke-tested in Task 13; here we commit the wiring).

```bash
git add marinelab/tasks/bluerov/mdp/events.py marinelab/tasks/bluerov/bluerov_env.py
git commit -m "fix(bluerov): route DR through core scale/set API, fix missing randomize_parameters (B1, B3)"
```

### Task 12: Delete dead code (~180 lines) + B6 volume warning + reward/obs buffer caching

**Files:**
- Modify: `marinelab/tasks/bluerov/bluerov_env_cfg.py` (delete `EventCfg`, `events` field, legacy fields)
- Modify: `marinelab/tasks/bluerov/bluerov_env.py` (`_create_task_from_config` simplify, cache `_up_w`, total-force buffers, drop dead pose duplication if now unused)
- Modify: `marinelab/tasks/bluerov/tasks/task_base.py` (delete `step`)
- Modify: `marinelab/tasks/bluerov/rewards/reward_manager.py` (delete `get_term_value`, `set_term_weight`)
- Modify: `marinelab/tasks/bluerov/rewards/reward_terms.py` (delete linear `orientation_upright`)
- Modify: `marinelab/tasks/bluerov/rewards/__init__.py` and `tasks/bluerov/__init__.py` exports as needed
- Modify: `marinelab/core/hydrodynamics.py` `_resolve_volume` (B6)

- [ ] **Step 1: Delete `EventCfg` and the `events` field** in `bluerov_env_cfg.py` (lines ~78-138 and the `events: EventCfg | None = None` field). Remove the now-unused `EventTerm` import and `bluerov_events` import if no longer referenced.

- [ ] **Step 2: Delete the now-orphaned event functions** in `mdp/events.py` that are unreachable: `randomize_thruster_params`, `randomize_ocean_current`, `randomize_robot_pose` (keep `randomize_hydrodynamics` — it's now live via Task 11). Verify with grep that none are referenced elsewhere in marinelab before deleting.

Run: `cd /workspace/marinelab && grep -rn "randomize_thruster_params\|randomize_ocean_current\|randomize_robot_pose" marinelab/ | grep -v __pycache__`
Expected: only the definitions in events.py (no other callers) — safe to delete.

- [ ] **Step 3: Simplify `_create_task_from_config`** in `bluerov_env.py`. Replace the whole method body (the None/backward-compat branches) with:
```python
        task_cfg = self.cfg.task
        if isinstance(task_cfg, AttitudeTaskCfg):
            return AttitudeTask(cfg=task_cfg, num_envs=self.num_envs, device=self.device)
        return HoverTask(cfg=task_cfg, num_envs=self.num_envs, device=self.device)
```
Remove the now-unused `TaskBaseCfg` import if unused, and the legacy `goal_pos_range`/`initial_height` fields on `BlueROVEnvCfg`.

- [ ] **Step 4: Delete `TaskBase.step`** (task_base.py), `RewardManager.get_term_value`/`set_term_weight` (reward_manager.py), and linear `orientation_upright` (reward_terms.py). Update `rewards/__init__.py` to drop the `orientation_upright` export.

- [ ] **Step 5: Cache `_up_w` and force buffers** in `bluerov_env.py`. In `__init__`, add:
```python
        self._up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
```
In `_get_observations`, replace the per-step `up_w = torch.tensor(...)` with `up_b = quat_apply_inverse(root_quat_w, self._up_w)`.

- [ ] **Step 6: B6 volume warning.** In `core/hydrodynamics.py` `_resolve_volume`, after computing `volume` from auto-calc, if it returns the 0.01 fallback, make the warning explicit with the resolved value (it already warns; ensure the message includes the fallback magnitude). Minimal: leave the existing warning but confirm it prints the fallback. No behavior change required beyond message clarity.

- [ ] **Step 7: Vectorize the initial-pose randomization** in `bluerov_env.py` `_reset_idx`. Replace the six separate `torch.rand(num_reset, device=...) * (hi - lo) + lo` blocks (pos_x/y/z, roll/pitch/yaw) with two batched samples. Replace the position block with:
```python
            pos_lo = torch.tensor(
                [r.position_x_range[0], r.position_y_range[0], r.position_z_range[0]], device=self.device
            )
            pos_hi = torch.tensor(
                [r.position_x_range[1], r.position_y_range[1], r.position_z_range[1]], device=self.device
            )
            pos = torch.rand(num_reset, 3, device=self.device) * (pos_hi - pos_lo) + pos_lo
            default_root_state[:, 0] += pos[:, 0]
            default_root_state[:, 1] += pos[:, 1]
            default_root_state[:, 2] = pos[:, 2]
```
and the orientation block with:
```python
            rot_lo = torch.tensor(
                [r.roll_range[0], r.pitch_range[0], r.yaw_range[0]], device=self.device
            )
            rot_hi = torch.tensor(
                [r.roll_range[1], r.pitch_range[1], r.yaw_range[1]], device=self.device
            )
            rpy = torch.rand(num_reset, 3, device=self.device) * (rot_hi - rot_lo) + rot_lo
            random_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            default_root_state[:, 3:7] = quat_mul(default_root_state[:, 3:7], random_quat)
```
where `r = self.cfg.randomization` (bind it once at the top of the `if self.cfg.randomization.enable:` pose block). Behavior is identical (same uniform ranges); only the allocation count drops from 6 to 2.

- [ ] **Step 8: Verify syntax of all touched files.**

Run: `cd /workspace/marinelab && python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('marinelab/**/*.py', recursive=True)]; print('all parse OK')"`
Expected: `all parse OK`.

- [ ] **Step 9: Commit**

```bash
git add marinelab/
git commit -m "refactor(bluerov): remove ~180 lines dead code, simplify task factory, cache hot-path buffers, vectorize pose DR, clarify volume fallback (B6)"
```

---

## Phase 8 — bluerov smoke test

### Task 13: bluerov smoke test (obs dim, DR no-crash) — logic-level, Isaac-Sim-free

Full env instantiation needs Isaac Sim. Instead, unit-test the two regressed behaviors at the logic level: the DR routing calls the core API without the missing-method crash, and the observation concatenation width is 18.

**Files:**
- Create: `tests/test_bluerov_smoke.py`

- [ ] **Step 1: Write the test.** Use the core models directly to prove `scale_parameters`/`set_parameters` accept the bluerov DR call shape (the exact path Task 11 wired), and assert the obs-width arithmetic.

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logic-level smoke tests for the bluerov DR path and observation width."""

import torch

from marinelab.core.hydrodynamics import HydrodynamicsModel


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


class _CurCfg:
    max_velocity = (0.0,) * 6
    noise_scale = (0.0,) * 6


def test_dr_path_no_missing_method_crash():
    # Mirrors the bluerov reset DR call shape (scale + offset) via the core API.
    m = HydrodynamicsModel(8, "cpu", _Cfg(), current_cfg=_CurCfg(), dt=0.005)
    env_ids = torch.arange(8)
    m.scale_parameters(
        env_ids,
        added_mass=(0.5, 1.0),
        linear_damping=(0.5, 1.0),
        quadratic_damping=(0.5, 1.0),
        volume=(0.9, 1.1),
        rigid_body_inertia=(0.8, 1.2),
    )
    base = m.base_parameters
    cob = base.center_of_buoyancy[env_ids].clone()
    cob[:, 2] = cob[:, 2] + (torch.rand(8) * 0.04 - 0.02)
    m.set_parameters(env_ids, center_of_buoyancy=cob)
    # buoyancy force stays positive and finite
    assert torch.all(torch.isfinite(m.buoyancy_force))
    assert torch.all(m.buoyancy_force > 0)


def test_observation_width_is_18():
    # pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+goal_obs(3)+up(2)
    assert 3 + 4 + 3 + 3 + 3 + 2 == 18
```

- [ ] **Step 2: Run**

Run: `cd /workspace/marinelab && python -m pytest tests/test_bluerov_smoke.py -v`
Expected: PASS (2 tests).

- [ ] **Step 3: Run the FULL suite.**

Run: `cd /workspace/marinelab && python -m pytest tests/ -v`
Expected: ALL PASS (parameters 3, ocean_current 5, hydrodynamics 12, thruster 4, attitude 2, bluerov_smoke 2 = 28 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_bluerov_smoke.py
git commit -m "test(bluerov): logic-level smoke for DR path and obs width"
```

---

## Phase 9 — README framework guide

### Task 14: Rewrite README as a framework guide

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README** to preserve install/setup content.

Run: `cd /workspace/marinelab && cat README.md`

- [ ] **Step 2: Rewrite README** with these sections (keep existing install steps; add the framework positioning + workflow). Headings in English, prose may be Korean, no emojis, no AI attribution.

Required sections:
1. **What is marinelab** — general-purpose UUV environment framework on Isaac Lab. `marinelab.core` = stable reusable API; `marinelab.tasks.bluerov` = reference example.
2. **Architecture** — the `core/ assets/ tasks/` layout and the boundary principle (core is general; tasks are examples; researchers extend like constrained-albc extends marinelab).
3. **Core API reference** — `HydrodynamicsModel` (compute_forces, update_physx_state, get_parameters/set_parameters/scale_parameters, base_parameters, current, reset, update_buoyancy_force), `ThrusterModel` (apply_dynamics, compute_wrench, randomize_parameters, state, reset), `OceanCurrent` (velocity_w, max_velocity, set, add_drift, reset), `HydroParams`.
4. **Researcher workflow** — a short code snippet showing: build OceanCurrent, inject into one or more HydrodynamicsModel, run compute_forces, do DR via `scale_parameters`/`set_parameters` (no private access). State that constrained-albc is the real-world example and bluerov is the minimal one.
5. **Domain randomization** — explain setter (absolute) vs scale (convenience) and the base-cache.
6. **Migration note** — `marinelab.physics` is a deprecated shim; import from `marinelab.core`.

- [ ] **Step 3: Verify README has no emojis and no AI attribution.**

Run: `cd /workspace/marinelab && grep -nE "Generated with|Created by AI|Co-Authored|🤖|😀" README.md || echo "clean"`
Expected: `clean`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as marinelab framework guide with core API + researcher workflow"
```

---

## Phase 10 — Final verification

### Task 15: Full suite + CHANGELOG + import-integrity sweep

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run the full test suite.**

Run: `cd /workspace/marinelab && python -m pytest tests/ -v`
Expected: ALL PASS (28 tests).

- [ ] **Step 2: Syntax-parse the whole package.**

Run: `cd /workspace/marinelab && python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('marinelab/**/*.py', recursive=True)]; print('all parse OK')"`
Expected: `all parse OK`.

- [ ] **Step 3: Grep for leftover references to moved modules** inside marinelab (should only be the shims).

Run: `cd /workspace/marinelab && grep -rn "marinelab.physics.hydrodynamics\|marinelab.physics.thruster\|marinelab.utils.volume" marinelab/ | grep -v __pycache__`
Expected: empty or only the shim files.

- [ ] **Step 4: Update CHANGELOG.md** with a new entry (Removed / Added / Changed / Fixed / Verification / Notes). Include: core/ framework split, setter-based DR API, OceanCurrent component, physics shim, A1/B1/B2/B4/B6 fixes, ~180 lines dead code removed, hot-path buffer caching, A2 resolution (whatever it was), pytest suite added. Note in **Notes** that constrained-albc migration is a separate later phase and that retraining is expected due to the A1 behavior change.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for marinelab framework-ization (core API, physics fixes, tests)"
```

- [ ] **Step 6: Report to user** — summarize what changed, that constrained-albc is untouched this phase (shim keeps it importable), that the A1 fix changes dynamics (retraining expected), the A2 resolution, and that the albc migration is the next coordinated phase. Do NOT push (ask first).

---

## Notes for the executor

- **NEVER edit `/workspace/constrained-albc` or `/workspace/isaaclab` in this plan.** The shim (Task 7) is what keeps albc working. albc migration is a separate phase.
- **If A2 (Task 9) cannot be resolved from evidence, STOP and ask the user.** Do not guess thruster geometry.
- **Behavior-identical guard:** Tasks 3, 5 move physics code. The regression tests (Task 4) and the known-value tests must pass to prove the move + buffer-caching didn't change numbers. If a number changes, the move was not faithful — fix before proceeding.
- **Commit after every task.** Do not batch.
- **Tests run without Isaac Sim** via `tests/conftest.py`. If a test needs `isaaclab.utils.math.X` not yet mocked, add it to conftest with a correct implementation (not the generic mock).
