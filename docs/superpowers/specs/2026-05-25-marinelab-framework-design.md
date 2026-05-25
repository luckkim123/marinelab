# marinelab Framework-ization — Design Spec

**Date:** 2026-05-25
**Status:** Approved (brainstorming complete)
**Scope:** This spec covers **marinelab only**. The constrained-albc migration is a separate, later phase (see §7).

## 1. Goal

Turn `marinelab` into a **general-purpose underwater-vehicle (UUV) environment framework**, mirroring how Isaac Lab provides a general simulation layer and `constrained-albc` overrides the details. `marinelab.core` becomes the stable, reusable API; `marinelab.tasks.bluerov` becomes a reference example of how to use it correctly. The framework workflow is documented in the README so future researchers extend it the intended way (config + setter API), never by reaching into private buffers.

Re-training of constrained-albc is already planned, so behavior changes (physics corrections) are acceptable.

## 2. Context & Constraints

- **Language:** Python (PyTorch tensors, GPU-parallel over 4096 envs, wired to PhysX). C++ rejected: pybind/GPU marshalling would hurt perf and break the in-place property contract. Efficiency wins come from removing per-step tensor reallocation, not from a language change.
- **Live API consumer:** `constrained-albc` is tightly coupled to marinelab's current physics surface (50+ call sites, in-place index writes to 12 properties, private access to `_current_velocity`/`_max_current_vel`/`_dr_base_cache`, two hydro instances `_hydro`+`_buoy_hydro`, instantiate-only — no subclassing). It is being edited concurrently in a separate session.
- **CRITICAL operational constraint:** Because albc is edited concurrently, this work completes marinelab first with a backward-compat shim so existing albc imports keep working, and defers the albc migration to a coordinated later phase. This avoids cross-session merge conflicts.
- **isaaclab stays pristine** (project rule). All work is in marinelab.
- Korean for user communication; English for code/markdown headings/code blocks; no emojis; no AI-attribution; commit only on explicit request.

## 3. Architecture — `marinelab/core/` split

```
marinelab/
├── core/                          # general-purpose framework (researchers import this)
│   ├── hydrodynamics.py           #   HydrodynamicsModel (Fossen) + setter-based param API
│   ├── thruster.py                #   ThrusterModel + allocation
│   ├── ocean_current.py           #   OceanCurrent (NEW, split out) — shared current source
│   ├── parameters.py              #   HydroParams dataclass + base-cache + inertia-fallback helper
│   └── volume.py                  #   USD volume util (moved from utils/)
├── assets/                        # vehicle cfgs (BlueROV, HeroAgent) — minimal change
│   ├── uuv_cfg.py                 #   base cfgs (HydrodynamicsCfg/ThrusterCfg/OceanCurrentCfg)
│   ├── bluerov/  hero_agent/
├── tasks/                         # reference usage example
│   └── bluerov/                   #   "how to use core correctly" reference
├── physics/                       # backward-compat shim: re-exports from core (deprecated)
└── __init__.py
```

**Boundary principle (documented in README):**
- `marinelab.core` = stable, general API (like Isaac Lab's sim layer). Researchers import it and tune via cfg + setter API.
- `marinelab.tasks.bluerov` = reference example that overrides/composes core (like constrained-albc).
- Researcher workflow: build your env on top of `core`, adjust DR/current via cfg + setter API. No private access needed.

**Backward-compat shim:** `marinelab.physics` re-exports `HydrodynamicsModel`, `ThrusterModel`, cfgs from `marinelab.core` (marked deprecated). Existing `from marinelab.physics import ...` (used by albc) keeps working until the albc migration phase. New code uses `marinelab.core`.

## 4. Parameter & DR API (setter-based, base-cache built in)

Replaces the leaky `hydro.linear_damping[ids] = base*scale` index-writes and albc's externally-attached `_dr_base_cache`.

```python
# parameters.py
@dataclass
class HydroParams:
    """Named bundle of per-env hydrodynamic parameter tensors."""
    added_mass: Tensor | None = None            # (N, 6)
    linear_damping: Tensor | None = None        # (N, 6)
    quadratic_damping: Tensor | None = None     # (N, 6)
    volume: Tensor | None = None                # (N,)
    water_density: Tensor | None = None         # (N,)
    center_of_buoyancy: Tensor | None = None    # (N, 3)
    center_of_gravity: Tensor | None = None     # (N, 3)
    rigid_body_inertia: Tensor | None = None    # (N, 3)
    body_mass: Tensor | None = None             # (N,)

def default_rigid_inertia(cfg) -> list[float]:
    """Single source for the added_mass[3:6]*0.5 fallback (was duplicated 4x)."""
```

```python
# HydrodynamicsModel new public surface
hydro.base_parameters -> HydroParams              # computed once from cfg in __init__, immutable
hydro.get_parameters(env_ids=None) -> HydroParams # current per-env values (for privileged obs)
hydro.set_parameters(env_ids, **HydroParams_fields)   # absolute values; None fields untouched;
                                                       # auto-calls update_buoyancy_force when
                                                       # volume/density/inertia change
hydro.scale_parameters(env_ids, added_mass=(lo,hi), ...)  # convenience: base * uniform(lo,hi)
```

**Design points:**
1. **Base-cache owned by the framework.** `base_parameters` computed once in `__init__`. Scale-based DR (`base*scale`) handled by `scale_parameters`. albc's `_HydroBaseCache`/`_dr_base_cache` becomes framework responsibility.
2. **`HydroParams` bundles reads and writes.** Privileged-obs extraction reads `hydro.get_parameters().volume` etc. instead of indexing 12 separate properties.
3. **setter = absolute values; scale = convenience.** A custom sampler (e.g., DORAEMON) computes absolute values and injects via `set_parameters`. `scale_parameters` is the simple uniform-scale path.
4. **Backward-compat:** existing `@property` getters (`linear_damping`, etc.) kept **read-only** (deprecated). Index-write paths migrate to the setter. Reads preserved.
5. **inertia fallback unified** into `default_rigid_inertia` (was in hydrodynamics.py `_init`, `_validate`, marinelab events.py, albc events.py).

## 5. Ocean Current — `OceanCurrent` component + shared source

Splits current out of hydro (was `_current_velocity` direct-write + `_max_current_vel` direct-read + manual two-hydro sharing).

```python
# core/ocean_current.py
class OceanCurrent:
    """World-frame 6-DOF current source. Shareable across multiple hydro models."""
    def __init__(self, num_envs, device, cfg: OceanCurrentCfg): ...
    velocity_w -> Tensor       # (N, 6) world frame (was _current_velocity)
    max_velocity -> Tensor     # (6,)   (was _max_current_vel)
    def set(self, env_ids, velocity=None, strength=None): ...   # was set_ocean_current
    def add_drift(self, delta): ...      # inject time-varying current (OU drift etc.)
    def reset(self, env_ids=None): ...
```

```python
# Dependency injection into hydro:
current = OceanCurrent(n, dev, cfg.ocean_current)
hydro   = HydrodynamicsModel(..., current=current)

# Shared source (albc's _hydro + _buoy_hydro):
main_hydro = HydrodynamicsModel(..., current=current)
buoy_hydro = HydrodynamicsModel(..., current=current)   # same object -> no manual copy
```

**Design points:**
1. **Shared source is the official pattern.** Injecting the same `OceanCurrent` replaces albc's per-step `_buoy_hydro._current_velocity = _hydro._current_velocity` copy. Private access ends.
2. **OU drift stays researcher-owned; framework provides hooks.** Domain-specific time-varying current is injected via `current.add_drift(delta)` / `current.set(...)`. `velocity_w`/`max_velocity` are public so the researcher can compute clamps.
3. **Backward-compat shim:** `hydro.set_ocean_current(...)` delegates to `self._current.set(...)`. Reads of `hydro._current_velocity` migrate to `hydro.current.velocity_w` (or direct `OceanCurrent` reference).
4. **angular current:** 6-DOF kept for cfg compatibility (default 0, inert). README notes angular components are usually 0 (non-standard).

## 6. Bug fixes, bluerov restructure, efficiency, tests, README

### 6a. Diagnosed bug / physics fixes (behavior change OK — retraining planned)

| ID | Fix | Risk |
|----|-----|------|
| A1 | Gyroscopic double-count: set `enable_gyroscopic_forces=False` in bluerov.py (keep use_full_coriolis=True). Check hero_agent cfg too. | shared, albc retrain |
| A2 | Allocation pitch/roll row geometric contradiction: verify against actual BlueROV2-Heavy geometry, then fix. **If uncertain, flag separately and ask the user** rather than guessing. | shared |
| B1 | bluerov DR crash (`self._hydro.randomize_parameters` does not exist): route `_reset_idx` through new `scale_parameters`/`set_parameters`. Resolves duplication C1 simultaneously. | SAFE (bluerov) |
| B2 | attitude `observation_space=12` -> `18`. | SAFE |
| B4 | attitude error missing `sign(w)`: `2.0 * sign(w) * vec`. | SAFE |
| B6 | volume-0 silent fallback: emit explicit warning/error instead of silent 0.01 m^3. | SAFE |

### 6b. bluerov as reference example (all SAFE, bluerov-internal)
- DR/pose-reset inline in `_reset_idx` -> calls to core API (removes duplication, fixes B1).
- Vectorize per-axis sampling ((N,3) in one call).
- Delete ~180 lines dead code: `EventCfg` + `events` field, `TaskBase.step`, `RewardManager.get_term_value`/`set_term_weight`, linear `orientation_upright`, `ThrusterModelCfg` alias, legacy task branch in `_create_task_from_config` + legacy `goal_pos_range`/`initial_height`.
- Cache reward/obs constant buffers (remove hot-path allocation).

### 6c. Efficiency (buffer caching — core must verify behavior-identical)
- Cache constant `up_dir_w` in `__init__` (runs 2x per env in albc research loop).
- Pre-compute thruster `tau` (avoid per-step `torch.tensor` + `torch.where` materialization).
- All guarded by **numerically-identical** pytest before/after.

### 6d. Tests (pytest, no Isaac Sim dependency — MockModule pattern per project rule)
- core physics unit tests with known-value regression: C_A sign, damping clamp, buoyancy/restoring, thruster discretization, current frame transform.
- setter API: set/scale/get round-trip; base-cache immutability.
- bluerov: obs_dim==18; DR `enable=True` no crash (smoke).
- "behavior-identical before/after restructure" gate.

### 6e. README framework guide
- Position marinelab as a general UUV framework.
- core API reference (HydrodynamicsModel / ThrusterModel / OceanCurrent + setter/scale/get).
- "Researcher workflow" section: import core -> own env -> tune DR/current via cfg + setter (constrained-albc as the real example, bluerov as the minimal example).
- State extension points explicitly; emphasize private access is unnecessary.

## 7. Phase separation (because of concurrent albc edits)

1. **This work = marinelab only.** core split, new API, bug fixes, bluerov restructure, tests, README. The `physics -> core` re-export shim keeps existing albc imports working -> no immediate conflict with the albc session.
2. **albc migration = separate later phase**, after the albc session's work stabilizes, done in coordination. Migrates the 5 albc files (events / observations / logging / albc_env / tdc) to the new core API and removes private-access patterns. Retraining is already planned.

## 8. Non-goals
- C++ / CUDA kernels (rejected; revisit only if profiling proves a bottleneck).
- Decoupling albc from marinelab via abstract interfaces (overkill; instantiate-only coupling is fine once the API is clean).
- New tasks beyond bluerov Hover/Attitude (YAGNI; no Track task exists).
- Touching isaaclab (pristine fork).

## 9. Success criteria
- `marinelab.core` exposes a clean setter/get DR API + `OceanCurrent`; no private access required to use it.
- `marinelab.physics` shim keeps existing albc imports importable (albc unbroken this phase).
- All diagnosed CRITICAL/HIGH bugs (A1, B1, B2, B4) fixed; A2 fixed or escalated.
- ~180 lines dead code removed; per-step hot-path allocations cached.
- pytest suite passes, including behavior-identical regression for refactored physics.
- README documents the framework workflow for future researchers.
