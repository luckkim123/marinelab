# marinelab

![version](https://img.shields.io/badge/version-0.2.0-blue)
![license](https://img.shields.io/badge/license-BSD--3--Clause-green)

A general-purpose underwater-vehicle (UUV) environment framework on top of
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).

marinelab separates a reusable physics core from the example environments built
on it:

- **`marinelab.core`** — the stable, reusable framework API: a Fossen-style
  hydrodynamics model, a first-order thruster model, an injectable ocean-current
  component, and a setter-based domain-randomization interface. Researchers build
  on this layer.
- **`marinelab.tasks.bluerov`** — a reference example: the BlueROV2 hover and
  attitude environments registered into the Isaac Lab Gym registry. It shows how
  to wire the core models into a `DirectRLEnv`.
- **`marinelab.assets`** — UUV configs and USD meshes (BlueROV2 and Hero Agent,
  tracked with Git LFS).

The intent is that downstream research overlays (for example, the private
`constrained-albc`) extend marinelab the same way marinelab extends Isaac Lab:
import the core models, configure them, and randomize through the public API —
without reaching into private buffers.

---

## Architecture

```
isaaclab          (clean upstream fork; Docker base image)
    ^  pip install / import isaaclab
marinelab         (this repo; public junior deploy layer)
    ^  pip install / import marinelab
constrained-albc  (private research overlay; optional)
```

`marinelab` depends only on `isaaclab`. It does not depend on `constrained-albc`.
There are no upward dependencies.

Inside marinelab the boundary is just as deliberate:

```
marinelab/
  core/     general, reusable physics — the framework API (stable)
  assets/   UUV configs + USD meshes
  tasks/    example environments built on core (bluerov is the reference)
```

`core/` knows nothing about any specific vehicle or task. `tasks/bluerov` is one
concrete consumer of `core/`; a research overlay is another. Keeping vehicles and
reward design out of `core/` is what lets multiple overlays share one physics
implementation.

---

## Quickstart

### Prerequisites

- Docker with NVIDIA runtime, or an existing editable isaaclab install.
- Git LFS (`apt-get install git-lfs` if absent).

### Step 1 — clone and pull meshes

```bash
git clone https://github.com/luckkim123/marinelab.git /workspace/marinelab
cd /workspace/marinelab
git lfs install
git lfs pull
```

### Step 2 — install into the isaaclab environment

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install -e /workspace/marinelab
```

`isaaclab` and `isaaclab_assets` are **not** pip dependencies of marinelab.
They are provided by the isaaclab editable install already present in the environment.

### Step 3 — verify

Run the BlueROV hover environment for a few steps to confirm registration:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

> Note: `import marinelab` outside a launched Isaac Sim app will raise
> `No module named pxr` because the environment transitively needs the USD
> runtime. Always verify through `./isaaclab.sh -p` or the Docker container.

---

## Core API reference

All of the following are importable from `marinelab.core`.

### `HydrodynamicsModel`

Fossen 6-DOF rigid-body hydrodynamics: added mass, linear and quadratic damping,
and buoyancy/restoring forces. Holds per-environment parameter buffers.

```python
HydrodynamicsModel(num_envs, device, cfg, current_cfg=None, dt=0.01,
                   articulation_prim_path=None, current=None)
```

- `compute_forces(root_lin_vel_w, root_ang_vel_w, root_quat_w)` — body-frame force
  and torque for the current state (includes the injected ocean current).
- `update_physx_state(body_com_acc_w, root_quat_w)` — feed PhysX acceleration for
  the added-mass force path (only when `apply_added_mass`).
- `get_parameters(env_ids=None)` — read the current parameters as a `HydroParams`.
- `set_parameters(env_ids, **fields)` — absolute writes; updates buoyancy
  automatically when volume or water density changes.
- `scale_parameters(env_ids, **ranges)` — convenience: `base * uniform(lo, hi)`.
- `update_buoyancy_force(env_ids=None)` — recompute the buoyancy magnitude from
  volume and density.
- `reset(env_ids=None)` — restore parameters to base and reset the current.
- Properties: `base_parameters`, `current`, `buoyancy_force`, `volume`,
  `added_mass_matrix`, `linear_damping`, `quadratic_damping`,
  `center_of_buoyancy`, `center_of_gravity`, `water_density`, `body_mass`,
  `rigid_body_inertia`, `apply_added_mass`.

### `ThrusterModel`

First-order thruster dynamics with an allocation matrix.

```python
ThrusterModel(cfg, num_envs, device, enable_randomization=False)
```

- `apply_dynamics(commands, dt)` — advance the per-thruster first-order state.
- `compute_wrench()` — `(forces, torques)` in the body frame.
- `randomize_parameters(env_ids, thrust_coeff_scale, time_constant_scale)`.
- `reset(env_ids)`.
- Properties: `state`, `body_forces`, `body_torques`.

### `OceanCurrent`

An injectable current source shared by one or more hydrodynamics models.

```python
OceanCurrent(num_envs, device, cfg)
```

- `set(env_ids, velocity=None, strength=None)` — set explicitly, or sample within
  `max_velocity` when `velocity` is omitted.
- `add_drift(delta)` — additive increment to the current velocity.
- `reset(env_ids=None)`.
- Properties: `velocity_w` (6-D world-frame current), `max_velocity`.

### `HydroParams`

A dataclass bundle of the per-environment hydrodynamic parameters returned by
`get_parameters` / `base_parameters`.

---

## Researcher workflow

Build a current, inject it into one or more hydrodynamics models, step the
physics, and randomize through the public API:

```python
import torch
from marinelab.core import HydrodynamicsModel, OceanCurrent
from marinelab.assets import BlueROVHydrodynamicsCfg, OceanCurrentCfg

num_envs, device = 4096, "cuda"

# One shared current source, injected into the model.
current = OceanCurrent(num_envs, device, OceanCurrentCfg())
hydro = HydrodynamicsModel(num_envs, device, BlueROVHydrodynamicsCfg(),
                           dt=0.005, current=current)

# Per-step: compute body-frame hydrodynamic wrench.
forces, torques = hydro.compute_forces(root_lin_vel_w, root_ang_vel_w, root_quat_w)

# Per-reset domain randomization — all through the public API.
env_ids = torch.arange(num_envs, device=device)
hydro.scale_parameters(env_ids, added_mass=(0.8, 1.2), linear_damping=(0.8, 1.2))
cob = hydro.base_parameters.center_of_buoyancy[env_ids].clone()
cob[:, 2] += torch.rand(num_envs, device=device) * 0.04 - 0.02
hydro.set_parameters(env_ids, center_of_buoyancy=cob)
```

`marinelab.tasks.bluerov.bluerov_env.BlueROVEnv` is the minimal end-to-end
example of this pattern inside a `DirectRLEnv`; the private `constrained-albc`
overlay is the full research example.

---

## Domain randomization

Two entry points, both writing per-environment parameter buffers:

- **`set_parameters(env_ids, **fields)`** — absolute writes. You compute the value
  (for example, base CoB plus a sampled offset) and assign it. Use this when the
  randomization is not a simple multiple of the base value.
- **`scale_parameters(env_ids, **ranges)`** — convenience for the common case:
  each field becomes `base * uniform(lo, hi)`.

Both read from an immutable snapshot of the base parameters captured at
construction (`base_parameters`), so repeated randomization always scales from the
original configuration rather than compounding. Changing volume or water density
through `set_parameters`/`scale_parameters` recomputes the buoyancy force
automatically.

---

## Migration note

`marinelab.physics` has been removed. Import physics models from `marinelab.core`:

```python
from marinelab.core import HydrodynamicsModel, ThrusterModel, OceanCurrent
```

---

## Registered environments

| Task ID | Description |
|:--------|:------------|
| `Isaac-BlueROV-Hover-Direct-v0` | Hover control — demo / random agent |
| `Isaac-BlueROV-Hover-Train-Direct-v0` | Hover control — training variant |
| `Isaac-BlueROV-Hover-Eval-Direct-v0` | Hover control — evaluation variant |
| `Isaac-BlueROV-Attitude-Direct-v0` | Attitude control — demo / random agent |
| `Isaac-BlueROV-Attitude-Train-Direct-v0` | Attitude control — training variant |

---

## Docker

See [docs/installation.md](docs/installation.md) for the Docker-based deploy path.

---

## Documentation

- [Installation guide](docs/installation.md) — local and Docker paths
- [Architecture overview](docs/architecture.md) — layer diagram, subpackage roles
- [Contributing](CONTRIBUTING.md) — how to add new UUV tasks
- [Changelog](CHANGELOG.md)

---

## License

BSD-3-Clause — see [LICENSE](LICENSE).
