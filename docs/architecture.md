# Architecture

## Three-layer overlay

marinelab is the middle layer in a three-repo stack. Each layer is an
independently installable Python package. Upper layers depend on lower layers;
there are no upward dependencies.

```
+---------------------------------------------------------------------+
|  constrained-albc  (private research overlay)                       |
|  ConstraintTRPO ALBC (attitude + full-DOF) + TDC + student          |
|  distillation + analysis                                            |
|  pip install -e constrained-albc   (on top of marinelab)            |
+----------------------------------+----------------------------------+
                                   |  import marinelab
+----------------------------------v----------------------------------+
|  marinelab  (this repo; public junior deploy layer)                 |
|  BlueROV environments + UUV assets (Git LFS) + marine physics       |
|  pip install -e marinelab         (on top of isaaclab)              |
+----------------------------------+----------------------------------+
                                   |  import isaaclab
+----------------------------------v----------------------------------+
|  isaaclab  (clean upstream fork; Docker base)                       |
|  Isaac Sim 5.1.0 + Isaac Lab v2.3.0 framework                       |
+---------------------------------------------------------------------+
```

**Dependency direction:** downward only.
- marinelab imports isaaclab; it never imports constrained-albc.
- constrained-albc imports marinelab (for shared physics and assets).
- isaaclab has no knowledge of either overlay.

---

## Overlay mechanism

marinelab registers its Gym environments via the `isaaclab.extensions`
entry-point declared in `pyproject.toml`:

```toml
[project.entry-points."isaaclab.extensions"]
marinelab = "marinelab"
```

Isaac Lab discovers and loads all installed extensions with this entry-point
at startup. marinelab therefore does **not** modify any file inside the isaaclab
fork — the isaaclab fork remains a zero-touch clean upstream fork.

---

## Subpackage roles

```
marinelab/
    __init__.py          Extension registration entry point.
                         Imports tasks/ so Gym IDs register on load.

    assets/              UUV robot configurations and 3D meshes.
        uuv_cfg.py       Shared base config for UUV articulations.
        bluerov/         BlueROV2 asset config + USD meshes (Git LFS).
        albc/            ALBC asset config + USD/DAE meshes (Git LFS).

    core/                Stable framework API. Import from here.
        hydrodynamics.py Fossen 6-DOF model: added mass, damping, restoring
                         forces; setter-based DR API. HydrodynamicsModel.
        thruster.py      First-order thruster dynamics + allocation matrix.
                         ThrusterModel, ThrusterCfg.
        ocean_current.py Injectable ocean-current component. OceanCurrent.
        parameters.py    HydroParams dataclass + default_rigid_inertia helper.
        volume.py        Buoyancy volume calculation for UUV geometries.

    algorithms/          Promoted DR engine, shared across overlays.
        doraemon.py      DORAEMON DR curriculum (Tiboni et al., ICLR 2024):
                         adaptive Beta-distribution scheduling. DoraemonCfg.

    tasks/               Gym-registered Isaac Lab environments.
        bluerov/         Five BlueROV2 Direct-task environments:
                           Hover (demo / train / eval),
                           Attitude (demo / train).
            bluerov_env.py      DirectRLEnv subclass.
            bluerov_env_cfg.py  Base @configclass config.
            hover_env_cfg.py    Hover task config.
            attitude_env_cfg.py Attitude task config.
            agents/             RL agent configs (rsl_rl, skrl, sb3, rl_games).
            mdp/                Domain randomization events.
            rewards/            Reward terms and reward manager.
            tasks/              Task wrappers (hover_task, attitude_task).

    utils/               Deprecated re-export shim. utils/volume re-exports
                         marinelab.core.volume for backward compatibility.
```

---

## Data flow at runtime

```
Isaac Sim app starts
    -> Isaac Lab loads isaaclab.extensions entry-points
    -> marinelab.__init__ is imported
    -> marinelab.tasks.__init__ imports tasks.bluerov.__init__
    -> gymnasium.register() called for all 5 BlueROV IDs
    -> random_agent.py (or training script) calls gym.make("Isaac-BlueROV-*")
    -> BluerovEnv.__init__ instantiates physics (hydrodynamics + thruster)
       and loads USD assets from marinelab/assets/bluerov/meshes/
```

---

## Asset storage (Git LFS)

Binary mesh files (`.usd`, `.dae`, `.obj`) are tracked with Git LFS to keep
repository clone size small. The actual mesh data is fetched with:

```bash
git lfs install
git lfs pull
```

This must be run before launching any environment — missing LFS objects cause
USD load errors at simulation startup.

---

## Constrained-albc relationship

`constrained-albc` is a private overlay that sits above marinelab. It imports
`marinelab.core` for the Fossen model and thruster dynamics, `marinelab.assets`
for UUV configurations, and `marinelab.algorithms.doraemon` for the DR
curriculum. There are zero cross-imports in the other direction: marinelab
has no references to constrained-albc code. This clean separation means
marinelab can be deployed to juniors independently, without exposing any private
research code.
