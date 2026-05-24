# marinelab

Underwater-vehicle environments and shared marine physics for Isaac Lab.

marinelab is a public overlay extension that sits on top of a clean
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) fork and provides:

- **BlueROV2 environments** registered directly into the Isaac Lab Gym registry.
- **Shared marine physics** — Fossen hydrodynamics model and thruster dynamics.
- **UUV assets** — BlueROV2 and Hero Agent USD meshes (tracked with Git LFS).
- **Volume utility** — buoyancy volume calculations for arbitrary UUV geometries.

---

## Three-layer overlay architecture

```
isaaclab          (clean upstream fork; Docker base image)
    ^  pip install / import isaaclab
marinelab         (this repo; public junior deploy layer)
    ^  pip install / import marinelab
constrained-albc  (private research overlay; optional)
```

`marinelab` depends only on `isaaclab`. It does not depend on `constrained-albc`.
There are no upward dependencies.

---

## Quickstart

### Prerequisites

- Docker with NVIDIA runtime, or an existing editable isaaclab install.
- Git LFS (`apt-get install git-lfs` if absent).

### Step 1 — clone and pull meshes

```bash
git clone <marinelab-url> /workspace/marinelab
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
