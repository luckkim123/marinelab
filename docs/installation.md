# Installation

marinelab can be deployed two ways: locally on top of an existing isaaclab
editable install, or via Docker with isaaclab as the base image.

---

## Prerequisites

| Requirement | Why | Provided by |
|:---|:---|:---|
| Git LFS | marinelab stores UUV meshes (`.usd`, `.dae`, `.obj`) in Git LFS | `apt-get install git-lfs` before cloning |
| `isaaclab` + `isaaclab_assets` | **Not** pip dependencies of marinelab — do not list in `pyproject.toml` or install separately | Local: the isaaclab editable install already on the host. Docker: baked into the base image |

---

## Path A — Local install

Use this path when you already have a working isaaclab editable install on the
host machine.

### 1. Clone the repo and pull meshes

```bash
git clone <marinelab-url> /workspace/marinelab
cd /workspace/marinelab
git lfs install
git lfs pull
```

### 2. Install marinelab into the isaaclab environment

The isaaclab environment exposes its own Python interpreter via `isaaclab.sh -p`.
Use that launcher so the install lands in the same Python that Isaac Sim uses:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install -e /workspace/marinelab
```

### 3. Verify

Verification must happen inside a launched Isaac Sim app because marinelab
transitively imports `pxr` (the USD runtime), which is only available once the
app is running. A bare `python -c "import marinelab"` will fail with
`No module named pxr` — that is expected outside the app.

Run the random agent instead:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

A clean exit (no import errors, no shape mismatches) confirms the install is
correct.

---

## Path B — Docker

Use this path for reproducible junior deployments or CI.

### Overview

```
isaaclab Docker base image   (built from the isaaclab clean fork)
        |  FROM
        v
marinelab Docker image       (marinelab/docker/Dockerfile)
        pip install -e marinelab
        git lfs pull
```

### 1. Build the isaaclab base image

Follow the build instructions in the isaaclab repository's `docker/` directory.
This produces a local image tagged `isaaclab:latest` (or a custom tag you
choose).

### 2. Build the marinelab image

```bash
cd /workspace/marinelab
docker compose -f docker/docker-compose.yaml build
```

To use a custom isaaclab image tag:

```bash
ISAACLAB_IMAGE=my-isaaclab:5.1.0 \
    docker compose -f docker/docker-compose.yaml build
```

### 3. Start the container

```bash
# Allow the container to open X11 windows on the host (if using GUI)
xhost +local:docker

docker compose -f docker/docker-compose.yaml run --rm marinelab bash
```

### 4. Verify inside the container

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

### Notes

- The `docker-compose.yaml` bind-mounts the marinelab source directory into the
  container so local edits are reflected immediately without a rebuild.
- GPU access uses the NVIDIA container runtime (`deploy.resources.reservations`).
  Ensure `nvidia-container-toolkit` is installed on the host.
- The exact Python launcher inside the isaaclab image is
  `${ISAACLAB_PATH}/isaaclab.sh -p -m pip` (or `/isaac-sim/python.sh -m pip`
  directly). The Dockerfile uses the direct path; if your isaaclab image uses a
  different layout, adjust accordingly.
