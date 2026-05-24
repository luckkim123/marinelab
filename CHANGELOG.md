# Changelog

All notable changes to marinelab are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] - 2026-05-25

### Added

- Initial extraction from the isaaclab monorepo.
- BlueROV2 environments (Hover, Attitude) with Direct task interface — 5 registered Gym IDs.
- UUV assets: BlueROV2 and Hero Agent USD meshes, tracked with Git LFS.
- Shared marine physics: Fossen hydrodynamics model (`physics/hydrodynamics.py`) and
  thruster dynamics (`physics/thruster.py`), migrated from `isaaclab_tasks/models/`.
- Volume utility: buoyancy volume calculator (`utils/volume.py`), migrated from
  `isaaclab/utils/volume.py`.
- `pyproject.toml` with `isaaclab.extensions` entry-point for self-registration.
- Docker deploy files: `docker/Dockerfile` and `docker/docker-compose.yaml`.
- Documentation: `README.md`, `docs/installation.md`, `docs/architecture.md`,
  `CONTRIBUTING.md`.
- BSD-3-Clause `LICENSE` (inherited from upstream Isaac Lab).
