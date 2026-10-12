# Copyright (c) 2026 Seungmin Kim. SPDX-License-Identifier: MIT
"""marinelab: underwater-vehicle environments and shared marine physics for Isaac Lab."""

# Importing the tasks subpackage triggers gym.register() calls (self-registration).
from . import tasks  # noqa: F401
