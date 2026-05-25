# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .parameters import HydroParams, default_rigid_inertia

__all__ = ["HydroParams", "default_rigid_inertia"]
