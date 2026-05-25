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
