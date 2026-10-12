# Copyright (c) 2026 Seungmin Kim. SPDX-License-Identifier: MIT
"""UUV robot configurations (BlueROV, ALBC) and base hydro/thruster cfgs."""

from .albc import (
    ALBC_CFG,
    ALBC_HEIGHT_OFFSET,
    ALBC_JOINT_NAMES,
    ALBC_LINK1_LENGTH,
    ALBC_LINK2_LENGTH,
    ALBC_USD_PATH,
    ALBCBuoyHydrodynamicsCfg,
    ALBCHydrodynamicsCfg,
)
from .bluerov import BLUEROV_CFG, BlueROVHydrodynamicsCfg, BlueROVThrusterCfg
from .uuv_cfg import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
