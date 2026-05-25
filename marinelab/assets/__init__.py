# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""UUV robot configurations (BlueROV, ALBC) and base hydro/thruster cfgs."""

from .uuv_cfg import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
from .bluerov import BLUEROV_CFG, BlueROVHydrodynamicsCfg, BlueROVThrusterCfg
from .albc import (
    ALBC_HEIGHT_OFFSET,
    ALBC_JOINT_NAMES,
    ALBC_LINK1_LENGTH,
    ALBC_LINK2_LENGTH,
    ALBC_CFG,
    ALBC_USD_PATH,
    ALBCBuoyHydrodynamicsCfg,
    ALBCHydrodynamicsCfg,
)
