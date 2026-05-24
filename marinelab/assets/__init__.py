# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""UUV robot configurations (BlueROV, Hero Agent) and base hydro/thruster cfgs."""

from .uuv_cfg import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
from .bluerov import BLUEROV_CFG, BlueROVHydrodynamicsCfg, BlueROVThrusterCfg
from .hero_agent import (
    HERO_AGENT_ALBC_HEIGHT_OFFSET,
    HERO_AGENT_ALBC_JOINT_NAMES,
    HERO_AGENT_ALBC_LINK1_LENGTH,
    HERO_AGENT_ALBC_LINK2_LENGTH,
    HERO_AGENT_CFG,
    HERO_AGENT_USD_PATH,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
)
