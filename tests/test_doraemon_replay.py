# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for DORAEMON curriculum record & replay (marinelab.algorithms)."""

import dataclasses
import sys

import pytest
import torch


def _configclass_with_factory(cls):
    for name, value in list(vars(cls).items()):
        if name in getattr(cls, "__annotations__", {}) and isinstance(value, (dict, list, set)):
            default = value
            setattr(cls, name, dataclasses.field(default_factory=lambda d=default: type(d)(d)))
    return dataclasses.dataclass(cls)


sys.modules["isaaclab.utils"].configclass = _configclass_with_factory

from marinelab.algorithms.doraemon import (  # noqa: E402
    DoraemonCfg,
    DoraemonScheduler,
)

# Two simple DR params for tests: (name, field_name, lo, hi).
_PARAM_DEFS = [
    ("p0", "p0_range", 0.0, 1.0),
    ("p1", "p1_range", 0.0, 2.0),
]


class _FakeDRCfg:
    p0_range = (0.0, 1.0)
    p1_range = (0.0, 2.0)


def _make_scheduler(step_interval=2, min_episodes=4, **cfg_kw):
    cfg = DoraemonCfg(
        step_interval=step_interval,
        min_episodes=min_episodes,
        buffer_size=64,
        init_concentration=30.0,
        **cfg_kw,
    )
    return DoraemonScheduler(
        cfg, torch.device("cpu"), dr_cfg=_FakeDRCfg(), param_defs=_PARAM_DEFS
    )


def _fill_buffer(sch, n=8):
    xi, log_probs = sch.sample(n)
    returns = torch.full((n,), 100.0)
    success = torch.ones(n)
    sch.record_episodes(xi, returns, success, log_probs)


def test_record_appends_on_update():
    sch = _make_scheduler(step_interval=2, min_episodes=4)
    _fill_buffer(sch, 8)
    # _step_count starts at 0; step(iteration) called each RL iter.
    # Update fires when _step_count % step_interval == 0 AND buffer >= min_episodes.
    sch.step(iteration=0)  # _step_count 0 -> update fires, records
    n_after_0 = len(sch.export_trajectory())
    sch.step(iteration=1)  # _step_count 1 -> no update, no record
    n_after_1 = len(sch.export_trajectory())
    sch.step(iteration=2)  # _step_count 2 -> update fires, records
    n_after_2 = len(sch.export_trajectory())

    assert n_after_0 == 1
    assert n_after_1 == 1  # unchanged between updates
    assert n_after_2 == 2
    # Recorded iteration is the RL iteration passed in, not _step_count.
    assert sch.export_trajectory()[0]["iter"] == 0
    assert sch.export_trajectory()[1]["iter"] == 2
    # Each entry carries the full (a, b) vectors.
    assert len(sch.export_trajectory()[0]["a"]) == 2
    assert len(sch.export_trajectory()[0]["b"]) == 2


def test_export_recording_has_metadata():
    sch = _make_scheduler(step_interval=2, min_episodes=4)
    _fill_buffer(sch, 8)
    sch.step(iteration=0)
    rec = sch.export_recording()
    assert rec["param_names"] == ["p0", "p1"]
    assert rec["param_bounds"] == [[0.0, 1.0], [0.0, 2.0]]
    assert rec["trajectory"] == sch.export_trajectory()
