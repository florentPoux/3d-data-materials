# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-build-a-2d-map-and-3d-evacuation-simulation-from-3d-scan-data-python-a4d7e71c0f0c
# Dataset:  https://learngeodata.eu/materials/evacuation-distance-map/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""On the real floor: the same numbers whatever the point order, and within a stated tolerance
when every coordinate is nudged by up to 0.5 mm (well below a scanner's own noise)."""
import os

import numpy as np
import pytest

from conftest import ev

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                    "s3dis_area1_open_5cm.ply")
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="the S3DIS floor is not in code/data")


@pytest.fixture(scope="module")
def floor():
    xyz, rgb, role, fz, _ = ev.load_floor()
    m, st = ev.solve(xyz, rgb, role, fz, sweep=False, write=False)
    return xyz, rgb, role, fz, m, st


def key(m):
    return (m["walk_m2"], m["field"]["farthest_m"], m["field"]["mean_m"], m["rooms"]["n"],
            m["dead_ends"]["longest_common_path_m"], [c["length_m"] for c in m["dead_ends"]["corridor"]])


def test_shuffled_points_give_identical_results(floor):
    xyz, rgb, role, fz, m, st = floor
    p = np.random.default_rng(7).permutation(len(xyz))
    m2, st2 = ev.solve(xyz[p], rgb[p], role[p], fz, sweep=False, write=False)
    assert key(m2) == key(m)
    assert np.array_equal(st2["D"], st["D"])


def test_two_runs_give_identical_results(floor):
    xyz, rgb, role, fz, m, st = floor
    m2, st2 = ev.solve(xyz, rgb, role, fz, sweep=False, write=False)
    assert key(m2) == key(m)


def test_half_millimetre_nudge_stays_within_tolerance(floor):
    xyz, rgb, role, fz, m, st = floor
    j = xyz + np.random.default_rng(8).uniform(-5e-4, 5e-4, xyz.shape)
    m2, _ = ev.solve(j, rgb, role, fz, sweep=False, write=False)
    print("nudge:", key(m), "->", key(m2))
    assert abs(m2["walk_m2"] / m["walk_m2"] - 1) < 0.01
    assert abs(m2["field"]["farthest_m"] - m["field"]["farthest_m"]) < 0.5
    assert abs(m2["field"]["mean_m"] - m["field"]["mean_m"]) < 0.1
