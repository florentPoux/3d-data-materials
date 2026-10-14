# -*- coding: utf-8 -*-
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/3d-scan-to-bim-101-hands-on-automation-course-92f9e45acd7d
# Dataset:  https://learngeodata.eu/materials/wall-thickness-to-ifc/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
#
# Synthetic walls with known answers. Run from code/:  python -m pytest tests -q
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import wall_thickness_to_ifc as W  # noqa: E402
from synth import H, two_rooms  # noqa: E402

quiet = lambda msg: None


def partition(res):
    """The one measured wall whose faces run along x (normal along y)."""
    ws = [w for w in res["walls"] if abs(w["y_axis"][1]) > 0.9]
    assert len(ws) == 1, [w["name"] for w in res["walls"]]
    return ws[0]


# ------------------------------------------------------------------ the building blocks
def test_runs_along_bridges_short_holes_and_cuts_long_ones():
    s = np.r_[np.arange(0, 1.0, 0.02), np.arange(1.5, 2.5, 0.02), np.arange(3.2, 3.4, 0.02)]
    runs = W.runs_along(s)
    # the 0.5 m hole is bridged, the 0.7 m hole cuts, the 0.2 m tail is too short to be a face
    assert len(runs) == 1
    assert runs[0][0] == pytest.approx(0.0) and runs[0][1] == pytest.approx(2.48)


def test_valley_is_the_centre_of_the_empty_stretch():
    off = np.r_[np.full(500, 0.005), np.full(500, 0.155)]
    cen, _, sm = W.profile(off)
    low, mid = W.valley_of(cen, sm, 0.005, 0.155)
    assert low == 0.0
    assert mid == pytest.approx(0.08, abs=0.011)     # halfway, not the first empty bin


def test_profile_does_not_depend_on_point_order():
    rng = np.random.default_rng(1)
    off = rng.normal(0.3, 0.01, 4000)
    a = W.profile(off)
    b = W.profile(off[rng.permutation(len(off))])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_directions_recover_a_turned_scan():
    xyz, cls, names, ids = two_rooms(turn_deg=23.0)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    ang = np.degrees(res["phi"]) % 90.0
    assert min(abs(ang - 23.0), abs(ang - 23.0 + 90)) < 0.2


# ------------------------------------------------------------------ known thicknesses
@pytest.mark.parametrize("t", [0.07, 0.12, 0.20])
def test_plumb_partition_thickness(t):
    xyz, cls, names, ids = two_rooms(t=t)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    w = partition(res)
    assert w["thickness"] == pytest.approx(t, abs=0.001)
    assert w["ci"][0] - 0.001 <= t <= w["ci"][1] + 0.001
    assert abs(w["lean_a_deg"]) < 0.1 and abs(w["lean_b_deg"]) < 0.1
    assert w["rooms_both"]


def test_leaning_face_gives_mid_height_thickness_and_its_lean():
    t, lean = 0.15, 2.0
    xyz, cls, names, ids = two_rooms(t=t, lean_a_deg=lean)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    w = partition(res)
    storey = res["level"]["storey_h"]
    expect = t - np.tan(np.radians(lean)) * storey / 2      # the volume-equivalent thickness
    assert w["thickness"] == pytest.approx(expect, abs=0.002)
    assert w["lean_a_deg"] == pytest.approx(lean, abs=0.15)
    assert abs(w["lean_b_deg"]) < 0.15
    assert w["t_floor"] == pytest.approx(t, abs=0.003)


def test_tilted_scan_is_levelled_and_measures_the_same():
    xyz, cls, names, ids = two_rooms(t=0.12, tilt_deg=1.5)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    assert res["level"]["tilt_deg"] == pytest.approx(1.5, abs=0.05)
    assert partition(res)["thickness"] == pytest.approx(0.12, abs=0.001)


def test_levelling_with_the_floor_only():
    xyz, cls, names, ids = two_rooms(t=0.12, tilt_deg=1.0, no_ceiling=True)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    assert res["level"]["tilt_deg"] == pytest.approx(1.0, abs=0.05)
    assert partition(res)["thickness"] == pytest.approx(0.12, abs=0.001)


# ------------------------------------------------------------------ the pairing tests
def test_lining_is_not_a_wall():
    xyz, cls, names, ids = two_rooms(t=0.12, lining=True)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    cands = [c for F in res["families"] for c in F["cands"]]
    lining = [c for c in cands if abs(c["gap"] - 0.08) < 0.015]
    assert lining, "the lining pair should be a candidate"
    assert all(c["verdict"].startswith("not seen from both sides") for c in lining)
    assert len(res["walls"]) == 1      # only the real partition


def test_partition_at_the_edge_of_a_cropped_scan_is_still_a_wall():
    xyz, cls, names, ids = two_rooms(t=0.12, crop_room_b=True)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    w = partition(res)
    assert w["thickness"] == pytest.approx(0.12, abs=0.001)
    assert w["room_b"] < 1.0 and not w["rooms_both"]     # no room shows: no data, not no room
    assert w["reach_b"] < 0.02


# ------------------------------------------------------------------ determinism
def test_point_order_does_not_change_the_numbers():
    xyz, cls, names, ids = two_rooms(t=0.12, lean_a_deg=1.0)
    a = W.measure(xyz, cls, names, ids, log=quiet)
    p = np.random.default_rng(5).permutation(len(xyz))
    b = W.measure(xyz[p], cls[p], names, ids, log=quiet)
    wa, wb = partition(a), partition(b)
    assert wa["thickness"] == pytest.approx(wb["thickness"], abs=1e-9)
    assert wa["ci"] == pytest.approx(wb["ci"], abs=1e-9)
    assert wa["lean_a_deg"] == pytest.approx(wb["lean_a_deg"], abs=1e-9)


# ------------------------------------------------------------------ degenerate inputs
def test_no_wall_points_measures_nothing_and_says_why():
    xyz, cls, names, ids = two_rooms()
    keep = cls != ids["wall"]
    res = W.measure(xyz[keep], cls[keep], names, ids, log=quiet)
    assert res["walls"] == [] and "wall points" in res["reason"]


def test_a_single_room_has_no_partition():
    xyz, cls, names, ids = two_rooms(t=0.12)
    keep = xyz[:, 1] < -0.01        # room A only: face B of the partition is gone
    res = W.measure(xyz[keep], cls[keep], names, ids, log=quiet)
    assert res["walls"] == []
    assert res["reason"] == "no pair of faces passed the four tests"


def test_no_floor_is_refused():
    xyz, cls, names, ids = two_rooms()
    keep = cls != ids["floor"]
    with pytest.raises(ValueError):
        W.measure(xyz[keep], cls[keep], names, ids, log=quiet)


# ------------------------------------------------------------------ the IFC round trip
def test_ifc_round_trip_keeps_the_thickness(tmp_path):
    pytest.importorskip("ifcopenshell")
    xyz, cls, names, ids = two_rooms(t=0.12)
    res = W.measure(xyz, cls, names, ids, log=quiet)
    path = str(tmp_path / "walls.ifc")
    W.write_ifc(res["walls"], path, scan_key="synthetic")
    wall = res["lev"][res["wall_mask"]]
    pts = [wall[np.r_[w["idx_a"], w["idx_b"]]] for w in res["walls"]]
    report, _ = W.read_back(path, res["walls"], pts)
    assert report["errors"] == 0
    for r, w in zip(report["walls"], res["walls"]):
        assert r["delta_mm"] < 1e-3
        assert r["qto_width"] == pytest.approx(w["thickness"], abs=1e-9)
        # a plumb synthetic wall fits its solid; the few misses are the corner walls meeting it
        assert r["fit_within_2cm"] > 0.97
    # the same scan writes the same GlobalIds (the byte order of a unit set may differ, the ids do not)
    import ifcopenshell
    path2 = str(tmp_path / "walls2.ifc")
    W.write_ifc(res["walls"], path2, scan_key="synthetic")
    ids = lambda p: sorted((e.is_a(), e.Name or "", e.GlobalId) for e in ifcopenshell.open(p).by_type("IfcRoot"))
    assert ids(path) == ids(path2)
