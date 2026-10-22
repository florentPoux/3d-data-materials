# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-automate-3d-deliverables-accessibility-map-from-3d-scan-3ecef96a5a6f
# Dataset:  https://learngeodata.eu/materials/wheelchair-accessibility-audit/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""Known-answer tests for the accessibility audit, on synthetic scans built point by point.

Every test builds a small scene whose true answer is known by construction (a corridor of
a given width, a doorway of a given width, a room of a given size, two rooms and a leak),
runs the audit's own functions on its points, and checks the answer.

    cd code
    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accessibility_audit as aa  # noqa: E402

CELL = aa.CELL
PROF = aa.PROFILES["wheelchair"]


# ------------------------------------------------------------------ scene builder
def box_faces(x0, y0, x1, y1, h=2.0, step=0.03):
    """Points on the four vertical faces of an axis-aligned wall box, like a scan sees them."""
    zs = np.arange(0.0, h, step)
    pts = []
    for xa, ya, xb, yb in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        n = max(int(np.hypot(xb - xa, yb - ya) / step), 1) + 1
        t = np.linspace(0, 1, n)
        xy = np.column_stack([xa + t * (xb - xa), ya + t * (yb - ya)])
        pts.append(np.column_stack([np.repeat(xy, len(zs), 0), np.tile(zs, len(xy))]))
    return np.vstack(pts)


def scene(extent, walls, doors=(), seed=0, jitter=0.002):
    """Floor points on a 2 cm lattice over extent (x0, y0, x1, y1), none under the walls;
    walls as boxes (class wall); doors as boxes of door-class points."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = extent
    gx, gy = np.meshgrid(np.arange(x0, x1, 0.02), np.arange(y0, y1, 0.02), indexing="ij")
    fl = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    for bx0, by0, bx1, by1 in walls:
        fl = fl[~((fl[:, 0] > bx0) & (fl[:, 0] < bx1) & (fl[:, 1] > by0) & (fl[:, 1] < by1))]
    parts, cls = [fl], [np.full(len(fl), aa.FLOOR)]
    for b in walls:
        p = box_faces(*b)
        parts.append(p)
        cls.append(np.full(len(p), 2))
    for b in doors:
        p = box_faces(*b)
        parts.append(p)
        cls.append(np.full(len(p), aa.DOOR))
    xyz = np.vstack(parts)
    xyz = xyz + rng.normal(0, jitter, xyz.shape)
    return xyz, np.concatenate(cls).astype(np.int16)


def grid(xyz, cls):
    aa.S.clear()
    aa.T.clear()
    return aa.step2_grid(xyz, cls)


def corridor(width, offset, length=4.0, t=0.15):
    """A corridor along x, clear `width` between two wall boxes, starting at y = offset."""
    a, b = offset, offset + width
    walls = [(0.0, a - t, length, a), (0.0, b, length, b + t)]
    return scene((-0.3, a - t - 0.3, length + 0.3, b + t + 0.3), walls)


def mid_width(g, W, x0=1.0, x1=3.0):
    i0, i1 = int((x0 - g["lo"][0]) / CELL), int((x1 - g["lo"][0]) / CELL)
    return float(W[i0:i1].max())


# ------------------------------------------------------------------ step 3: clear width
@pytest.mark.parametrize("k", range(1, 25))
def test_straight_run_of_k_free_cells_reads_k_plus_one_cells(k):
    """Parity: an even run has no centre cell; the half-cell grid still finds its centre line."""
    free = np.zeros((40, 40), bool)
    free[5:35, 5:5 + k] = True
    W = aa.clear_width(free)
    assert W[20].max() == pytest.approx((k + 1) * CELL)


def test_textbook_formula_loses_a_cell_on_even_runs():
    """The v1 formula 2 d - cell, kept here to show the bias it had."""
    for k in (16, 17, 18):
        free = np.zeros((40, 40), bool)
        free[5:35, 5:5 + k] = True
        old = (2 * ndi.distance_transform_edt(free) * CELL - CELL)[20].max()
        assert old == pytest.approx((k - (k % 2 == 0)) * CELL)


def test_corridor_width_from_points_is_unbiased():
    """A 0.90 m corridor at 20 positions against the lattice: every reading within one cell,
    and on average within 1 cm. The v1 estimate read the same corridors about 7 cm narrow."""
    errs, old_errs = [], []
    for s in range(20):
        off = 0.37 + s * 0.0123
        xyz, cls = corridor(0.90, off)
        g = grid(xyz, cls)
        W = aa.clear_width(g["free_chair"])
        errs.append(mid_width(g, W) - 0.90)
        old = np.where(g["free_chair"], 2 * ndi.distance_transform_edt(g["free_chair"]) * CELL - CELL, 0)
        old_errs.append(mid_width(g, old) - 0.90)
    errs = np.array(errs)
    assert np.abs(errs).max() <= CELL + 1e-9
    assert abs(errs.mean()) < 0.01
    assert np.mean(old_errs) < -0.05


@pytest.mark.parametrize("width,passes", [(0.70, False), (0.80, False), (1.00, True), (1.10, True)])
def test_corridor_verdict_against_0_90(width, passes):
    xyz, cls = corridor(width, 0.411)
    g = grid(xyz, cls)
    W = aa.clear_width(g["free_chair"])
    assert (mid_width(g, W) >= PROF["passage"]) == passes


# ------------------------------------------------------------------ step 5: doors
def two_rooms(door_w, y_open=1.137, gap_door=True):
    """Two 3 x 3 m rooms side by side, one wall at x 3.00-3.15 with an opening door_w wide."""
    walls = [(-0.15, -0.15, 6.3, 0.0), (-0.15, 3.0, 6.3, 3.15),          # south, north
             (-0.15, 0.0, 0.0, 3.0), (6.15, 0.0, 6.3, 3.0),              # west, east
             (3.0, 0.0, 3.15, y_open), (3.0, y_open + door_w, 3.15, 3.0)]  # the dividing wall
    doors = [(3.06, y_open + 0.02, 3.09, y_open + door_w - 0.02)] if gap_door else []
    return scene((-0.4, -0.4, 6.6, 3.4), walls, doors)


@pytest.mark.parametrize("door_w", [0.62, 0.78, 0.83, 0.95])
def test_door_width_within_a_quarter_cell(door_w):
    for y_open in (1.113, 1.137, 1.162):
        xyz, cls = two_rooms(door_w, y_open)
        g = grid(xyz, cls)
        doors = aa.step5_doors(g, PROF, xyz, cls)
        assert len(doors) == 1
        assert abs(doors[0]["width_m"] - door_w) <= CELL / 2 + 1e-9


def reach_rooms(door_w):
    xyz, cls = two_rooms(door_w)
    g = grid(xyz, cls)
    aa.ENTRANCES = [(1.5, 1.5)]
    W = aa.step3_width(g, PROF)
    C, doorzone = aa.step4_centre(g, W, PROF)
    R = aa.step7_reach(g, C, W, PROF)
    return R


def test_wide_door_lets_the_wheelchair_through():
    R = reach_rooms(0.95)
    assert R["nrooms"] == 2 and list(R["status"]) == [2, 2]


def test_narrow_door_stops_the_wheelchair_not_the_walker():
    R = reach_rooms(0.62)
    assert R["nrooms"] == 2 and sorted(R["status"]) == [1, 2]


# ------------------------------------------------------------------ step 6: turning
def square_room(s, off=0.213):
    t = 0.15
    walls = [(off - t, off - t, off + s + t, off), (off - t, off + s, off + s + t, off + s + t),
             (off - t, off, off, off + s), (off + s, off, off + s + t, off + s)]
    return scene((off - 0.5, off - 0.5, off + s + 0.5, off + s + 0.5), walls)


def test_element_is_the_full_diameter():
    assert aa.se_cells(1.50) == 29          # 1.50 m face to face = 29 free cells + the one the walls claim
    assert aa.disk(29).shape == (29, 29)


@pytest.mark.parametrize("s,fits", [(1.40, False), (1.60, True), (2.00, True)])
def test_turning_circle_in_a_square_room(s, fits):
    g = grid(*square_room(s))
    centres, turn, wrong = aa.step6_turning(g, PROF)
    assert centres.any() == fits
    assert aa.S["turning"]["edt_agreement"] == 1.0


def test_half_diameter_mistake_accepts_a_1_m_room():
    g = grid(*square_room(1.00))
    centres, turn, wrong = aa.step6_turning(g, PROF)
    assert not centres.any() and wrong.any()


# ------------------------------------------------------------------ step 7: rooms
def test_a_leak_in_a_wall_does_not_merge_two_rooms():
    """Two rooms, no door, a 12 cm hole in the dividing wall (missing points): still two rooms."""
    xyz, cls = two_rooms(0.12, gap_door=False)
    g = grid(xyz, cls)
    rooms, n = aa.rooms_of(g)
    assert n == 2


def test_a_wide_opening_makes_one_room():
    xyz, cls = two_rooms(1.40, gap_door=False)
    g = grid(xyz, cls)
    rooms, n = aa.rooms_of(g)
    assert n == 1


# ------------------------------------------------------------------ determinism
def test_point_order_does_not_matter():
    xyz, cls = two_rooms(0.83)
    g1 = grid(xyz, cls)
    d1 = aa.step5_doors(g1, PROF, xyz, cls)
    perm = np.random.default_rng(5).permutation(len(xyz))
    g2 = grid(xyz[perm], cls[perm])
    d2 = aa.step5_doors(g2, PROF, xyz[perm], cls[perm])
    for k in ("H", "struct", "door", "free_chair", "free_foot"):
        assert np.array_equal(g1[k], g2[k], equal_nan=True), k
    assert [d["width_m"] for d in d1] == [d["width_m"] for d in d2]


def test_grid_sits_on_a_fixed_lattice():
    """Moving the scan's extreme point by 1 mm must not move every cell (v1 anchored the grid
    on the minimum coordinate, and it did)."""
    xyz, cls = two_rooms(0.83)
    g1 = grid(xyz, cls)
    xyz2 = xyz.copy()
    xyz2[np.argmin(xyz2[:, 0]), 0] -= 0.001
    g2 = grid(xyz2, cls)
    assert np.allclose(g1["lo"], g2["lo"])
    assert np.array_equal(g1["free_chair"], g2["free_chair"])


def test_voxel_thinning_is_order_free():
    import prepare_s3dis as ps
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 1, (20000, 3))
    cls = rng.integers(0, 14, 20000).astype(np.uint8)
    k1, l1 = ps.thin_access(xyz, cls)
    perm = rng.permutation(len(xyz))
    k2, l2 = ps.thin_access(xyz[perm], cls[perm])
    assert np.array_equal(xyz[k1], xyz[perm][k2]) and np.array_equal(l1, l2)
    assert np.array_equal(cls[k1], l1)          # every kept point carries its own class
