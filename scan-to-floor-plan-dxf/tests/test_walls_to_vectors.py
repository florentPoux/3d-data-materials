# -*- coding: utf-8 -*-
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-turn-a-3d-scan-into-a-cad-floor-plan-f38562744833
# Dataset:  https://learngeodata.eu/materials/scan-to-floor-plan-dxf/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""Tests for walls_to_vectors.py on synthetic scans with known answers.

Every room here is generated: we know where each wall face is to the
millimetre, so the pipeline's lengths, areas, gaps and flags can be checked
against the truth, not against a previous run.

Run from the code/ folder:   python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest
import ezdxf
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import walls_to_vectors as W  # noqa: E402

H = 2.5          # room height of every synthetic scan (m)


def wall_face(a, b, rng, spacing=0.015, noise=0.002, z0=0.0, z1=H, keep=None):
    """Points on a vertical face from a to b (plan), with normal noise.
    keep(t, z) -> bool mask lets a test cut holes or keep only stubs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    n = int(L * (z1 - z0) / spacing ** 2)
    t = rng.uniform(0, L, n)
    z = rng.uniform(z0, z1, n)
    u = (b - a) / L
    nrm = np.array([-u[1], u[0]])
    xy = a + t[:, None] * u + rng.normal(0, noise, n)[:, None] * nrm
    m = np.ones(n, bool) if keep is None else keep(t, z)
    return np.c_[xy[m], z[m]]


def slab(corners, z, rng, n=20000):
    """Floor or ceiling points filling the polygon's bounding rectangle (convex room)."""
    c = np.asarray(corners)
    o, ex, ey = c[0], c[1] - c[0], c[3] - c[0]
    s = rng.uniform(0, 1, (n, 2))
    xy = o + s[:, :1] * ex + s[:, 1:] * ey
    return np.c_[xy, np.full(n, z) + rng.normal(0, 0.002, n)]


def box_room(w=6.0, d=4.0, angle=15.0, keeps=None, board=False, walls=(0, 1, 2, 3), seed=0):
    """A w x d room (inner faces), rotated by angle. Returns xyz, cls, corners."""
    rng = np.random.default_rng(seed)
    th = np.radians(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    local = np.array([[0, 0], [w, 0], [w, d], [0, d]], float)
    C = local @ R.T + np.array([3.0, -2.0])
    pts, cls = [], []
    keeps = keeps or {}
    for k in walls:
        P = wall_face(C[k], C[(k + 1) % 4], rng, keep=keeps.get(k))
        pts.append(P)
        cls.append(np.full(len(P), W.WALL_CLASS))
    if board:            # a 1.2 m board standing 5 cm proud of wall 0, labelled wall
        u = (C[1] - C[0]) / w
        nrm = np.array([-u[1], u[0]])                    # points into the room
        a = C[0] + 2.0 * u + 0.05 * nrm
        P = wall_face(a, a + 1.2 * u, rng, z0=0.9, z1=2.0)
        pts.append(P)
        cls.append(np.full(len(P), W.WALL_CLASS))
    for z, c in ((0.0, W.FLOOR_CLASS), (H, W.CEILING_CLASS)):
        P = slab(C, z, rng)
        pts.append(P)
        cls.append(np.full(len(P), c))
    return np.vstack(pts), np.concatenate(cls), C


def quiet_run(xyz, cls, out_dir=None):
    return W.run_pipeline(xyz, cls, out_dir)


# --- whole-pipeline answers ------------------------------------------------
def test_box_room_known_dimensions():
    """6.00 x 4.00 m at 15 degrees, a board fixed on one wall: four walls on the
    faces within 1 cm, the area within 0.5 %, every corner joined."""
    xyz, cls, _ = box_room(board=True)
    m, st = quiet_run(xyz, cls)
    L = sorted(w["length_m"] for w in m["measure"]["walls"])
    assert m["merge"]["walls"] == 4
    assert L[0] == pytest.approx(4.0, abs=0.01) and L[1] == pytest.approx(4.0, abs=0.01)
    assert L[2] == pytest.approx(6.0, abs=0.01) and L[3] == pytest.approx(6.0, abs=0.01)
    assert m["measure"]["room_area_m2"] == pytest.approx(24.0, rel=0.005)
    assert m["square"]["corners_joined"] == 4
    assert m["square"]["room_axis_deg"] == pytest.approx(15.0, abs=0.1)
    assert all(w["seen"] > 0.95 and not w["assumed"] for w in m["measure"]["walls"])


def test_snap_puts_the_line_on_the_face(monkeypatch):
    """Without the snap the board drags its wall towards the room; with it, the
    room is back to its true area."""
    xyz, cls, _ = box_room(board=True)
    m_on, _ = quiet_run(xyz, cls)
    monkeypatch.setattr(W, "SNAP", False)
    m_off, _ = quiet_run(xyz, cls)
    err_on = abs(m_on["measure"]["room_area_m2"] - 24.0)
    err_off = abs(m_off["measure"]["room_area_m2"] - 24.0)
    assert err_on < err_off
    assert err_on < 0.05


def test_corridor_width_and_open_ends(tmp_path):
    """Two parallel 8 m walls 1.20 m apart, no end walls: two walls, the right
    width, no room, and a DXF that still writes."""
    rng = np.random.default_rng(1)
    a = wall_face((0, 0), (8, 0), rng)
    b = wall_face((8, 1.2), (0, 1.2), rng)
    f = slab([(0, 0), (8, 0), (8, 1.2), (0, 1.2)], 0.0, rng)
    c = slab([(0, 0), (8, 0), (8, 1.2), (0, 1.2)], H, rng)
    xyz = np.vstack([a, b, f, c])
    cls = np.concatenate([np.full(len(a) + len(b), W.WALL_CLASS),
                          np.full(len(f), W.FLOOR_CLASS), np.full(len(c), W.CEILING_CLASS)])
    m, st = quiet_run(xyz, cls, str(tmp_path))
    walls = st["walls"]
    assert len(walls) == 2 and m["measure"]["rooms"] == 0
    u, n, _ = W.frame(walls[0])
    mid = (walls[1]["p0"] + walls[1]["p1"]) / 2
    assert abs(np.dot(mid - walls[0]["p0"], n)) == pytest.approx(1.20, abs=0.01)
    assert os.path.exists(tmp_path / "floor_plan.dxf")


def test_gap_is_measured_edge_to_edge():
    """A 1.00 m full-height hole in a wall is reported as 1.00 m (+- 2 stations)."""
    xyz, cls, _ = box_room(keeps={0: lambda t, z: (t < 2.5) | (t > 3.5)})
    m, _ = quiet_run(xyz, cls)
    gaps = [g for w in m["measure"]["walls"] for g in w["gaps_m"]]
    assert len(gaps) == 1
    assert gaps[0] == pytest.approx(1.00, abs=0.04)


def test_stub_wall_is_assumed_but_placed_by_its_base():
    """Wall 2 keeps only 0.5 m stubs in the band plus its skirting: it is drawn
    assumed, the room still closes, and the skirting confirms its position."""
    keep = {2: lambda t, z: (t < 0.5) | (t > 5.5) | (z < 0.2)}
    xyz, cls, _ = box_room(keeps=keep)
    m, _ = quiet_run(xyz, cls)
    ws = m["measure"]["walls"]
    assumed = [w for w in ws if w["assumed"]]
    assert len(assumed) == 1
    assert assumed[0]["seen"] < 0.25 and assumed[0]["base"] > 0.95
    assert abs(assumed[0]["base_offset_mm"]) < 5
    assert m["measure"]["room_area_m2"] == pytest.approx(24.0, rel=0.01)


def test_point_order_does_not_matter():
    """Shuffling the points changes nothing but the timings."""
    xyz, cls, _ = box_room(board=True)
    m1, _ = quiet_run(xyz, cls)
    p = np.random.default_rng(7).permutation(len(xyz))
    m2, _ = quiet_run(xyz[p], cls[p])
    for m in (m1, m2):
        m.pop("times_ms")
        m.pop("total_ms")
    assert m1 == m2


def test_dxf_units_layers_and_evidence(tmp_path):
    xyz, cls, _ = box_room()
    m, _ = quiet_run(xyz, cls, str(tmp_path))
    doc = ezdxf.readfile(tmp_path / "floor_plan.dxf")
    assert doc.units == ezdxf.units.M
    lines = [e for e in doc.modelspace() if e.dxftype() == "LINE" and e.dxf.layer == "WALL"]
    assert len(lines) == 4 and all(e.has_xdata("LGO_EVIDENCE") for e in lines)
    assert m["dxf"]["dxf_entities"]["DIMENSIONS"] == 4


# --- degenerate inputs -----------------------------------------------------
def test_no_wall_points_is_a_clear_error():
    xyz, cls, _ = box_room()
    cls = np.where(cls == W.WALL_CLASS, 5, cls)
    with pytest.raises(ValueError, match="no points of class"):
        quiet_run(xyz, cls)


def test_single_wall_no_room_no_crash(tmp_path):
    xyz, cls, _ = box_room(walls=(0,))
    m, _ = quiet_run(xyz, cls, str(tmp_path))
    assert m["merge"]["walls"] == 1 and m["measure"]["rooms"] == 0
    assert m["dxf"]["dxf_entities"]["WALL"] == 1


# --- single steps ----------------------------------------------------------
@pytest.mark.parametrize("thick,gap,bridged", [(3, 2, True), (3, 3, False),
                                               (5, 4, True), (5, 5, False),
                                               (2, 1, False)])
def test_closing_bridges_what_the_disk_can_hold(thick, gap, bridged):
    """CONTENT L74: know what the element really does. A radius-2 disk bridges a
    2-cell gap in a 3-cell band and a 4-cell gap in a 5-cell band, and nothing
    in a band 2 cells thick: the erosion only keeps a bridge the disk fits in."""
    g = np.zeros((60, 140), bool)
    g[28:28 + thick, 10:60] = True
    g[28:28 + thick, 60 + gap:130] = True
    closed, _ = W.close_grid(g)
    n = ndimage.label(closed, structure=np.ones((3, 3)))[1]
    assert (n == 1) == bridged


def test_chain_code_length_is_unbiased_at_15_degrees():
    """A 10 m wall at 15 degrees, thinned to one cell: Kulpa's weights land within
    2 %, the naive 1 / sqrt(2) count is 7 % long."""
    g = np.zeros((400, 600), bool)
    th = np.radians(15)
    for s in np.linspace(0, 10, 4000):
        x, y = 0.5 + s * np.cos(th), 0.5 + s * np.sin(th)
        i, j = int(y / W.CELL), int(x / W.CELL)
        g[i - 1:i + 2, j - 1:j + 2] = True
    sk = skeletonize(g)
    straight, diag = W.link_counts(sk)
    naive = (straight + np.sqrt(2) * diag) * W.CELL
    assert W.pixel_length(sk) == pytest.approx(10.0, rel=0.02)
    assert naive > 10.0 * 1.05


def test_split_cuts_an_l_at_its_corner_and_tls_recovers_angles():
    rng = np.random.default_rng(3)
    a = np.c_[np.arange(0, 2.0, 0.02), np.zeros(100)]
    b = np.c_[np.full(50, 2.0), np.arange(0.02, 1.02, 0.02)]
    P = np.vstack([a, b]) + rng.normal(0, 0.003, (150, 2))
    pieces = W.split_points(P)
    assert len(pieces) == 2
    angs = sorted(W.angle_of(*W.fit_line(P[i:j + 1])[:2]) for i, j in pieces)
    assert angs[0] == pytest.approx(0.0, abs=1.0) or angs[0] == pytest.approx(180.0, abs=1.0)
    assert angs[1] == pytest.approx(90.0, abs=1.0)


def test_merge_joins_collinear_and_keeps_a_step_apart():
    rng = np.random.default_rng(4)
    mk = lambda x0, x1, y: np.c_[np.arange(x0, x1, 0.02), np.full(int(round((x1 - x0) / 0.02)), y)] \
        + rng.normal(0, 0.003, (int(round((x1 - x0) / 0.02)), 2))
    runs = []
    for P in (mk(0, 2, 0.0), mk(3, 5, 0.0), mk(0, 2, 1.0), mk(3, 5, 1.25)):
        p0, p1, rms, worst = W.fit_line(P)
        runs.append(dict(p0=p0, p1=p1, pts=P, rms=rms, worst=worst))
    walls, frags, info = W.merge_collinear(runs)
    assert info["merges"] == 1 and len(walls) == 3
