# -*- coding: utf-8 -*-
#
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
#
# ---------------------------------------------------------------------------
# From raster wall evidence to clean vector walls, and a DXF you can open.
# ---------------------------------------------------------------------------
#
#   classified scan -> wall-evidence grid (the input, built quickly)
#   -> close the grid -> skeleton (centre line) -> pixel chains
#   -> straight runs (split + total least squares) -> merge collinear pieces
#   -> square, snap onto the raw points, join the corners
#   -> measure (lengths, seen share, base support, gaps, room)
#   -> DXF with layers and per-wall evidence, read back to prove it opens.
#
# Data: code/data/classroom_scene.ply, a classified classroom scan of
# 1,422,440 points (class 2 = wall). Any classified indoor scan works: point
# SCAN_FILE at yours and set WALL_CLASS / FLOOR_CLASS / CEILING_CLASS.
#
# Run:  python walls_to_vectors.py        (writes code/out/)
# Test: python -m pytest tests            (synthetic rooms with known answers)
#
# v2 (2026-09-30): order-independent levelling, chain-code length corrected,
# walls snapped onto their raw points (the skeleton sat up to 17.7 mm inside
# the room), seen share measured on raw points with each point owned by one
# wall, gap lengths without the off-by-one and radius shrink, base support
# and what sits behind each gap, degenerate inputs handled. See CHANGES_v2.md.
#
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, skeletonize
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
import ezdxf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:          # a redirected stream (pytest, io.StringIO)
    pass

# --- configuration ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_FILE = os.path.join(HERE, "data", "classroom_scene.ply")
OUT = os.path.join(HERE, "out")

SEED = 0
FLOOR_CLASS, WALL_CLASS, CEILING_CLASS, DOOR_CLASS = 1, 2, 0, 6

CELL = 0.02            # grid cell, metres
BAND_MARGIN = 0.30     # keep wall points this far from floor and ceiling
MIN_HITS = 2           # points per cell to call it wall evidence
CLOSE_RADIUS = 2       # closing disk radius, in cells (bridges 2 cells in a 3-cell band, tested)
MIN_BLOB = 25          # connected wall blobs smaller than this are noise (cells)
SPUR_LEN = 0.15        # skeleton twigs shorter than this are pruned (m)
SPLIT_TOL = 0.04       # a straight run may bend by at most this much (m)
MIN_RUN = 0.30         # shorter runs are not walls (m)
MERGE_ANGLE = 5.0      # two runs are one wall within this angle (deg)
MERGE_OFFSET = 0.10    # ... and one line explains both within this (m)
MERGE_TILT = 2.0       # ... without tilting off a piece of 1 m or more by more than this (deg)
MIN_WALL = 0.75        # merged pieces shorter than this are fragments (m)
SNAP_ANGLE = 8.0       # walls this close to the room axis are squared (deg)
SNAP = True            # move each wall onto the median of its raw points
SNAP_WINDOWS = (0.10, 0.04)   # two passes: points this close to the line vote (m)
SNAP_TRIM = 0.20       # ignore this much at each wall end (the next wall lives there)
SNAP_MIN_POINTS = 20   # fewer voters than this: leave the wall where it is
JOIN_DIST = 0.75       # a wall end this close to a crossing is joined (m)
SEEN_RADIUS = 0.06     # a station is "seen" if a wall point lies this close to the line (m)
STATION = 0.02         # stations along each wall, one per 2 cm
SEEN_MIN = 0.50        # below this share of seen length, a wall is assumed
GAP_MIN = 0.40         # unseen stretches longer than this are reported (m)
RECESS_MAX = 0.60      # look this far behind a gap for a recess or a door (m)
DIM_OFFSET = 0.60      # dimension lines sit this far outside the room (m)

TIMES = {}


def say(step, msg):
    print(f"[{step}] {msg}")


def tick(name, t0):
    TIMES[name] = round((time.perf_counter() - t0) * 1000, 1)


# --- step 1: the wall-evidence grid (input) --------------------------------
def read_ply(path):
    """Binary little-endian PLY reader for float/uchar vertex properties."""
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4",
             "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                names.append((name, types[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(names), count=n)


def level_rotation(floor_pts):
    """Rotation that turns the floor's total-least-squares normal onto +Z.
    Uses every floor point (the 3 x 3 scatter matrix), so the answer does not
    depend on the order the points arrive in."""
    if len(floor_pts) < 3:
        return np.eye(3), 0.0
    c = floor_pts - floor_pts.mean(axis=0)
    n = np.linalg.eigh(c.T @ c)[1][:, 0]              # least-variance direction
    n = n if n[2] > 0 else -n
    v, cth = np.cross(n, [0, 0, 1]), n[2]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3), 0.0
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
    return R, float(np.degrees(np.arccos(np.clip(cth, -1, 1))))


def wall_evidence_grid(xyz, cls):
    """Classified points -> levelled wall band -> 2 cm occupancy grid."""
    xyz = np.asarray(xyz, np.float64)
    cls = np.asarray(cls).astype(int)
    if not (cls == WALL_CLASS).any():
        raise ValueError(f"no points of class {WALL_CLASS} (WALL_CLASS): nothing to draw")
    say("Step 1", f"{len(xyz):,} points, {int((cls == WALL_CLASS).sum()):,} wall")
    R, tilt = level_rotation(xyz[cls == FLOOR_CLASS])
    xyz = xyz @ R.T
    z = xyz[:, 2]
    wz = z[cls == WALL_CLASS]
    floor_z = float(np.median(z[cls == FLOOR_CLASS])) if (cls == FLOOR_CLASS).any() \
        else float(np.percentile(wz, 0.5))
    ceil_z = float(np.median(z[cls == CEILING_CLASS])) if (cls == CEILING_CLASS).any() \
        else float(np.percentile(wz, 99.5))
    wall = cls == WALL_CLASS
    band = wall & (z > floor_z + BAND_MARGIN) & (z < ceil_z - BAND_MARGIN)
    if not band.any():
        raise ValueError("no wall points between floor and ceiling: check the class ids")
    low = wall & (z <= floor_z + BAND_MARGIN)          # skirting, the base of the wall
    xy = xyz[band, :2]
    origin = xy.min(axis=0) - 10 * CELL
    shape = (int(np.ceil((xy[:, 1].max() - origin[1]) / CELL)) + 10,
             int(np.ceil((xy[:, 0].max() - origin[0]) / CELL)) + 10)
    ij = np.floor((xy - origin) / CELL).astype(int)
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    grid = counts >= MIN_HITS
    say("Step 1", f"tilt {tilt:.2f} deg, room height {ceil_z - floor_z:.2f} m, "
                  f"{band.sum():,} band points, grid {shape[0]} x {shape[1]}, "
                  f"{int(grid.sum()):,} wall cells")
    info = dict(points=int(len(xyz)), wall_points=int(wall.sum()),
                band_points=int(band.sum()), low_wall_points=int(low.sum()),
                tilt_deg=round(tilt, 2), room_height_m=round(ceil_z - floor_z, 2),
                grid_shape=list(shape), wall_cells=int(grid.sum()))
    ev = dict(band_xy=xy, low_xy=xyz[low, :2], door_xy=xyz[cls == DOOR_CLASS, :2])
    return grid, origin, xyz, band, floor_z, ev, info


# --- step 2: close the wall grid -------------------------------------------
def close_grid(grid):
    """Bridge the small holes a scan leaves in a wall, then drop dust blobs.
    What a disk of radius r bridges depends on the band's thickness: with
    r = 2, a 2-cell gap in a 3-cell band, a 4-cell gap in a 5-cell band, and
    nothing in a band 1 or 2 cells thick (tests/test_walls_to_vectors.py)."""
    pad = CLOSE_RADIUS + 2
    g = np.pad(grid, pad)
    if CLOSE_RADIUS > 0:
        closed = ndimage.binary_closing(g, structure=disk(CLOSE_RADIUS))[pad:-pad, pad:-pad]
    else:
        closed = grid.copy()
    closed |= grid
    lab, n = ndimage.label(closed, structure=np.ones((3, 3)))
    sizes = ndimage.sum(closed, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_BLOB))
    filled = int((keep & ~grid).sum())
    dropped = int(n - (sizes >= MIN_BLOB).sum())
    say("Step 2", f"closing filled {filled} cells, dropped {dropped} dust blobs, "
                  f"{int(keep.sum()):,} cells kept")
    return keep, dict(cells_filled=filled, blobs_dropped=dropped,
                      cells_after_close=int(keep.sum()))


# --- step 3: skeleton ------------------------------------------------------
def link_counts(skel):
    """Straight and diagonal links between on-cells of a one-cell-wide curve."""
    s = skel.astype(np.int32)
    straight = int((s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum())
    diag = int((s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum())
    return straight, diag


def pixel_length(skel):
    """Length of a one-cell-wide curve with Kulpa's chain-code weights.
    Counting 1 per straight link and sqrt(2) per diagonal overestimates a line at
    15 degrees by 7 %; Kulpa's 0.948 / 1.343 are unbiased over all directions."""
    straight, diag = link_counts(skel)
    return (0.948 * straight + 1.343 * diag) * CELL


def centre_line(closed):
    """Peel the wall band down to the one-cell line along its middle."""
    skel = skeletonize(closed)
    straight, diag = link_counts(skel)
    naive = (straight + np.sqrt(2) * diag) * CELL
    say("Step 3", f"skeleton {int(skel.sum()):,} cells, about {pixel_length(skel):.2f} m "
                  f"of centre line (naive 8-neighbour count {naive:.2f} m)")
    return skel, dict(skeleton_cells=int(skel.sum()),
                      centre_line_m=round(pixel_length(skel), 2),
                      centre_line_naive_m=round(float(naive), 2))


# --- step 4: trace the skeleton into ordered chains -------------------------
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_chains(skel):
    """Split the skeleton at ends and junctions and walk each branch in order."""
    on = set((int(r), int(c)) for r, c in zip(*np.nonzero(skel)))

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in OFFS
                if (p[0] + dy, p[1] + dx) in on]

    deg = {p: len(nbrs(p)) for p in on}
    nodes = {p for p, d in deg.items() if d != 2}
    seen_edges, chains = set(), []

    def walk(a, b):
        chain = [a, b]
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in nbrs(cur) if q != prev and (cur, q) not in seen_edges]
            if not nxt:
                break
            seen_edges.update({(cur, nxt[0]), (nxt[0], cur)})
            prev, cur = cur, nxt[0]
            chain.append(cur)
            if cur == a:
                break
        return chain

    for a in sorted(nodes):                             # sorted: same order every run
        for b in nbrs(a):
            if (a, b) in seen_edges:
                continue
            seen_edges.update({(a, b), (b, a)})
            chains.append(walk(a, b))
    left = on - {p for c in chains for p in c}          # pure loops, no node
    while left:
        a = min(left)
        b = nbrs(a)[0]
        seen_edges.update({(a, b), (b, a)})
        c = walk(a, b)
        chains.append(c)
        left -= set(c)
    return chains, nodes, deg


def chain_len(c):
    a = np.asarray(c, float)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) * CELL


def chains_from_skeleton(skel):
    chains, nodes, deg = trace_chains(skel)
    ends = {p for p in nodes if deg[p] == 1}
    spurs = [c for c in chains
             if (c[0] in ends or c[-1] in ends) and chain_len(c) < SPUR_LEN
             and not (c[0] in ends and c[-1] in ends)]
    kept = [c for c in chains if not any(c is s for s in spurs)]
    say("Step 4", f"{len(chains)} chains, {len(spurs)} spurs pruned, "
                  f"{len(kept)} kept, {len(ends)} end cells, "
                  f"{sum(1 for p in nodes if deg[p] >= 3)} junction cells")
    return kept, dict(chains=len(chains), spurs_pruned=len(spurs),
                      chains_kept=len(kept), end_cells=len(ends),
                      junction_cells=sum(1 for p in nodes if deg[p] >= 3))


# --- step 5: straight runs -------------------------------------------------
def to_metres(rc, origin):
    """(row, col) cell centres -> (x, y) metres in the scan frame."""
    rc = np.asarray(rc, float)
    return np.stack([origin[0] + (rc[:, 1] + 0.5) * CELL,
                     origin[1] + (rc[:, 0] + 0.5) * CELL], axis=1)


def split_points(P, tol=None):
    """Recursive split: cut an ordered chain where it bends away from its chord."""
    tol = SPLIT_TOL if tol is None else tol
    if len(P) < 3:
        return [(0, len(P) - 1)]
    a, b = P[0], P[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:                                        # a closed loop: farthest from a
        dist = np.linalg.norm(P - a, axis=1)
    else:
        dist = np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / L
    k = int(np.argmax(dist))
    if dist[k] <= tol or k in (0, len(P) - 1):
        return [(0, len(P) - 1)]
    left = split_points(P[:k + 1], tol)
    right = split_points(P[k:], tol)
    return left + [(i + k, j + k) for i, j in right]


def fit_line(P):
    """Total least squares: the line through P that minimises perpendicular error."""
    c = P.mean(axis=0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ u
    r = (P - c) @ np.array([-u[1], u[0]])
    return c + t.min() * u, c + t.max() * u, float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def straight_runs(chains, origin, tol=None):
    """Split every chain into straight pieces, fit each, keep the long ones."""
    runs, pieces = [], 0
    for c in chains:
        P = to_metres(c, origin)
        for i, j in split_points(P, tol):
            pieces += 1
            seg = P[i:j + 1]
            if len(seg) < 2:
                continue
            p0, p1, rms, worst = fit_line(seg)
            if np.linalg.norm(p1 - p0) >= MIN_RUN:
                runs.append(dict(p0=p0, p1=p1, pts=seg, rms=rms, worst=worst))
    lens = [np.linalg.norm(r["p1"] - r["p0"]) for r in runs]
    rms = [r["rms"] for r in runs] or [0.0]
    worst = [r["worst"] for r in runs] or [0.0]
    say("Step 5", f"{pieces} pieces, {len(runs)} straight runs of {MIN_RUN} m or more, "
                  f"{sum(lens):.2f} m, median fit rms {1000 * float(np.median(rms)):.1f} mm, "
                  f"worst cell {1000 * max(worst):.1f} mm off its line")
    return runs, dict(pieces=pieces, runs=len(runs), runs_total_m=round(float(sum(lens)), 2),
                      worst_run_rms_mm=round(1000 * max(rms), 1),
                      median_run_rms_mm=round(1000 * float(np.median(rms)), 1),
                      worst_cell_off_line_mm=round(1000 * max(worst), 1))


# --- step 6: merge collinear pieces of one wall -----------------------------
def angle_of(p0, p1):
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))) % 180.0


def direction(P):
    """Unit TLS direction of P and its extent along it."""
    c = P.mean(axis=0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ u
    return u, float(t.max() - t.min())


def one_line_explains(P, Q, offset=None):
    """Fit one line through both pieces; accept if 95 % of each piece's cells sit
    within the offset of it. One guard: when the longer piece is long enough to
    know its own direction (1 m or more), the shared line may not tilt away from
    it by more than MERGE_TILT. Without it, a tilted line can thread two parallel
    pieces with a step between them (a recess, a pilaster) and call them one."""
    offset = MERGE_OFFSET if offset is None else offset
    both = np.vstack([P, Q])
    c = both.mean(axis=0)
    u = np.linalg.svd(both - c, full_matrices=False)[2][0]
    nrm = np.array([-u[1], u[0]])
    worst = max(np.percentile(np.abs((P - c) @ nrm), 95),
                np.percentile(np.abs((Q - c) @ nrm), 95))
    (up, lp), (uq, lq) = direction(P), direction(Q)
    ul, ll = (up, lp) if lp >= lq else (uq, lq)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(ul @ u))))))
    if ll >= 1.0 and tilt > MERGE_TILT:
        return False, float(worst)
    return worst <= offset, float(worst)


def merge_collinear(runs):
    """Greedy merge: two groups become one wall when their directions agree and
    a single total-least-squares line explains both. Short, noisy pieces are
    judged by that shared line, never by their own shaky angle."""
    groups = [dict(pts=r["pts"], members=1,
                   ang=angle_of(r["p0"], r["p1"]),
                   L=float(np.linalg.norm(r["p1"] - r["p0"]))) for r in runs]
    merged_pairs = 0
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(groups)), key=lambda i: (-groups[i]["L"], i))
        for a in order:
            for b in order:
                if a == b or groups[a] is None or groups[b] is None:
                    continue
                ga, gb = groups[a], groups[b]
                da = abs(ga["ang"] - gb["ang"])
                # the angle gate loosens for short pieces: a 0.5 m run can't
                # hold its direction to better than a few degrees
                tol = MERGE_ANGLE + np.degrees(np.arctan2(2 * CELL, min(ga["L"], gb["L"])))
                if min(da, 180 - da) > tol:
                    continue
                ok, _ = one_line_explains(ga["pts"], gb["pts"])
                if not ok:
                    continue
                P = np.vstack([ga["pts"], gb["pts"]])
                p0, p1, _, _ = fit_line(P)
                groups[a] = dict(pts=P, members=ga["members"] + gb["members"],
                                 ang=angle_of(p0, p1), L=float(np.linalg.norm(p1 - p0)))
                groups[b] = None
                merged_pairs += 1
                changed = True
        groups = [g for g in groups if g is not None]
    pieces = []
    for g in groups:
        p0, p1, rms, _ = fit_line(g["pts"])
        pieces.append(dict(p0=p0, p1=p1, members=g["members"], rms=rms,
                           length=float(np.linalg.norm(p1 - p0))))
    pieces.sort(key=lambda w: (-w["length"], w["p0"][0], w["p0"][1]))
    walls = [w for w in pieces if w["length"] >= MIN_WALL]
    frags = [w for w in pieces if w["length"] < MIN_WALL]
    total = sum(w["length"] for w in walls)
    frag_txt = ", ".join("%.2f" % f["length"] for f in frags)
    say("Step 6", f"{len(runs)} runs -> {len(pieces)} pieces after {merged_pairs} merges: "
                  f"{len(walls)} walls ({total:.2f} m), {len(frags)} fragments "
                  f"({frag_txt} m)")
    return walls, frags, dict(pieces=len(pieces), merges=merged_pairs, walls=len(walls),
                              walls_total_m=round(total, 2),
                              wall_lengths_m=[round(w["length"], 2) for w in walls],
                              members=[w["members"] for w in walls],
                              fragments=len(frags),
                              fragment_lengths_m=[round(f["length"], 2) for f in frags])


# --- step 7: square, snap and join -----------------------------------------
def room_axis(walls):
    """Length-weighted mean direction, folded modulo 90 degrees (angle x 4)."""
    L = np.array([np.linalg.norm(w["p1"] - w["p0"]) for w in walls])
    th = np.radians([angle_of(w["p0"], w["p1"]) for w in walls])
    return float(np.degrees(np.arctan2((L * np.sin(4 * th)).sum(),
                                       (L * np.cos(4 * th)).sum()) / 4)) % 90.0


def off_axis(a, axis):
    d = (a - axis) % 90.0
    return min(d, 90.0 - d)


def cross_point(a0, a1, b0, b1):
    da, db = a1 - a0, b1 - b0
    den = da[0] * db[1] - da[1] * db[0]
    if abs(den) < 1e-9:
        return None
    t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / den
    return a0 + t * da


def frame(w):
    """Unit direction, unit left normal and length of a wall."""
    d = w["p1"] - w["p0"]
    L = float(np.linalg.norm(d))
    u = d / L
    return u, np.array([-u[1], u[0]]), L


def snap_to_points(walls, pts):
    """Slide each wall along its normal onto the median of its raw wall points.
    The skeleton is the middle of the evidence band, and anything fixed to a wall
    (a board, a tray, a radiator) thickens that band towards the room. The median
    of the points closest to the line ignores those and lands on the face."""
    shifts = []
    for w in walls:
        u, n, L = frame(w)
        total = 0.0
        for win in SNAP_WINDOWS:
            rel = pts - w["p0"]
            t, d = rel @ u, rel @ n
            m = (np.abs(d) <= win) & (t >= SNAP_TRIM) & (t <= L - SNAP_TRIM)
            if m.sum() < SNAP_MIN_POINTS:
                break
            off = float(np.median(d[m]))
            w["p0"], w["p1"] = w["p0"] + off * n, w["p1"] + off * n
            total += off
        w["snap_mm"] = 1000 * total
        shifts.append(total)
    return shifts


def square_and_join(walls, pts=None):
    if not walls:
        return walls, dict(room_axis_deg=None, max_off_axis_before_deg=0.0,
                           max_off_axis_after_deg=0.0, snap_mm=[], corners_joined=0,
                           tees_joined=0, max_corner_gap_before_cm=0.0,
                           max_corner_gap_after_cm=0.0)
    axis = room_axis(walls)
    before_off = max(off_axis(angle_of(w["p0"], w["p1"]), axis) for w in walls)
    for w in walls:                                     # square
        a = angle_of(w["p0"], w["p1"])
        if off_axis(a, axis) <= SNAP_ANGLE:
            target = min([axis, axis + 90, axis + 180, axis - 90],
                         key=lambda t: min(abs(a - t) % 180, 180 - abs(a - t) % 180))
            c = (w["p0"] + w["p1"]) / 2
            half = np.linalg.norm(w["p1"] - w["p0"]) / 2
            u = np.array([np.cos(np.radians(target)), np.sin(np.radians(target))])
            if np.dot(u, w["p1"] - w["p0"]) < 0:
                u = -u
            w["p0"], w["p1"] = c - half * u, c + half * u
    after_off = max(off_axis(angle_of(w["p0"], w["p1"]), axis) for w in walls)
    shifts = snap_to_points(walls, pts) if (SNAP and pts is not None) else [0.0] * len(walls)

    pairs = []                                          # join
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            da = abs(angle_of(a["p0"], a["p1"]) - angle_of(b["p0"], b["p1"]))
            if min(da, 180 - da) < 30:
                continue
            X = cross_point(a["p0"], a["p1"], b["p0"], b["p1"])
            if X is None:
                continue
            ea = min(("p0", "p1"), key=lambda k: np.linalg.norm(a[k] - X))
            eb = min(("p0", "p1"), key=lambda k: np.linalg.norm(b[k] - X))
            pairs.append((np.linalg.norm(a[ea] - X) + np.linalg.norm(b[eb] - X),
                          i, j, ea, eb, X))
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))

    def inside(w, p):
        t = np.dot(p - w["p0"], w["p1"] - w["p0"]) / np.dot(w["p1"] - w["p0"],
                                                           w["p1"] - w["p0"])
        return 0.0 < t < 1.0

    gaps_before, corners, tees, used = [], 0, 0, set()
    for _, i, j, ea, eb, X in pairs:
        a, b = walls[i], walls[j]
        da, db = np.linalg.norm(a[ea] - X), np.linalg.norm(b[eb] - X)
        if da <= JOIN_DIST and db <= JOIN_DIST and (i, ea) not in used and (j, eb) not in used:
            gaps_before.append(float(np.linalg.norm(a[ea] - b[eb])))
            a[ea], b[eb] = X.copy(), X.copy()
            used.update({(i, ea), (j, eb)})
            corners += 1
        elif da <= JOIN_DIST and inside(b, X) and (i, ea) not in used:
            a[ea] = X.copy()
            used.add((i, ea))
            tees += 1
        elif db <= JOIN_DIST and inside(a, X) and (j, eb) not in used:
            b[eb] = X.copy()
            used.add((j, eb))
            tees += 1
    gaps_after = []
    for (i, ea) in used:
        others = [walls[j][k] for (j, k) in used if j != i]
        if others:
            gaps_after.append(min(np.linalg.norm(walls[i][ea] - o) for o in others))
    info = dict(room_axis_deg=round(axis, 2), max_off_axis_before_deg=round(before_off, 2),
                max_off_axis_after_deg=round(after_off, 4),
                snap_mm=[round(1000 * s, 1) for s in shifts],
                corners_joined=corners, tees_joined=tees,
                max_corner_gap_before_cm=round(100 * max(gaps_before), 1) if gaps_before else 0.0,
                max_corner_gap_after_cm=round(100 * max(gaps_after), 2) if gaps_after else 0.0)
    say("Step 7", f"axis {axis:.2f} deg, off-axis {before_off:.2f} -> {after_off:.4f} deg, "
                  f"snapped by {', '.join('%+.1f' % (1000 * s) for s in shifts)} mm, "
                  f"{corners} corners + {tees} tees joined, largest corner gap "
                  f"{info['max_corner_gap_before_cm']} cm -> {info['max_corner_gap_after_cm']} cm")
    return walls, info


# --- step 8: measure -------------------------------------------------------
def seg_dist(P, a, b):
    """Distance from each point of P to the segment ab."""
    d = b - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def owners(P, walls):
    """Index of the nearest wall for every point (each point counts for one wall)."""
    if len(P) == 0 or not walls:
        return np.zeros(len(P), int)
    D = np.stack([seg_dist(P, w["p0"], w["p1"]) for w in walls], axis=1)
    return np.argmin(D, axis=1)


def stations_hit(w, P):
    """Boolean per 2 cm station along wall w: does a point of P fall on it?"""
    u, n, L = frame(w)
    nst = max(1, int(round(L / STATION)))
    rel = P - w["p0"]
    t, d = rel @ u, rel @ n
    m = (np.abs(d) <= SEEN_RADIUS) & (t >= 0) & (t < L)
    hit = np.zeros(nst, bool)
    hit[np.minimum((t[m] / L * nst).astype(int), nst - 1)] = True
    return hit, t[m], d[m]


def unseen_gaps(hit, L):
    """Runs of empty stations longer than GAP_MIN, as (t_start, t_end) in metres.
    A run of k empty stations is k stations long, measured edge to edge."""
    nst, gaps, i = len(hit), [], 0
    step = L / nst
    while i < nst:
        if hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < nst and not hit[j + 1]:
            j += 1
        if (j - i + 1) * step >= GAP_MIN:
            gaps.append((i * step, (j + 1) * step))
        i = j + 1
    return gaps


def behind_gap(w, g, outward, ev):
    """What sits behind an unseen stretch: a recess (wall points set back) and
    door-class points (their extent along the wall)."""
    u, n, L = frame(w)
    out = {}
    rel = ev["band_xy"] - w["p0"]
    t, d = rel @ u, (rel @ n) * outward
    m = (t > g[0]) & (t < g[1]) & (d > SEEN_RADIUS) & (d < RECESS_MAX)
    if m.sum() >= 50:
        out["recess_depth_m"] = round(float(np.median(d[m])), 2)
    if len(ev["door_xy"]):
        rel = ev["door_xy"] - w["p0"]
        t, d = rel @ u, (rel @ n) * outward
        m = (t > g[0] - 0.3) & (t < g[1] + 0.3) & (np.abs(d) < RECESS_MAX)
        if m.sum() >= 200:
            lo, hi = np.percentile(t[m], [1, 99])
            out["door_points"] = int(m.sum())
            out["door_width_m"] = round(float(hi - lo), 2)
    return out


def measure(walls, ev):
    """Length, seen share, base support, unseen gaps, and the room the walls enclose."""
    lines = []
    for w in walls:                                     # tiny overshoot so ends meet
        u, n, L = frame(w)
        w["length"] = L
        lines.append(LineString([w["p0"] - 0.01 * u, w["p1"] + 0.01 * u]))
    rooms = sorted(polygonize(unary_union(lines)), key=lambda p: -p.area) if lines else []
    room = rooms[0] if rooms else None
    centre = np.array(room.centroid.coords[0]) if room is not None else \
        (np.mean([(w["p0"] + w["p1"]) / 2 for w in walls], axis=0) if walls else np.zeros(2))
    ob = owners(ev["band_xy"], walls)
    ol = owners(ev["low_xy"], walls)
    for k, w in enumerate(walls):
        u, n, L = frame(w)
        outward = -1.0 if np.dot(centre - w["p0"], n) > 0 else 1.0
        hit, _, _ = stations_hit(w, ev["band_xy"][ob == k])
        base, _, dl = stations_hit(w, ev["low_xy"][ol == k])
        w["seen"] = float(hit.mean())
        w["base"] = float(base.mean())
        w["base_offset_mm"] = float(1000 * np.median(dl * outward)) if len(dl) else None
        w["assumed"] = w["seen"] < SEEN_MIN
        w["gap_t"] = [] if w["assumed"] else unseen_gaps(hit, L)
        w["gaps"] = [(w["p0"] + a * u, w["p0"] + b * u) for a, b in w["gap_t"]]
        w["behind"] = [behind_gap(w, g, outward, ev) for g in w["gap_t"]]
    info = dict(walls=[dict(length_m=round(w["length"], 3), seen=round(w["seen"], 3),
                            base=round(w["base"], 3),
                            base_offset_mm=None if w["base_offset_mm"] is None
                            else round(w["base_offset_mm"], 1),
                            assumed=bool(w["assumed"]),
                            gaps_m=[round(b - a, 2) for a, b in w["gap_t"]],
                            behind_gaps=w["behind"])
                       for w in walls],
                total_wall_m=round(sum(w["length"] for w in walls), 2),
                rooms=len(rooms),
                room_area_m2=round(room.area, 2) if room else None,
                room_perimeter_m=round(room.length, 2) if room else None)
    say("Step 8", "walls " + "; ".join(
        f"{w['length']:.3f} m ({100 * w['seen']:.0f}% seen, base {100 * w['base']:.0f}%"
        + (", assumed" if w["assumed"] else "")
        + "".join(f", gap {b - a:.2f} m" for a, b in w["gap_t"]) + ")"
        for w in walls)
        + (f" | room {room.area:.2f} m2, perimeter {room.length:.2f} m" if room else " | no closed room"))
    return walls, room, info


# --- step 9: export DXF ----------------------------------------------------
def export_dxf(walls, frags, room, path):
    """One LINE per wall: WALL when the scan saw it, WALL_ASSUMED when it
    mostly did not, with its evidence attached as XDATA. Unseen stretches go on
    GAPS, leftovers on FRAGMENTS, the enclosed room on ROOM, an aligned
    dimension per wall on DIMENSIONS."""
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.header["$MEASUREMENT"] = 1
    doc.appids.add("LGO_EVIDENCE")
    doc.layers.add("WALL", color=7, lineweight=50)
    doc.layers.add("WALL_ASSUMED", color=30, linetype="DASHED", lineweight=50)
    doc.layers.add("GAPS", color=1, linetype="DASHED", lineweight=25)
    doc.layers.add("FRAGMENTS", color=8, lineweight=18)
    doc.layers.add("ROOM", color=8)
    doc.layers.add("DIMENSIONS", color=5)
    doc.styles.add("PLAN", font="arial.ttf")
    doc.header["$LTSCALE"] = 0.5
    msp = doc.modelspace()
    centre = np.array(room.centroid.coords[0]) if room is not None else \
        (np.mean([w["p0"] for w in walls], axis=0) if walls else np.zeros(2))
    for w in walls:
        layer = "WALL_ASSUMED" if w["assumed"] else "WALL"
        line = msp.add_line(tuple(w["p0"]), tuple(w["p1"]), dxfattribs={"layer": layer})
        line.set_xdata("LGO_EVIDENCE", [(1000, f"seen={w['seen']:.3f}"),
                                        (1000, f"base={w['base']:.3f}")])
        for a, b in w["gaps"]:
            msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "GAPS"})
        u, left, L = frame(w)
        mid = (w["p0"] + w["p1"]) / 2
        side = 1.0 if np.dot(left, mid - centre) > 0 else -1.0   # outward
        dim = msp.add_aligned_dim(p1=tuple(w["p0"]), p2=tuple(w["p1"]),
                                  distance=side * DIM_OFFSET, dimstyle="EZDXF",
                                  dxfattribs={"layer": "DIMENSIONS"},
                                  override={"dimdec": 2, "dimtxt": 0.18, "dimasz": 0.12,
                                            "dimlfac": 1.0, "dimpost": "<> m"})
        dim.render()
    for f in frags:
        msp.add_line(tuple(f["p0"]), tuple(f["p1"]), dxfattribs={"layer": "FRAGMENTS"})
    if room is not None:
        msp.add_lwpolyline(list(room.exterior.coords), close=True,
                           dxfattribs={"layer": "ROOM"})
        c = room.representative_point()
        msp.add_text(f"Room 1  {room.area:.1f} m2", height=0.25,
                     dxfattribs={"layer": "ROOM", "style": "PLAN"}).set_placement(
                         (c.x, c.y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(path)
    back = ezdxf.readfile(path)                         # prove it opens
    per_layer, tagged = {}, 0
    for e in back.modelspace():
        per_layer[e.dxf.layer] = per_layer.get(e.dxf.layer, 0) + 1
        if e.dxftype() == "LINE" and e.has_xdata("LGO_EVIDENCE"):
            tagged += 1
    size_kb = os.path.getsize(path) / 1024
    say("Step 9", f"wrote {os.path.basename(path)} ({size_kb:.1f} kB), read back: {per_layer}, "
                  f"{tagged} walls carry their evidence")
    return dict(dxf_kb=round(size_kb, 1), dxf_entities=per_layer, walls_with_evidence=tagged)


# --- the whole chain -------------------------------------------------------
def run_pipeline(xyz, cls, out_dir=None):
    """Every step on arrays. Returns (metrics, stages). Writes the DXF to
    out_dir when given. The tests call this on synthetic rooms."""
    TIMES.clear()
    m, st = {}, {}
    t = time.perf_counter()
    grid, origin, lxyz, band, floor_z, ev, m["grid"] = wall_evidence_grid(xyz, cls)
    tick("1_grid", t)
    t = time.perf_counter()
    closed, m["close"] = close_grid(grid)
    tick("2_close", t)
    t = time.perf_counter()
    skel, m["skeleton"] = centre_line(closed)
    tick("3_skeleton", t)
    t = time.perf_counter()
    chains, m["chains"] = chains_from_skeleton(skel)
    tick("4_chains", t)
    t = time.perf_counter()
    runs, m["runs"] = straight_runs(chains, origin)
    tick("5_runs", t)
    t = time.perf_counter()
    walls, frags, m["merge"] = merge_collinear(runs)
    merged_copy = [dict(p0=w["p0"].copy(), p1=w["p1"].copy()) for w in walls]
    tick("6_merge", t)
    t = time.perf_counter()
    walls, m["square"] = square_and_join(walls, ev["band_xy"])
    tick("7_square", t)
    t = time.perf_counter()
    walls, room, m["measure"] = measure(walls, ev)
    tick("8_measure", t)
    if out_dir:
        t = time.perf_counter()
        m["dxf"] = export_dxf(walls, frags, room, os.path.join(out_dir, "floor_plan.dxf"))
        tick("9_dxf", t)
    m["times_ms"] = dict(TIMES)
    m["total_ms"] = round(sum(TIMES.values()), 1)
    st.update(origin=origin, grid=grid, closed=closed, skel=skel, chains=chains, runs=runs,
              merged=merged_copy, walls=walls, frags=frags, room=room, xyz=lxyz,
              band=band, floor_z=floor_z, ev=ev)
    return m, st


# --- run -------------------------------------------------------------------
class Tee:
    """Print to the console and keep a copy in out/run_log.txt."""
    def __init__(self, path):
        self.f, self.con = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        self.con.write(s)

    def flush(self):
        self.con.flush()


def params():
    return dict(CELL=CELL, BAND_MARGIN=BAND_MARGIN, MIN_HITS=MIN_HITS,
                CLOSE_RADIUS=CLOSE_RADIUS, MIN_BLOB=MIN_BLOB, SPUR_LEN=SPUR_LEN,
                SPLIT_TOL=SPLIT_TOL, MIN_RUN=MIN_RUN, MERGE_ANGLE=MERGE_ANGLE,
                MERGE_OFFSET=MERGE_OFFSET, MERGE_TILT=MERGE_TILT, SNAP_ANGLE=SNAP_ANGLE, SNAP=SNAP,
                SNAP_WINDOWS=list(SNAP_WINDOWS), SNAP_TRIM=SNAP_TRIM,
                JOIN_DIST=JOIN_DIST, SEEN_RADIUS=SEEN_RADIUS, STATION=STATION,
                SEEN_MIN=SEEN_MIN, MIN_WALL=MIN_WALL, GAP_MIN=GAP_MIN,
                RECESS_MAX=RECESS_MAX, DIM_OFFSET=DIM_OFFSET)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT, "run_log.txt"))
    t = time.perf_counter()
    v = read_ply(SCAN_FILE)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    cls = np.rint(v["scalar_semantic_label"]).astype(int)
    read_ms = round((time.perf_counter() - t) * 1000, 1)
    m, st = run_pipeline(xyz, cls, OUT)
    m["read_ms"] = read_ms
    m["params"] = params()
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    rng = np.random.default_rng(SEED)
    lxyz = st["xyz"]
    keep = rng.choice(len(lxyz), min(400000, len(lxyz)), replace=False)
    walls, frags, room = st["walls"], st["frags"], st["room"]
    np.savez_compressed(
        os.path.join(OUT, "stages.npz"),
        origin=st["origin"], cell=CELL, floor_z=st["floor_z"], grid=st["grid"],
        closed=st["closed"], skel=st["skel"],
        cloud=lxyz[keep].astype(np.float32), cloud_cls=cls[keep].astype(np.int8),
        band_xy=st["ev"]["band_xy"].astype(np.float32),
        low_xy=st["ev"]["low_xy"].astype(np.float32),
        chains=np.array([np.asarray(c, np.int32) for c in st["chains"]], dtype=object),
        runs=np.array([[*r["p0"], *r["p1"]] for r in st["runs"]]).reshape(-1, 4),
        merged=np.array([[*w["p0"], *w["p1"]] for w in st["merged"]]).reshape(-1, 4),
        walls=np.array([[*w["p0"], *w["p1"], w["seen"]] for w in walls]).reshape(-1, 5),
        frags=np.array([[*f["p0"], *f["p1"]] for f in frags]).reshape(-1, 4),
        gaps=np.array([[*a, *b] for w in walls for a, b in w["gaps"]]).reshape(-1, 4),
        room=np.asarray(room.exterior.coords) if room is not None else np.zeros((0, 2)))
    print(f"done in {m['total_ms']:.0f} ms after reading the scan in {read_ms:.0f} ms, "
          f"outputs in {OUT}")
    tee = sys.stdout
    sys.stdout = tee.con
    tee.f.close()


if __name__ == "__main__":
    main()
