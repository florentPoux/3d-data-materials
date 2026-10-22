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
"""
Accessibility audit from a classified indoor scan: can a wheelchair fit and turn?

Input : a classified point cloud of a building floor (S3DIS Area 2 part, 14 classes).
Output: out/audit_grids.npz   every grid the audit builds (for figures and the 3D replay)
        out/issues.csv        the numbered issue list (the checklist handed over)
        out/summary.json      every number printed below
        out/metrics.json      the headline numbers, flat, with the input's fingerprint

Steps (each one prints what it measured):
  1. load the scan and count its classes
  2. build the walkable grid (floor height, obstacles, stairs, height jumps)
  3. clear width everywhere from a Euclidean distance transform
  4. where the wheelchair's centre can be (the configuration space)
  5. doors measured against the required width
  6. turning circles by a morphological opening (element = full diameter)
  7. reachability from the entrances by connected components
  8. why each room is cut off + the numbered issue list
  9. act on it: widen one door, ramp one flight, and re-judge

CPU only, numpy + scipy. Deterministic: no randomness in the audit itself, and the result
does not depend on the order of the points (tests/test_audit.py checks it, with known-answer
scenes for every measurement; check_stability.py checks it on the real scan).

    python accessibility_audit.py [--data scan.ply] [--out folder]
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "s3dis_area2_access.ply"
OUT = HERE / "out"

# ---------------------------------------------------------------- configuration
CELL = 0.05                       # grid cell, metres
LAYER = 0.08                      # a cell's floor = its lowest 8 cm of floor points
FILL_M = 0.30                     # fill a cell with no floor from a neighbour this close
OBST_LO, OBST_HI = 0.10, 1.80     # anything in this band above the floor is in the way
MIN_PTS = 2                       # points needed before a cell counts as wall/door/stairs
JUMP_GUARD = 0.05                 # coarse guard: a 3x3 height range above this is not rolled over
ENTRANCES = [(22.58, 2.98), (22.57, -7.11)]   # the two east doors of the example
SEED_M = 0.75                     # an entrance seeds every passable cell this close
DOOR_MIN_M2 = 0.03                # a door is at least this much door-class floor plan
DOOR_WINDOW_M = 1.2               # each door is measured again in a window this far around it
DOOR_MIN_PTS = 8                  # wall points that make a 2.5 cm cell a jamb (a jamb is tall: a stray pair is not one)
EDGE_CELLS, EDGE_M2 = 3, 0.03      # a door with this much unscanned ground within 15 cm is where the scan stops
DOOR_UNMEASURED_M = 0.30          # an opening read narrower than this is a closed leaf or a scan edge
ROOM_MIN_M2 = 2.0                 # smaller pockets are not rooms
ROOM_CORE_M2 = 1.0                # ... and at least this much floor once the leaks are sealed
ROOM_SEAL_M = 0.50               # a gap narrower than a walker does not join two rooms
REACH_MIN_M2 = 0.5                # a room counts as reached when this much of it is

# The profile the audit judges against. Metric wheelchair profile (the default) and the
# ADA 2010 figures for comparison. Walking is the on-foot reference.
PROFILES = {
    "wheelchair": dict(passage=0.90, door=0.80, turn=1.50, body=0.45, stairs=False),
    "ada2010":    dict(passage=0.915, door=0.815, turn=1.525, body=0.46, stairs=False),
    "walking":    dict(passage=0.50, door=0.50, turn=None, body=0.25, stairs=True),
}
PROFILE = "wheelchair"

CLASSES = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table",
           "chair", "sofa", "bookcase", "board", "stairs", "clutter"]
FLOOR, DOOR, STAIRS = 1, 6, 12
STRUCTURE = [2, 3, 4, 5, 11]                  # wall, beam, column, window, board
FURNITURE = [7, 8, 9, 10, 13]                 # table, chair, sofa, bookcase, clutter

T = {}
S = {}


def tick(name, t0):
    T[name] = round((time.perf_counter() - t0) * 1000, 1)


def disk(diameter_cells):
    """A filled disk structuring element whose FULL width is diameter_cells (odd)."""
    r = diameter_cells // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def odd_cells(metres):
    """Metres to a whole, odd number of cells, rounded up (CONTENT L74: never half)."""
    n = int(np.ceil(metres / CELL - 1e-9))
    return n if n % 2 == 1 else n + 1


def se_cells(metres):
    """A clear space of `metres`, face to face, as a structuring element of FREE cells.

    The element is the full diameter, never the radius (CONTENT L74). The wall points claim
    one cell of every clear space (see clear_width), so 1.50 m of floor between two walls
    shows up as 29 free cells; rounded up to an odd count so the disk has a centre."""
    n = int(np.ceil((metres - CELL) / CELL - 1e-9))
    return n if n % 2 == 1 else n + 1


# %% Step 1. Load the scan and count what it holds
def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        props = [ln.split()[1:] for ln in header.decode().splitlines() if ln.startswith("property")]
        kind = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}
        dt = np.dtype([(name, kind[t]) for t, name in props])
        return np.fromfile(f, dtype=dt)


def step1_load():
    t0 = time.perf_counter()
    v = read_ply(DATA)
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    rgb = np.column_stack([v["red"], v["green"], v["blue"]])
    cls = v["classification"].astype(np.int16)
    counts = {CLASSES[k]: int((cls == k).sum()) for k in range(len(CLASSES)) if (cls == k).any()}
    tick("1_load", t0)
    S["points"] = len(xyz)
    S["class_counts"] = counts
    S["extent_m"] = [round(float(np.ptp(xyz[:, i])), 2) for i in range(3)]
    print(f"[1] {len(xyz):,} points, extent {S['extent_m']} m")
    for k, n in counts.items():
        print(f"    {k:<9} {n:>9,}")
    return xyz, rgb, cls


# %% Step 2. The walkable grid, in one pass
def step2_grid(xyz, cls):
    t0 = time.perf_counter()
    # the grid sits on a fixed world lattice (multiples of CELL), so moving the scan's
    # extreme point by a millimetre cannot shift every cell boundary (v1 anchored it there)
    lo = np.floor(xyz[:, :2].min(0) / CELL) * CELL - 2 * CELL
    nx, ny = (np.ceil((xyz[:, :2].max(0) - lo) / CELL).astype(int) + 2)
    ij = ((xyz[:, :2] - lo) / CELL).astype(np.int64)
    lin = ij[:, 0] * ny + ij[:, 1]
    z = xyz[:, 2]

    # floor height per cell: median of the lowest 8 cm of its floor points (a loft's floor
    # and the room under it share cells; the one rolled on is the lower one)
    f = cls == FLOOR
    zmin = np.full(nx * ny, np.inf)
    np.minimum.at(zmin, lin[f], z[f])
    keep = z[f] <= zmin[lin[f]] + LAYER
    lf, zf = lin[f][keep], z[f][keep]
    order = np.lexsort((zf, lf))
    lf, zf = lf[order], zf[order]
    cells, start, count = np.unique(lf, return_index=True, return_counts=True)
    H = np.full(nx * ny, np.nan)
    H[cells] = zf[start + count // 2]
    H = H.reshape(nx, ny)
    measured = ~np.isnan(H)

    # nearest measured height: filled within 30 cm (doorways, thresholds), and used far
    # away only to read how high an obstacle point stands
    dist, (ii, jj) = ndi.distance_transform_edt(~measured, return_indices=True)
    Hnear = H[ii, jj]
    has_floor = dist * CELL <= FILL_M
    has_floor[[0, -1], :] = False       # a closed frame: the distance transforms below
    has_floor[:, [0, -1]] = False       # must see "not free" beyond the scan

    def mask_of(sel, rel_band=None):
        l = lin[sel]
        if rel_band is not None:
            rel = z[sel] - Hnear.ravel()[l]
            l = l[(rel >= rel_band[0]) & (rel <= rel_band[1])]
        return (np.bincount(l, minlength=nx * ny) >= MIN_PTS).reshape(nx, ny)

    struct = mask_of(np.isin(cls, STRUCTURE), (OBST_LO, OBST_HI))
    furn = mask_of(np.isin(cls, FURNITURE), (OBST_LO, OBST_HI))
    door = mask_of(cls == DOOR)
    stairs = ndi.binary_dilation(mask_of(cls == STAIRS), iterations=1)
    stairs[[0, -1], :] = False
    stairs[:, [0, -1]] = False

    # the door leaf is not an obstacle: the opening is where people go through
    obstacle = (struct | furn) & ~door

    # coarse height-jump guard (the real step survey, measured to 2 cm, is a Neurones stage)
    Hmax = ndi.maximum_filter(np.where(measured, H, -np.inf), size=3)
    Hmin = ndi.minimum_filter(np.where(measured, H, np.inf), size=3)
    jump = measured & ((Hmax - Hmin) > JUMP_GUARD)

    free_foot = (has_floor | stairs) & ~obstacle
    free_chair = free_foot & ~stairs & ~jump
    tick("2_grid", t0)

    a = CELL * CELL
    S["grid"] = dict(cells=[int(nx), int(ny)], cell_m=CELL,
                     floor_measured_m2=round(measured.sum() * a, 1),
                     floor_filled_m2=round((has_floor & ~measured).sum() * a, 1),
                     obstacle_on_floor_m2=round((obstacle & has_floor).sum() * a, 1),
                     stairs_m2=round(stairs.sum() * a, 1),
                     jump_m2=round(jump.sum() * a, 1),
                     free_foot_m2=round(free_foot.sum() * a, 1),
                     free_chair_m2=round(free_chair.sum() * a, 1),
                     floor_z_range=[round(float(np.nanmin(H)), 2), round(float(np.nanmax(H)), 2)])
    print(f"[2] grid {nx} x {ny} at {CELL * 100:.0f} cm: {S['grid']}")
    return dict(lo=lo, nx=nx, ny=ny, lin=lin, H=H, Hnear=Hnear, measured=measured, has_floor=has_floor,
                struct=struct, furn=furn, door=door, stairs=stairs, obstacle=obstacle, jump=jump,
                free_foot=free_foot, free_chair=free_chair)


def cell_of(g, xy):
    return tuple(((np.asarray(xy) - g["lo"]) / CELL).astype(int))


# %% Step 3. Clear width everywhere, from one distance transform
def clear_width(free, cell=CELL):
    """The widest body whose centre can stand somewhere in this cell, face to face.

    Measured on a grid twice as fine (so a corridor an even number of cells wide has a
    centre line too), then one cell is given back: a wall point turns its whole cell into
    obstacle, so each face sits on average half a cell beyond the last free cell.
    For a straight run of k free cells this is exactly (k + 1) * cell."""
    fine = np.repeat(np.repeat(free, 2, axis=0), 2, axis=1)
    d = ndi.distance_transform_edt(fine)                       # in half-cells
    d = d.reshape(free.shape[0], 2, free.shape[1], 2).max(axis=(1, 3))
    return np.where(free, d * cell + cell, 0.0)


def step3_width(g, prof):
    t0 = time.perf_counter()
    W = clear_width(g["free_chair"])
    tick("3_width", t0)
    w = W[g["free_chair"]]
    a = CELL * CELL
    S["width"] = dict(median_m=round(float(np.median(w)), 2), max_m=round(float(w.max()), 2),
                      ge_passage_m2=round((w >= prof["passage"]).sum() * a, 1),
                      lt_passage_m2=round((w < prof["passage"]).sum() * a, 1),
                      ms=T["3_width"])
    print(f"[3] clear width: {S['width']}")
    return W


# %% Step 4. Where the wheelchair's centre can be
def centre_space(g, W, doorzone, prof):
    return g["free_chair"] & ((W >= prof["passage"]) | (doorzone & (W >= prof["door"])))


def step4_centre(g, W, prof):
    t0 = time.perf_counter()
    doorzone = ndi.binary_dilation(g["door"], structure=disk(odd_cells(0.8)))
    C = centre_space(g, W, doorzone, prof)
    # the floor the body covers: every cell within half the body width of a centre
    covered = (ndi.distance_transform_edt(~C) * CELL <= prof["body"]) & g["free_chair"]
    tick("4_centre", t0)
    a = CELL * CELL
    S["centre"] = dict(centre_m2=round(C.sum() * a, 1), covered_m2=round(covered.sum() * a, 1),
                       free_chair_m2=S["grid"]["free_chair_m2"])
    print(f"[4] centre space {S['centre']}")
    return C, doorzone


# %% Step 5. Doors, measured against the required width
def up2(a):
    """A 5 cm mask on the 2.5 cm grid (every cell becomes 2 x 2)."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def door_width(g, m, wall_xy):
    """One door's clear width, on a 2.5 cm grid of its own wall points.

    The 5 cm grid can only say a jamb is in this cell or the next, so a door read there
    moves by 5 cm when a few wall points are picked differently. Around each door the
    wall points are gridded again at half the cell, and the same width estimate as step 3
    runs there: the reading tightens to +-2.5 cm."""
    cells = np.argwhere(m)
    pad = int(round(DOOR_WINDOW_M / CELL))
    i0, j0 = np.maximum(cells.min(0) - pad, 0)
    i1, j1 = np.minimum(cells.max(0) + pad + 1, m.shape)
    x0 = g["lo"] + np.array([i0, j0]) * CELL
    fine_cell = CELL / 2
    shape = (2 * (i1 - i0), 2 * (j1 - j0))
    sel = ((wall_xy[:, 0] >= x0[0]) & (wall_xy[:, 1] >= x0[1])
           & (wall_xy[:, 0] < x0[0] + shape[0] * fine_cell) & (wall_xy[:, 1] < x0[1] + shape[1] * fine_cell))
    fij = np.floor((wall_xy[sel] - x0) / fine_cell).astype(np.int64)
    fij = np.minimum(fij, np.array(shape) - 1)
    wall = np.bincount(fij[:, 0] * shape[1] + fij[:, 1], minlength=shape[0] * shape[1]).reshape(shape) >= DOOR_MIN_PTS
    floor = up2((g["has_floor"] | g["stairs"])[i0:i1, j0:j1])
    through = floor & ~wall
    through[[0, -1], :] = False                 # the window is closed: nothing beyond it counts
    through[:, [0, -1]] = False
    inside = up2(m[i0:i1, j0:j1]) & through
    return float(clear_width(through, fine_cell)[inside].max()) if inside.any() else 0.0


def step5_doors(g, prof, xyz, cls):
    t0 = time.perf_counter()
    # merge the bits of one door (leaf, frame) that the grid splits
    lab, n = ndi.label(ndi.binary_dilation(g["door"], iterations=2), structure=np.ones((3, 3)))
    lab = np.where(g["door"], lab, 0)
    # the jambs: structure points 0.10-1.80 m above their floor (furniture is ignored here:
    # a chair in a doorway is an issue of use, and it shows up in step 7)
    sel = np.isin(cls, STRUCTURE)
    rel = xyz[sel, 2] - g["Hnear"].ravel()[g["lin"][sel]]
    wall_xy = xyz[sel][(rel >= OBST_LO) & (rel <= OBST_HI), :2]
    # unscanned ground: no floor within FILL_M, no wall, no door, no stairs
    void = ~(g["has_floor"] | g["stairs"] | g["struct"] | g["door"])
    doors = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * CELL * CELL < DOOR_MIN_M2:
            continue
        cells = np.argwhere(m)
        cx, cy = (cells.mean(0) + 0.5) * CELL + g["lo"]
        width = door_width(g, m, wall_xy)
        # not measurable: a closed leaf with no floor read through it, or a door where the
        # scan stops (the building's own entrances: nothing measured outside them)
        edge = (ndi.binary_dilation(m, iterations=EDGE_CELLS) & void).sum() * CELL * CELL >= EDGE_M2
        doors.append(dict(id=len(doors) + 1, x=round(float(cx), 2), y=round(float(cy), 2),
                          width_m=round(width, 3), measurable=bool(width >= DOOR_UNMEASURED_M and not edge),
                          scan_edge=bool(edge), cells=m))
    tick("5_doors", t0)
    ws = np.array([dd["width_m"] for dd in doors])
    measured = np.array([dd["measurable"] for dd in doors])
    S["doors"] = dict(count=len(doors), measured=int(measured.sum()),
                      unmeasured=int((~measured).sum()),
                      under_required=int((measured & (ws < prof["door"])).sum()),
                      narrowest_m=round(float(ws[measured].min()), 2),
                      median_m=round(float(np.median(ws[measured])), 2),
                      widest_m=round(float(ws[measured].max()), 2), required_m=prof["door"],
                      tolerance_m=CELL / 2, ms=T["5_doors"])
    print(f"[5] doors {S['doors']}")
    return doors


# %% Step 6. Turning circles by a morphological opening
def step6_turning(g, prof):
    t0 = time.perf_counter()
    n = se_cells(prof["turn"])             # the FULL diameter in free cells (29 = 1.50 m face to face)
    se = disk(n)
    level = g["free_chair"]
    centres = ndi.binary_erosion(level, structure=se)
    turn = ndi.binary_dilation(centres, structure=se) & level     # = opening
    tick("6_turning", t0)

    # the same centres from the distance transform, as a cross-check
    centres_edt = ndi.distance_transform_edt(level) > n // 2       # in cells, no rounding
    agree = float((centres_edt == centres).mean())

    # what the half-diameter mistake would have accepted (CONTENT L74)
    wrong = ndi.binary_opening(level, structure=disk(se_cells(prof["turn"] / 2)))
    a = CELL * CELL
    S["turning"] = dict(se_cells=n, se_m=round((n + 1) * CELL, 2), turn_m2=round(turn.sum() * a, 1),
                        centres_m2=round(centres.sum() * a, 1),
                        edt_agreement=round(agree, 5),
                        half_diameter_wrong_m2=round(wrong.sum() * a, 1), ms=T["6_turning"])
    print(f"[6] turning {S['turning']}")
    return centres, turn, wrong


# %% Step 7. Reachability from the entrances by connected components
def seeds_of(g, mask):
    """Every cell of the mask within SEED_M of an entrance (a disk, cell centre to point)."""
    ci = (np.arange(mask.shape[0]) + 0.5) * CELL + g["lo"][0]
    cj = (np.arange(mask.shape[1]) + 0.5) * CELL + g["lo"][1]
    s = np.zeros_like(mask)
    for ex, ey in ENTRANCES:
        s |= ((ci[:, None] - ex) ** 2 + (cj[None, :] - ey) ** 2) <= SEED_M ** 2
    return s & mask


def reach(g, mask):
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))          # 8-connected
    ids = np.unique(lab[seeds_of(g, mask)])
    ids = ids[ids > 0]
    return np.isin(lab, ids) & mask


def rooms_of(g):
    """Rooms: the floor cut by the structure and at the doors (a door is where one room ends).

    v1 labelled that floor directly, and a gap of one or two cells in a wall mask (a few wall
    points picked differently) joined two rooms or split one. Now the leaks are sealed first:
    an erosion by a walker's width (0.50 m) keeps each room's core, the cores are labelled,
    and the labels grow back over the floor they came from, one cell per pass, never across
    a wall. A gap no walker fits through no longer decides what a room is."""
    if "_rooms" in g:
        return g["_rooms"]
    base = (g["has_floor"] | g["stairs"]) & ~g["struct"] & ~ndi.binary_dilation(g["door"], iterations=2)
    core = ndi.binary_erosion(base, structure=disk(se_cells(ROOM_SEAL_M)))
    lab, n = ndi.label(core)
    core_m2 = np.bincount(lab.ravel(), minlength=n + 1)[1:] * CELL * CELL
    cross = ndi.generate_binary_structure(2, 1)
    for _ in range(200):                                   # up to 10 m of growth, far more than needed
        grown = ndi.grey_dilation(lab, footprint=cross)
        nxt = np.where((lab == 0) & base, grown, lab)
        if np.array_equal(nxt, lab):
            break
        lab = nxt
    sizes = ndi.sum(lab > 0, lab, index=np.arange(1, n + 1)) * CELL * CELL
    # a room has a core a body can stand in, not just an area: a sliver along a wall or a
    # scan edge has almost no core, and its area alone would flip around the threshold
    keep = np.flatnonzero((sizes >= ROOM_MIN_M2) & (core_m2 >= ROOM_CORE_M2)) + 1
    rooms = np.zeros_like(lab)
    for new, old in enumerate(keep[np.argsort(-sizes[keep - 1], kind="stable")], start=1):
        rooms[lab == old] = new
    g["_rooms"] = (rooms, len(keep))
    return g["_rooms"]


def covered_from(centre_reached, free, body):
    return (ndi.distance_transform_edt(~centre_reached) * CELL <= body) & free


def room_status(rooms, nrooms, cov_chair, cov_foot):
    a = CELL * CELL
    chair = ndi.sum(cov_chair, rooms, index=np.arange(1, nrooms + 1)) * a
    foot = ndi.sum(cov_foot, rooms, index=np.arange(1, nrooms + 1)) * a
    return np.where(chair >= REACH_MIN_M2, 2, np.where(foot >= REACH_MIN_M2, 1, 0))


def step7_reach(g, C, W, prof):
    t0 = time.perf_counter()
    Rc = reach(g, C)
    cov_c = covered_from(Rc, g["free_chair"], prof["body"])
    walk = PROFILES["walking"]
    Wf = clear_width(g["free_foot"])
    Cf = g["free_foot"] & (Wf >= walk["passage"])
    Rf = reach(g, Cf)
    cov_f = covered_from(Rf, g["free_foot"], walk["body"])
    rooms, nrooms = rooms_of(g)
    status = room_status(rooms, nrooms, cov_c, cov_f)
    tick("7_reach", t0)
    a = CELL * CELL
    S["reach"] = dict(rooms=int(nrooms), wheelchair=int((status == 2).sum()),
                      foot_only=int((status == 1).sum()), not_reached=int((status == 0).sum()),
                      on_foot_total=int((status >= 1).sum()),
                      chair_centre_reached_m2=round(Rc.sum() * a, 1),
                      chair_covered_m2=round(cov_c.sum() * a, 1),
                      foot_covered_m2=round(cov_f.sum() * a, 1), ms=T["7_reach"])
    print(f"[7] reach {S['reach']}")
    return dict(Rc=Rc, cov_c=cov_c, Cf=Cf, Rf=Rf, cov_f=cov_f, Wf=Wf, rooms=rooms,
                nrooms=nrooms, status=status)


# %% Step 8. Why each room is cut off, and the numbered issue list
def step8_issues(g, W, doorzone, R, doors, centres, prof):
    t0 = time.perf_counter()
    a = CELL * CELL
    rooms, nrooms, status = R["rooms"], R["nrooms"], R["status"]
    walk = PROFILES["walking"]

    # on foot but without the stairs and the height jumps: can the room be reached at all
    # without climbing? If not, the barrier is a change of level.
    flat_foot = g["free_chair"] & (W >= walk["passage"])
    cov_flat = covered_from(reach(g, flat_foot), g["free_chair"], walk["body"])
    flat_ok = ndi.sum(cov_flat, rooms, index=np.arange(1, nrooms + 1)) * a >= REACH_MIN_M2

    # for a room cut off by width only: the widest bottleneck on the way in, found by
    # lowering the required width until the room joins the entrance component
    sweep = np.round(np.arange(prof["passage"] - 0.05, walk["passage"] - 1e-9, -0.05), 2)
    joins_at, pinch_at, prev = {}, {}, R["Rc"]
    for t in sweep:
        p = dict(prof, passage=t, door=min(prof["door"], t))
        Rt = reach(g, centre_space(g, W, doorzone, p))
        cov = covered_from(Rt, g["free_chair"], prof["body"])
        got = ndi.sum(cov, rooms, index=np.arange(1, nrooms + 1)) * a >= REACH_MIN_M2
        # the pinch: centre cells newly reached at this width that are narrower than the last one
        pinch_at[float(t)] = np.argwhere(Rt & ~prev & (W < t + 0.05))
        prev = Rt
        for r in np.flatnonzero(got):
            joins_at.setdefault(int(r) + 1, float(t))

    # the place to pin a cut-off room: its cell nearest to the floor the wheelchair reaches
    _, (ii, jj) = ndi.distance_transform_edt(~R["cov_c"], return_indices=True)

    def xy(i, j):
        return [round(float((i + 0.5) * CELL + g["lo"][0]), 2), round(float((j + 0.5) * CELL + g["lo"][1]), 2)]

    issues = []
    for r in range(1, nrooms + 1):
        m = rooms == r
        area = round(m.sum() * a, 1)
        cells = np.argwhere(m & R["cov_f"])
        if status[r - 1] == 1:
            # the room cell closest to where the chair's reach stops
            dd = np.hypot(cells[:, 0] - ii[cells[:, 0], cells[:, 1]], cells[:, 1] - jj[cells[:, 0], cells[:, 1]])
            ci, cj = cells[np.argmin(dd)]
            if not flat_ok[r - 1]:
                issues.append(dict(kind="Stairs or step only", room=r, room_m2=area, pos=xy(ci, cj),
                                   measured="no level way in", required="step-free way", severity="high"))
            else:
                t = joins_at.get(r)
                meas = f"{t:.2f}-{t + 0.05:.2f} m" if t else f"< {walk['passage']:.2f} m"
                pc = pinch_at.get(t) if t else None
                if pc is not None and len(pc):
                    # pin the issue on the pinch nearest to the room, not on the room itself
                    ci, cj = pc[np.argmin(np.hypot(pc[:, 0] - ci, pc[:, 1] - cj))]
                issues.append(dict(kind="Passage too narrow", room=r, room_m2=area, pos=xy(ci, cj),
                                   measured=meas, required=f"{prof['passage']:.2f} m", severity="high"))
    for dd in doors:
        on_route = (dd["cells"] & R["cov_f"]).any()
        if on_route and not dd["measurable"]:
            why = "scan edge" if dd["scan_edge"] else "closed leaf"
            issues.append(dict(kind="Door not measured", room=0, room_m2=None, pos=[dd["x"], dd["y"]],
                               measured=why, required="check on site", severity="low", door=dd["id"]))
        elif on_route and dd["width_m"] < prof["door"]:
            # the room the door serves: of the rooms it opens onto, the smaller one (an
            # office off a corridor is filed under the office, not the corridor)
            near = rooms[ndi.binary_dilation(dd["cells"], iterations=4)]
            ids = [int(k) for k in np.unique(near) if k > 0 and (near == k).sum() >= 4]
            r = min(ids, key=lambda k: (rooms == k).sum()) if ids else 0
            issues.append(dict(kind="Door too narrow", room=int(r), room_m2=None, pos=[dd["x"], dd["y"]],
                               measured=f"{dd['width_m']:.3f} m", required=f"{prof['door']:.2f} m",
                               severity="medium", door=dd["id"]))
    for r in np.flatnonzero(status == 2) + 1:
        m = rooms == r
        if not (centres & m).any():
            cells = np.argwhere(m & R["cov_c"])
            ci, cj = cells[len(cells) // 2]
            # the biggest circle the opening of step 6 would accept in this room: the largest
            # odd element whose erosion keeps a centre here (EDT > n // 2, as cross-checked)
            e = float(ndi.distance_transform_edt(g["free_chair"])[m].max())
            dmax = (2 * (int(np.ceil(e - 1e-9)) - 1) + 2) * CELL
            issues.append(dict(kind="No turning space", room=int(r), room_m2=round(m.sum() * a, 1),
                               pos=xy(ci, cj), measured=f"{dmax:.2f} m circle",
                               required=f"{prof['turn']:.2f} m circle", severity="medium"))
    sev = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda s: (sev[s["severity"]], -round(s["pos"][1]), s["pos"][0]))
    for n, s in enumerate(issues, 1):
        s["n"] = n
    tick("8_issues", t0)
    kinds = {}
    for s in issues:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    S["issues"] = dict(count=len(issues), high=sum(s["severity"] == "high" for s in issues),
                       medium=sum(s["severity"] == "medium" for s in issues),
                       low=sum(s["severity"] == "low" for s in issues), by_kind=kinds,
                       bottlenecks={str(k): v for k, v in sorted(joins_at.items())
                                    if status[k - 1] == 1})
    print(f"[8] issues {S['issues']}")
    for s in issues:
        print(f"    #{s['n']:>2} {s['severity']:<6} {s['kind']:<20} room {s['room']:>2} at {s['pos']}"
              f"  measured {s['measured']}  required {s['required']}")
    return issues


# %% Step 9. Act on it: widen one door, ramp one flight, and judge again
def step9_act(g, W, doorzone, R, doors, prof):
    t0 = time.perf_counter()
    a = CELL * CELL
    base = int((R["status"] == 2).sum())

    def rooms_reached(C):
        cov = covered_from(reach(g, C), g["free_chair"], prof["body"])
        got = ndi.sum(cov, R["rooms"], index=np.arange(1, R["nrooms"] + 1)) * a
        return int((got >= REACH_MIN_M2).sum())

    C0 = centre_space(g, W, doorzone, prof)
    widen = []
    for dd in doors:
        if dd["width_m"] >= prof["door"] or not dd["measurable"] or not (dd["cells"] & R["cov_f"]).any():
            continue
        # widen the door to the required width: its zone becomes passable centre space
        zone = ndi.binary_dilation(dd["cells"], structure=disk(odd_cells(0.8))) & g["free_foot"] & ~g["stairs"]
        widen.append(dict(door=dd["id"], width_m=dd["width_m"], rooms=rooms_reached(C0 | zone) - base))

    lab, n = ndi.label(g["stairs"], structure=np.ones((3, 3)))
    ramps = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * a < 0.5:
            continue
        # a ramp in place of this flight: the flight and the jumps around it become rollable
        zone = ndi.binary_dilation(m, iterations=int(0.5 / CELL)) & g["free_foot"]
        cells = np.argwhere(m)
        cx, cy = (cells.mean(0) + 0.5) * CELL + g["lo"]
        ramps.append(dict(flight=k, x=round(float(cx), 2), y=round(float(cy), 2),
                          area_m2=round(m.sum() * a, 1), rooms=rooms_reached(C0 | zone) - base))
    tick("9_act", t0)
    S["act"] = dict(base_rooms=base, widen=widen, ramps=ramps)
    print(f"[9] act: base {base} rooms; widen {widen}; ramps {ramps}")
    return widen, ramps


def judge_profile(g, name):
    """Steps 3-7 again for another profile: the survey (step 2) does not change."""
    t0 = time.perf_counter()
    prof = PROFILES[name]
    W = clear_width(g["free_foot"] if prof["stairs"] else g["free_chair"])
    doorzone = ndi.binary_dilation(g["door"], structure=disk(odd_cells(0.8)))
    base = g["free_foot"] if prof["stairs"] else g["free_chair"]
    C = base & ((W >= prof["passage"]) | (doorzone & (W >= prof["door"])))
    cov = covered_from(reach(g, C), base, prof["body"])
    rooms, nrooms = rooms_of(g)
    got = ndi.sum(cov, rooms, index=np.arange(1, nrooms + 1)) * CELL * CELL
    return dict(profile=name, rooms_reached=int((got >= REACH_MIN_M2).sum()), rooms=int(nrooms),
                covered_m2=round(cov.sum() * CELL * CELL, 1),
                ms=round((time.perf_counter() - t0) * 1000, 1))


def fingerprint(xyz, cls):
    """SHA-256 of x, y, z (float32) and class, in file order: which scan was audited."""
    h = hashlib.sha256()
    for a in (xyz[:, 0], xyz[:, 1], xyz[:, 2]):
        h.update(np.ascontiguousarray(a.astype("<f4")).tobytes())
    h.update(np.ascontiguousarray(cls.astype("<f4")).tobytes())
    return h.hexdigest()


def headline(S):
    """The numbers the article quotes, flat, in out/metrics.json."""
    best = max(S["act"]["ramps"], key=lambda r: r["rooms"]) if S["act"]["ramps"] else None
    return dict(
        input_sha256=S["input_sha256"], points=S["points"], grid_cells=S["grid"]["cells"],
        floor_measured_m2=S["grid"]["floor_measured_m2"], floor_filled_m2=S["grid"]["floor_filled_m2"],
        obstacle_on_floor_m2=S["grid"]["obstacle_on_floor_m2"], jump_m2=S["grid"]["jump_m2"],
        free_chair_m2=S["grid"]["free_chair_m2"],
        width_median_m=S["width"]["median_m"], width_max_m=S["width"]["max_m"],
        width_ge_passage_m2=S["width"]["ge_passage_m2"], width_lt_passage_m2=S["width"]["lt_passage_m2"],
        centre_m2=S["centre"]["centre_m2"], covered_m2=S["centre"]["covered_m2"],
        doors=S["doors"]["count"], doors_measured=S["doors"]["measured"],
        doors_unmeasured=S["doors"]["unmeasured"], doors_under_required=S["doors"]["under_required"],
        door_narrowest_m=S["doors"]["narrowest_m"], door_median_m=S["doors"]["median_m"],
        door_widest_m=S["doors"]["widest_m"],
        turn_se_cells=S["turning"]["se_cells"], turn_m2=S["turning"]["turn_m2"],
        turn_half_diameter_m2=S["turning"]["half_diameter_wrong_m2"],
        turn_edt_agreement=S["turning"]["edt_agreement"],
        reached_rooms_with_circle=S["turning"]["reached_rooms_with_circle"],
        rooms=S["reach"]["rooms"], rooms_wheelchair=S["reach"]["wheelchair"],
        rooms_on_foot=S["reach"]["on_foot_total"], rooms_foot_only=S["reach"]["foot_only"],
        rooms_not_reached=S["reach"]["not_reached"], chair_covered_m2=S["reach"]["chair_covered_m2"],
        foot_covered_m2=S["reach"]["foot_covered_m2"],
        issues=S["issues"]["count"], issues_high=S["issues"]["high"],
        issues_medium=S["issues"]["medium"], issues_low=S["issues"]["low"],
        issues_by_kind=S["issues"]["by_kind"],
        best_ramp=best, rooms_after_best_ramp=(S["act"]["base_rooms"] + best["rooms"]) if best else None,
        profiles={p["profile"]: p["rooms_reached"] for p in S["profiles"]},
        total_s=S["total_s"], ms=S["ms"])


def main(data=None, out=None):
    global DATA, OUT
    DATA = Path(data) if data else DATA
    OUT = Path(out) if out else OUT
    S.clear()
    T.clear()
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    prof = PROFILES[PROFILE]
    S["profile"] = dict(name=PROFILE, **prof)
    xyz, rgb, cls = step1_load()
    S["input_sha256"] = fingerprint(xyz, cls)
    g = step2_grid(xyz, cls)
    W = step3_width(g, prof)
    C, doorzone = step4_centre(g, W, prof)
    doors = step5_doors(g, prof, xyz, cls)
    centres, turn, wrong = step6_turning(g, prof)
    R = step7_reach(g, C, W, prof)
    # turning per reached room
    reached = np.flatnonzero(R["status"] == 2) + 1
    with_turn = [int(r) for r in reached if (centres & (R["rooms"] == r)).any()]
    S["turning"]["reached_rooms_with_circle"] = f"{len(with_turn)} of {len(reached)}"
    issues = step8_issues(g, W, doorzone, R, doors, centres, prof)
    widen, ramps = step9_act(g, W, doorzone, R, doors, prof)
    S["profiles"] = [judge_profile(g, p) for p in PROFILES]
    print(f"[+] profiles {S['profiles']}")
    S["ms"] = T
    S["total_s"] = round(time.perf_counter() - t_all, 2)

    door_map = np.zeros_like(R["rooms"])
    door_w = {}
    for dd in doors:
        door_map[dd["cells"]] = dd["id"]
        door_w[dd["id"]] = dd["width_m"]
    np.savez_compressed(OUT / "audit_grids.npz", lo=g["lo"], cell=CELL, H=g["H"], Hnear=g["Hnear"],
                        has_floor=g["has_floor"], struct=g["struct"], furn=g["furn"], door=g["door"],
                        stairs=g["stairs"], obstacle=g["obstacle"], jump=g["jump"],
                        free_foot=g["free_foot"], free_chair=g["free_chair"], W=W, C=C,
                        doorzone=doorzone, centres=centres, turn=turn, wrong=wrong,
                        Rc=R["Rc"], cov_c=R["cov_c"], Rf=R["Rf"], cov_f=R["cov_f"],
                        rooms=R["rooms"], status=R["status"], door_map=door_map,
                        door_ids=np.array(list(door_w.keys())), door_widths=np.array(list(door_w.values())),
                        door_measurable=np.array([dd["measurable"] for dd in doors]),
                        entrances=np.array(ENTRANCES))
    with open(OUT / "issues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "severity", "issue", "room", "x", "y", "measured", "required"])
        for s in issues:
            w.writerow([s["n"], s["severity"], s["kind"], s["room"], s["pos"][0], s["pos"][1],
                        s["measured"], s["required"]])
    S["issue_list"] = [{k: v for k, v in s.items() if k != "cells"} for s in issues]
    S["door_list"] = [{k: v for k, v in dd.items() if k != "cells"} for dd in doors]
    (OUT / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(headline(S), indent=1), encoding="utf-8")
    print(f"[=] done in {S['total_s']} s; step ms {T}")
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Accessibility audit from a classified scan.")
    ap.add_argument("--data", help="the classified PLY (default: data/s3dis_area2_access.ply)")
    ap.add_argument("--out", help="the output folder (default: out/)")
    a = ap.parse_args()
    main(a.data, a.out)
