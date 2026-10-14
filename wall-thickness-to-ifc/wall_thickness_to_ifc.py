# -*- coding: utf-8 -*-
#
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
# Two faces, one wall: measure the thickness of a wall you never touched,
# from a classified indoor scan, then write it as an IFC4 wall and read it back.
#
#   Step 1  load the classified scan and name its classes (from the .n3d.json sidecar)
#   Step 2  level the scan between its floor and its ceiling
#   Step 3  find the two wall directions from the wall normals
#   Step 4  project wall points onto each wall normal (the 1D profile)
#   Step 5  find the faces as density peaks, cut them into runs along the wall
#   Step 6  pair the two faces of a wall (gap, overlap, empty between, seen from both sides)
#   Step 7  measure the thickness cell by cell on the wall (matched heights, cell bootstrap, lean)
#   Step 8  write an IFC4 wall with that thickness (ifcopenshell)
#   Step 9  open the IFC back and prove it: schema, geometry, fit to the points
#
# Everything lands in code/out/ (metrics.json holds every number the article quotes).
# Deterministic: two runs, or the same scan with its points shuffled, give the same numbers.
#
# Run:  python wall_thickness_to_ifc.py
# Tests (synthetic walls with known answers):  python -m pytest tests
#
import json
import os
import sys
import time

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.stats import theilslopes

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "s3dis_office.ply")
SIDECAR = DATA + ".n3d.json"
OUT = os.path.join(HERE, "out")

# ---------------------------------------------------------------- parameters
SEED = 42
NORMAL_K = 16            # neighbours for a PCA normal (points are about 3 to 4 cm apart)
VERTICAL_MAX_NZ = 0.25   # a wall normal is close to horizontal: |nz| under this
PLANARITY_MIN = 0.45     # (l2 - l3) / l1: flat enough to trust the normal
MIN_GOOD_NORMALS = 50    # fewer trusted normals than this and there is no wall direction to find
FAMILY_TOL_DEG = 15.0    # a normal this close to a direction joins its family
BIN = 0.01               # 1 cm bins for the offset profile
SMOOTH = np.array([1, 2, 3, 2, 1], float) / 9.0
PEAK_MIN_PTS = 60        # a face needs at least this many points in its peak bin (smoothed)
PEAK_MIN_SEP = 0.04      # two faces closer than 4 cm are one face
FACE_HALF = 0.03         # a point within 3 cm of the peak belongs to the face (to trace its runs)
RUN_GAP = 0.60           # a hole shorter than this does not cut the face (a door is wider)
RUN_MIN = 0.50           # a run shorter than 50 cm is a jamb or noise
T_MIN, T_MAX = 0.04, 0.45   # what a wall can be
OVERLAP_MIN = 0.50       # two faces must face each other over at least 50 cm
EMPTY_MAX = 0.15         # the profile's valley between two faces, as a share of the weaker peak
FACE_REACH = 0.10        # a face keeps its points up to 10 cm beyond its peak (a leaning face spreads)
CELL = 0.25              # the wall is cut into 25 x 25 cm cells (along x height) to compare the faces
CELL_MIN_PTS = 5         # a face is "seen" in a cell with at least this many points
COVIS_MIN = 0.30         # share of cells where BOTH faces are seen: a lining hides the wall behind it
CELL_TOL = 0.02          # a cell whose median sits more than 2 cm off its face plane is something else
ROOM_NEAR, ROOM_FAR = 0.10, 0.60   # the strip beyond a face where room points are counted (reported)
ROOM_MIN_DENSITY = 40.0  # room points per m2 in that strip: "a room shows" (reported, not a gate)
N_BOOT = 1000            # bootstrap resamples of the cells for the thickness interval
SLICE = 0.50             # the thickness is also reported per 50 cm of height

LOG = {}


def say(msg):
    print(msg, flush=True)


# ============================================================ Step 1: load
def read_ply(path):
    """Minimal binary little-endian PLY reader: returns a dict of 1D arrays."""
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1",
             "uint8": "u1", "char": "i1", "int8": "i1", "int": "<i4", "int32": "<i4",
             "uint": "<u4", "uint32": "<u4", "ushort": "<u2", "uint16": "<u2", "short": "<i2",
             "int16": "<i2"}
    fields, n = [], 0
    for line in header:
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            n = int(tok[2])
        elif tok and tok[0] == "property":
            fields.append((tok[2], types[tok[1]]))
    arr = np.frombuffer(raw[end:], dtype=np.dtype(fields), count=n)
    return {name: np.asarray(arr[name]) for name, _ in fields}


def load_scan(path, sidecar):
    ply = read_ply(path)
    xyz = np.c_[ply["x"], ply["y"], ply["z"]].astype(np.float64)
    rgb = np.c_[ply["red"], ply["green"], ply["blue"]].astype(np.float64) / 255.0
    cls = np.rint(ply["classification"]).astype(int)
    with open(sidecar, encoding="utf-8") as f:
        names = {int(k): v for k, v in json.load(f)["class_labels"].items()}
    # name -> id, so the code never hard-codes S3DIS numbers
    ids = {v: k for k, v in names.items()}
    return xyz, rgb, cls, names, ids


# ============================================================ Step 2: level
def rotation_between(a, b):
    """Rotation matrix turning unit vector a onto unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def fit_plane(pts, iters=5, keep=0.02):
    """Least-squares plane (normal pointing up), refitted on the points within `keep` of it."""
    sel = np.ones(len(pts), bool)
    for _ in range(iters):
        c = pts[sel].mean(0)
        _, _, vt = np.linalg.svd(pts[sel] - c, full_matrices=False)
        n = vt[2] * (1.0 if vt[2][2] >= 0 else -1.0)
        r = (pts - c) @ n
        new = np.abs(r) < keep
        if new.sum() < 3:
            break
        sel = new
    return n, c, r[sel]


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), -1.0, 1.0))))


def level_scan(xyz, cls, ids):
    """Rotate the scan so its vertical is the bisector of the floor and ceiling normals.

    A floor and a ceiling are two independent horizontal references. When they disagree
    (a bent or badly registered scan), levelling on one tilts the other and every wall
    with it; their bisector splits the difference. With no ceiling, the floor alone."""
    have = lambda name: name in ids and (cls == ids[name]).sum() >= 3
    if not have("floor"):
        raise ValueError("no floor class: the scan cannot be levelled")
    n_f, c_f, res = fit_plane(xyz[cls == ids["floor"]])
    info = {"floor_tilt_deg": angle_deg(n_f, [0, 0, 1.0]), "floor_sd": float(np.std(res))}
    up = n_f
    if have("ceiling"):
        n_c, _, _ = fit_plane(xyz[cls == ids["ceiling"]])
        up = n_f + n_c
        up /= np.linalg.norm(up)
        info.update(ceiling_tilt_deg=angle_deg(n_c, [0, 0, 1.0]),
                    floor_ceiling_deg=angle_deg(n_f, n_c))
    R = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    lev = (xyz - c_f) @ R.T
    floor_z = float(np.median(lev[cls == ids["floor"], 2]))
    lev[:, 2] -= floor_z
    info["tilt_deg"] = angle_deg(up, [0, 0, 1.0])
    if have("ceiling"):
        # storey height: median of ceiling minus floor, and how much it varies (1 m cells)
        fl, ce = lev[cls == ids["floor"]], lev[cls == ids["ceiling"]]
        info["storey_h"] = float(np.median(ce[:, 2]) - np.median(fl[:, 2]))
        key = lambda p: np.floor(p[:, :2] / 1.0).astype(int)
        hs = []
        kf, kc = key(fl), key(ce)
        cells = set(map(tuple, kf)) & set(map(tuple, kc))
        for c in sorted(cells):
            mf = (kf[:, 0] == c[0]) & (kf[:, 1] == c[1])
            mc = (kc[:, 0] == c[0]) & (kc[:, 1] == c[1])
            if mf.sum() >= 20 and mc.sum() >= 20:
                hs.append(np.median(ce[mc, 2]) - np.median(fl[mf, 2]))
        if hs:
            info["storey_h_p5"], info["storey_h_p95"] = map(float, np.percentile(hs, [5, 95]))
    return lev, R, c_f, info


# ============================================================ Step 3: directions
def pca_normals(pts, k=None):
    """Per-point normal (smallest eigenvector) and planarity from k neighbours."""
    k = min(k or NORMAL_K, len(pts))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    nb = pts[idx]                                   # (N, k, 3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)                      # ascending eigenvalues
    normals = v[:, :, 0]
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    planarity = (l2 - l3) / np.maximum(l1, 1e-12)
    return normals, planarity


def wall_directions(wall):
    normals, planarity = pca_normals(wall)
    good = (np.abs(normals[:, 2]) < VERTICAL_MAX_NZ) & (planarity > PLANARITY_MIN)
    if good.sum() < MIN_GOOD_NORMALS:
        raise ValueError(f"only {good.sum()} trusted wall normals: no wall direction to find")
    ang = np.arctan2(normals[good, 1], normals[good, 0])
    # fold four ways: a normal, its opposite and its perpendicular vote for one axis
    phi = 0.25 * np.arctan2(np.sin(4 * ang).sum(), np.cos(4 * ang).sum())
    dirs = [np.array([np.cos(phi), np.sin(phi), 0.0]),
            np.array([-np.sin(phi), np.cos(phi), 0.0])]
    dev = np.degrees(np.abs(((ang - phi) + np.pi / 4) % (np.pi / 2) - np.pi / 4))
    return normals, planarity, good, phi, dirs, dev


# ============================================================ Step 4: project
def family_offsets(wall, normals, good, direction):
    """Points whose normal follows `direction`, their offset along it, and position along the wall."""
    cosang = np.abs(normals @ direction)
    member = good & (cosang > np.cos(np.radians(FAMILY_TOL_DEG)))
    along_dir = np.array([-direction[1], direction[0], 0.0])
    off = wall[member] @ direction
    s = wall[member] @ along_dir
    return member, off, s, along_dir


def profile(off):
    """1 cm histogram of the offsets on a grid anchored at 0 (so it never depends on point order)."""
    lo = np.floor(off.min() / BIN) * BIN - 3 * BIN
    hi = np.ceil(off.max() / BIN) * BIN + 3 * BIN
    n = int(round((hi - lo) / BIN))
    edges = lo + BIN * np.arange(n + 1)
    hist, _ = np.histogram(off, bins=edges)
    smooth = np.convolve(hist, SMOOTH, mode="same")
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, hist, smooth


# ============================================================ Step 5: faces
def runs_along(s, gap=RUN_GAP, min_len=RUN_MIN):
    """Stretches of the wall where a face has points, bridging holes shorter than `gap`.

    The hole is measured between real neighbouring points, not between 10 cm cells: on cells,
    one point crossing a cell border changes a hole by 10 cm, so a hole near the threshold
    splits or not with the rounding of the coordinates."""
    if len(s) == 0:
        return []
    s = np.sort(s)
    cut = np.flatnonzero(np.diff(s) > gap)
    starts, ends = np.r_[s[0], s[cut + 1]], np.r_[s[cut], s[-1]]
    return [(float(a), float(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def find_faces(centres, smooth, off, s, fam):
    peaks, _ = find_peaks(smooth, height=PEAK_MIN_PTS,
                          distance=max(1, int(round(PEAK_MIN_SEP / BIN))),
                          prominence=PEAK_MIN_PTS * 0.5)
    faces = []
    for p in peaks:
        o = centres[p]
        near = np.abs(off - o) < FACE_HALF
        for a, b in runs_along(s[near]):
            inside = near & (s >= a) & (s <= b)
            faces.append({"family": fam, "peak": float(o), "s0": float(a), "s1": float(b),
                          "n": int(inside.sum()), "idx": np.flatnonzero(inside)})
    return peaks, faces


# ============================================================ Step 6: pair
def valley_of(cen, sm, pa, pb):
    """Lowest point of the profile between two peaks, and the centre of that low stretch."""
    inner = np.flatnonzero((cen > pa + FACE_HALF) & (cen < pb - FACE_HALF))
    if len(inner) == 0:
        return 0.0, 0.5 * (pa + pb)
    low = sm[inner].min()
    at = inner[sm[inner] <= low + 1e-9]
    # the longest run of lowest bins, and its centre (not the first zero next to face A)
    runs = np.split(at, np.flatnonzero(np.diff(at) > 1) + 1)
    best = max(runs, key=len)
    return float(low), float(0.5 * (cen[best[0]] + cen[best[-1]]))


def face_sides(off, pa, pb, mid):
    """Points of face A (low offset) and face B: each side of the valley, up to FACE_REACH out."""
    in_a = (off >= pa - FACE_REACH) & (off < mid)
    in_b = (off > mid) & (off <= pb + FACE_REACH)
    return in_a, in_b


def cell_keys(s, z):
    return np.floor(s / CELL).astype(int), np.floor(z / CELL).astype(int)


def seen_cells(s, z):
    """The (along, height) cells where a face has at least CELL_MIN_PTS points."""
    if len(s) == 0:
        return set()
    ks, kz = cell_keys(s, z)
    u, cnt = np.unique(np.c_[ks, kz], axis=0, return_counts=True)
    return {tuple(x) for x in u[cnt >= CELL_MIN_PTS]}


def room_density(room_xy, direction, along, face_off, outward, s0, s1):
    """Room points (anything under 1.2 m that is not wall or ceiling) per m2 in a strip beyond a face."""
    o = room_xy @ direction[:2]
    t = room_xy @ along[:2]
    near, far = face_off + outward * ROOM_NEAR, face_off + outward * ROOM_FAR
    lo, hi = min(near, far), max(near, far)
    n = ((o > lo) & (o < hi) & (t >= s0) & (t <= s1)).sum()
    return float(n / ((hi - lo) * (s1 - s0)))


def scan_reach(all_xy, direction, along, face_off, outward, s0, s1):
    """How far the scan itself extends beyond a face (over the same stretch): no data is not no room."""
    o = all_xy @ direction[:2]
    t = all_xy @ along[:2]
    on = (t >= s0) & (t <= s1)
    beyond = outward * (o[on] - face_off)
    return float(max(0.0, beyond.max())) if beyond.size else 0.0


def objects_beyond(obj_xy, direction, along, face_off, outward, s0, s1, reach=0.10):
    """Non-wall points just beyond a face (1 to 10 cm): furniture or clutter of the room behind it."""
    o = outward * (obj_xy @ direction[:2] - face_off)
    t = obj_xy @ along[:2]
    return int(((o > 0.01) & (o < reach) & (t >= s0) & (t <= s1)).sum())


def pair_faces(faces, off, s, direction, along, room_xy, all_xy, obj_xy, every):
    """Two parallel faces T_MIN..T_MAX apart, empty between them, and seen from both sides.

    off, s: the family's trusted points (they found the faces). every = (off, s, z) of ALL wall
    points: the tests that look at the wall itself use them, whatever their normal."""
    off_w, s_w, z_w = every
    cands = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if faces[i]["family"] != faces[j]["family"]:
                continue
            # a is always the face with the smaller offset
            i2, j2 = (i, j) if faces[i]["peak"] < faces[j]["peak"] else (j, i)
            a, b = faces[i2], faces[j2]
            gap = b["peak"] - a["peak"]
            if not (T_MIN <= gap <= T_MAX):
                continue
            s0, s1 = max(a["s0"], b["s0"]), min(a["s1"], b["s1"])
            if s1 - s0 < OVERLAP_MIN:
                continue
            # test 3, empty between: the valley of the profile, over the shared stretch only
            on = (s >= s0) & (s <= s1)
            cen, _, sm = profile(off[on])
            pk_a = sm[np.argmin(np.abs(cen - a["peak"]))]
            pk_b = sm[np.argmin(np.abs(cen - b["peak"]))]
            valley, mid = valley_of(cen, sm, a["peak"], b["peak"])
            ratio = valley / max(min(pk_a, pk_b), 1e-9)
            # test 4, seen from both sides: where the wall was scanned, are BOTH faces there?
            on_w = (s_w >= s0) & (s_w <= s1)
            in_a, in_b = face_sides(off_w, a["peak"], b["peak"], mid)
            ca = seen_cells(s_w[on_w & in_a], z_w[on_w & in_a])
            cb = seen_cells(s_w[on_w & in_b], z_w[on_w & in_b])
            both = len(ca & cb)
            covis = both / max(1, min(len(ca), len(cb)))
            # reported, not a gate: room points beyond each face, and how far the scan reaches
            dens_a = room_density(room_xy, direction, along, a["peak"], -1.0, s0, s1)
            dens_b = room_density(room_xy, direction, along, b["peak"], +1.0, s0, s1)
            reach_a = scan_reach(all_xy, direction, along, a["peak"], -1.0, s0, s1)
            reach_b = scan_reach(all_xy, direction, along, b["peak"], +1.0, s0, s1)
            obj_a = objects_beyond(obj_xy, direction, along, a["peak"], -1.0, s0, s1)
            obj_b = objects_beyond(obj_xy, direction, along, b["peak"], +1.0, s0, s1)
            verdict = "WALL"
            if ratio > EMPTY_MAX:
                verdict = "not empty between"
            elif covis < COVIS_MIN:
                verdict = "not seen from both sides (a lining or a panel in front of a wall)"
            cands.append({"i": i2, "j": j2, "gap": float(gap), "s0": s0, "s1": s1, "mid": mid,
                          "valley_ratio": float(ratio), "cells_a": len(ca), "cells_b": len(cb),
                          "cells_both": both, "covis": float(covis),
                          "room_a": dens_a, "room_b": dens_b, "reach_a": reach_a,
                          "reach_b": reach_b, "objects_a": obj_a, "objects_b": obj_b,
                          "verdict": verdict})
    # longest shared stretch first; ties broken by the faces' indices (order-free)
    cands.sort(key=lambda c: (-(c["s1"] - c["s0"]), c["i"], c["j"]))
    used, pairs = set(), []
    for c in cands:
        if c["verdict"] != "WALL" or c["i"] in used or c["j"] in used:
            continue
        used.update((c["i"], c["j"]))
        pairs.append(c)
    return cands, pairs


# ============================================================ Step 7: thickness
def face_line(zc, oc):
    """Robust line offset = c + slope * z through per-cell medians (Theil-Sen)."""
    if len(zc) >= 3 and np.ptp(zc) > 0:
        slope, icpt, _, _ = theilslopes(oc, zc)
        return float(slope), float(icpt)
    return 0.0, float(np.median(oc)) if len(oc) else 0.0


def cell_medians(s, z, off):
    """Median offset per (along, height) cell with at least CELL_MIN_PTS points: {(ks, kz): (median, n)}."""
    if len(s) == 0:
        return {}
    ks, kz = cell_keys(s, z)
    order = np.lexsort((off, kz, ks))              # by cell, then offset: independent of point order
    ks, kz, o = ks[order], kz[order], off[order]
    keys, start, cnt = np.unique(np.c_[ks, kz], axis=0, return_index=True, return_counts=True)
    return {(int(k[0]), int(k[1])): (float(np.median(o[a:a + c])), int(c))
            for k, a, c in zip(keys, start, cnt) if c >= CELL_MIN_PTS}


def ols_line(zc, oc):
    """offset = icpt + slope * z by least squares on cell medians (flat if the cells span no height)."""
    if len(zc) >= 2 and np.ptp(zc) >= CELL:
        slope, icpt = np.polyfit(zc, oc, 1)
        return float(slope), float(icpt)
    return 0.0, float(np.median(oc))


def measure_thickness(pair, off, s, z, z_ref, seed=SEED):
    """Thickness from the two faces compared cell by cell on the wall (25 x 25 cm, along x height).

    1. each face: its median offset in every cell where it has CELL_MIN_PTS points
    2. a robust plane per face (Theil-Sen through its cells); cells off it by more than CELL_TOL
       are something else (a board, a shelf, a reveal) and leave
    3. matched cells: both faces clean there, and the gap between them empty in that cell too
    4. a straight line per face through the matched cells: its slope is the face's lean, and the
       thickness is the distance between the two lines at z_ref (mid storey height), which is the
       thickness that gives the wall its true volume even if a face leans
    5. the 95% interval resamples the matched cells (neighbouring points share their errors,
       cells much less so) and refits both lines each time
    """
    on = (s >= pair["s0"]) & (s <= pair["s1"])
    in_a, in_b = face_sides(off, pair["pa"], pair["pb"], pair["mid"])
    ia, ib = np.flatnonzero(on & in_a), np.flatnonzero(on & in_b)
    per = {"a": cell_medians(s[ia], z[ia], off[ia]), "b": cell_medians(s[ib], z[ib], off[ib])}
    zc_of = lambda keys: np.array([(k[1] + 0.5) * CELL for k in keys])
    clean, robust = {}, {}
    for side in ("a", "b"):
        keys = sorted(per[side])
        if len(keys) < 3:
            return None
        zc, oc = zc_of(keys), np.array([per[side][k][0] for k in keys])
        slope, icpt = face_line(zc, oc)
        robust[side] = (slope, icpt)
        clean[side] = {k for k, zz, o in zip(keys, zc, oc) if abs(o - (icpt + slope * zz)) <= CELL_TOL}
    # the gap must be empty in the cell too: points strictly between the two robust planes
    (sa, ca_), (sb, cb_) = robust["a"], robust["b"]
    between = on & (off > ca_ + sa * z + FACE_HALF) & (off < cb_ + sb * z - FACE_HALF)
    gk = cell_keys(s[between], z[between])
    gap_count = {}
    for k in zip(gk[0].tolist(), gk[1].tolist()):
        gap_count[k] = gap_count.get(k, 0) + 1
    common = sorted(k for k in clean["a"] & clean["b"]
                    if gap_count.get(k, 0) <= EMPTY_MAX * min(per["a"][k][1], per["b"][k][1]))
    if len(common) < 3:
        return None
    zc = zc_of(common)
    oa = np.array([per["a"][k][0] for k in common])
    ob = np.array([per["b"][k][0] for k in common])
    t_cells = ob - oa

    def t_at(idx):
        la, lb = ols_line(zc[idx], oa[idx]), ols_line(zc[idx], ob[idx])
        return (lb[1] + lb[0] * z_ref) - (la[1] + la[0] * z_ref), la, lb

    all_idx = np.arange(len(common))
    t, la, lb = t_at(all_idx)
    rng = np.random.default_rng(seed)
    boot = np.array([t_at(rng.integers(0, len(common), len(common)))[0] for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lean = [float(np.degrees(np.arctan(la[0]))), float(np.degrees(np.arctan(lb[0])))]
    # face noise: spread of the points of the matched cells around their own line
    noise, n_pts, used = [], [], []
    cset = set(common)
    for idx, (slope, icpt) in ((ia, la), (ib, lb)):
        ks, kz = cell_keys(s[idx], z[idx])
        keep = np.array([(a_, b_) in cset for a_, b_ in zip(ks.tolist(), kz.tolist())], bool)
        idx = idx[keep] if keep.size else idx
        r = off[idx] - (icpt + slope * z[idx])
        noise.append(float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else 0.0)
        n_pts.append(int(len(idx)))
        used.append(idx)
    # thickness per 50 cm of height, straight from the matched cells in that band
    per_h = []
    for h0 in np.arange(0.0, zc.max() + SLICE, SLICE):
        m = (zc >= h0) & (zc < h0 + SLICE)
        if m.sum() >= 2:
            per_h.append((float(h0), float(h0 + SLICE), float(np.median(t_cells[m])),
                          int(m.sum()), 0))
    face_a = la[1] + la[0] * z_ref
    return {"t": float(t), "ci": (float(lo), float(hi)), "boot": boot, "n_cells": len(common),
            "cells_a": len(per["a"]), "cells_b": len(per["b"]),
            "clean_a": len(clean["a"]), "clean_b": len(clean["b"]),
            "t_cells": t_cells, "z_cells": zc, "cells": [list(k) for k in common],
            "t_median_cells": float(np.median(t_cells)),
            "face_a": float(face_a), "face_b": float(face_a + t),
            "noise_a": noise[0], "noise_b": noise[1],
            "n_a": n_pts[0], "n_b": n_pts[1], "ia": used[0], "ib": used[1],
            "lean_a_deg": lean[0], "lean_b_deg": lean[1], "lines": [la, lb],
            "t_min_cell": float(np.percentile(t_cells, 5)),
            "t_max_cell": float(np.percentile(t_cells, 95)),
            # the two face lines, read at the floor and at twice z_ref (the ceiling)
            "t_floor": float(lb[1] - la[1]),
            "t_ceiling": float((lb[1] - la[1]) + (lb[0] - la[0]) * 2 * z_ref),
            "per_height": per_h}


# ============================================================ the measurement, no I/O
def measure(xyz, cls, names, ids, log=say):
    """Steps 2 to 7 on arrays. Returns everything the IFC writer, the figures and the tests need."""
    res = {"walls": [], "families": [], "reason": ""}
    lev, R, c0, info = level_scan(xyz, cls, ids)
    res.update(lev=lev, R=R, level=info)
    log(f"Step 2  level: floor tilt {info['floor_tilt_deg']:.2f} deg"
        + (f", ceiling tilt {info['ceiling_tilt_deg']:.2f} deg, floor vs ceiling "
           f"{info['floor_ceiling_deg']:.2f} deg" if "ceiling_tilt_deg" in info else "")
        + f"; vertical tilt removed {info['tilt_deg']:.2f} deg")
    if "storey_h" in info:
        log(f"  floor flat within {100*info['floor_sd']:.2f} cm (1 sd), storey height "
            f"{info['storey_h']:.2f} m (5th to 95th percentile over 1 m cells "
            f"{info.get('storey_h_p5', float('nan')):.2f} to {info.get('storey_h_p95', float('nan')):.2f} m)")

    wall_mask = cls == ids.get("wall", -999)
    res["wall_mask"] = wall_mask
    if wall_mask.sum() < MIN_GOOD_NORMALS:
        res["reason"] = f"only {int(wall_mask.sum())} wall points"
        log("  " + res["reason"] + ": nothing to measure")
        return res
    wall = lev[wall_mask]
    normals, planarity, good, phi, dirs, dev = wall_directions(wall)
    res.update(normals=normals, good=good, phi=phi, dev=dev, dirs=dirs)
    log(f"Step 3  {len(wall):,} wall points, {good.sum():,} with a trusted horizontal normal; "
        f"axis {np.degrees(phi):.2f} deg, {np.mean(dev < 10)*100:.1f}% within 10 deg, "
        f"median deviation {np.median(dev):.2f} deg")

    room = (~wall_mask) & (lev[:, 2] < 1.2)
    if "ceiling" in ids:
        room &= cls != ids["ceiling"]
    room_xy, all_xy = lev[room, :2], lev[:, :2]
    obj = ~wall_mask & (lev[:, 2] > 0.05)
    for name in ("floor", "ceiling"):
        if name in ids:
            obj &= cls != ids[name]
    obj_xy = lev[obj, :2]
    fams = []
    for k, d in enumerate(dirs):
        member, off, s, along = family_offsets(wall, normals, good, d)
        F = {"k": k, "dir": d, "along": along, "member": member, "off": off, "s": s,
             # every wall point on this family's axes: the thickness is measured on all of them,
             # not only on the points whose normal was trusted (that filter drops the points
             # nearest the other face first, and would make a thin wall read thicker)
             "off_all": wall @ d, "s_all": wall @ along, "z_all": wall[:, 2],
             "z": wall[member, 2], "faces": [], "cands": [], "pairs": [], "peaks": np.array([], int)}
        fams.append(F)
        if member.sum() == 0:
            continue
        F["centres"], F["hist"], F["smooth"] = profile(off)
        F["peaks"], F["faces"] = find_faces(F["centres"], F["smooth"], off, s, k)
        F["cands"], F["pairs"] = pair_faces(F["faces"], off, s, d, along, room_xy, all_xy, obj_xy,
                                            (F["off_all"], F["s_all"], F["z_all"]))
    res["families"] = fams

    n = 0
    for F in fams:
        F["measures"] = []
        # walls are named in a fixed order: family, then position across the room
        F["pairs"].sort(key=lambda p: F["faces"][p["i"]]["peak"])
        for p in F["pairs"]:
            a, b = F["faces"][p["i"]], F["faces"][p["j"]]
            p["pa"], p["pb"] = a["peak"], b["peak"]
            m = measure_thickness(p, F["off_all"], F["s_all"], F["z_all"],
                                  0.5 * info.get("storey_h", 3.0),
                                  seed=SEED + n)
            if m is None:
                p["verdict"] = "no cell where both faces are seen"
                continue
            n += 1
            F["measures"].append(m)
            # frame of the IFC wall: x along, y across from face A to face B, origin on face A
            y_ax = F["dir"].copy()
            x_ax = np.cross(y_ax, [0, 0, 1.0])
            x_ax /= np.linalg.norm(x_ax)
            s_start = p["s0"] if np.dot(x_ax, F["along"]) > 0 else p["s1"]
            origin = F["dir"] * m["face_a"] + F["along"] * s_start
            origin[2] = 0.0
            rooms_both = min(p["room_a"], p["room_b"]) >= ROOM_MIN_DENSITY
            res["walls"].append({
                "name": f"Wall {n}", "family": "AB"[F["k"]], "thickness": m["t"], "ci": m["ci"],
                "length": p["s1"] - p["s0"], "height": info.get("storey_h", 3.0),
                "x_axis": x_ax, "y_axis": y_ax, "origin": origin,
                "n_a": m["n_a"], "n_b": m["n_b"], "noise_a": m["noise_a"], "noise_b": m["noise_b"],
                "n_cells": m["n_cells"], "t_p5": m["t_min_cell"], "t_p95": m["t_max_cell"],
                "per_height": m["per_height"], "face_a": m["face_a"], "face_b": m["face_b"],
                "s0": p["s0"], "s1": p["s1"], "gap_peak": p["gap"],
                "valley_ratio": p["valley_ratio"], "covis": p["covis"],
                "room_a": p["room_a"], "room_b": p["room_b"],
                "reach_a": p["reach_a"], "reach_b": p["reach_b"], "rooms_both": bool(rooms_both),
                "lean_a_deg": m["lean_a_deg"], "lean_b_deg": m["lean_b_deg"],
                "t_floor": m["t_floor"], "t_ceiling": m["t_ceiling"],
                "t_median_cells": m["t_median_cells"], "cells_seen_a": m["cells_a"],
                "cells_seen_b": m["cells_b"], "clean_a": m["clean_a"], "clean_b": m["clean_b"],
                "objects_a": p["objects_a"], "objects_b": p["objects_b"],
                "idx_a": m["ia"], "idx_b": m["ib"]})
    if not res["walls"] and not res["reason"]:
        res["reason"] = "no pair of faces passed the four tests"
    return res


# ============================================================ Step 8: IFC4
def write_ifc(walls, path, scan_key="s3dis_office"):
    import uuid

    import ifcopenshell.api.aggregate as aggregate
    import ifcopenshell.api.context as context
    import ifcopenshell.api.geometry as geometry
    import ifcopenshell.api.material as material
    import ifcopenshell.api.project as project
    import ifcopenshell.api.pset as pset
    import ifcopenshell.api.root as root
    import ifcopenshell.api.spatial as spatial
    import ifcopenshell.api.type as ifctype
    import ifcopenshell.api.unit as unit
    import ifcopenshell.guid

    f = project.create_file(version="IFC4")
    proj = root.create_entity(f, ifc_class="IfcProject", name="Scan to BIM, measured walls")
    # metres, square metres, cubic metres: the scan's own units, so no silent x1000
    unit.assign_unit(f, units=[unit.add_si_unit(f, unit_type=u) for u in
                               ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT")])
    model = context.add_context(f, context_type="Model")
    body = context.add_context(f, context_type="Model", context_identifier="Body",
                               target_view="MODEL_VIEW", parent=model)
    site = root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = root.create_entity(f, ifc_class="IfcBuilding", name="S3DIS office")
    storey = root.create_entity(f, ifc_class="IfcBuildingStorey", name="Level 0")
    aggregate.assign_object(f, products=[site], relating_object=proj)
    aggregate.assign_object(f, products=[building], relating_object=site)
    aggregate.assign_object(f, products=[storey], relating_object=building)
    unknown = material.add_material(f, name="Unknown, not visible in a scan")

    for w in walls:
        # PARTITIONING only when a room shows on both sides; otherwise we do not know
        ptype = "PARTITIONING" if w.get("rooms_both") else "NOTDEFINED"
        # a wall TYPE carries the layer set (the thickness), the occurrence carries a usage
        wtype = root.create_entity(f, ifc_class="IfcWallType", predefined_type=ptype,
                                   name=f"Wall {1000 * w['thickness']:.0f} mm, as measured")
        lset = material.add_material_set(f, name=wtype.Name, set_type="IfcMaterialLayerSet")
        layer = material.add_layer(f, layer_set=lset, material=unknown)
        material.edit_layer(f, layer=layer, attributes={"LayerThickness": w["thickness"]})
        material.assign_material(f, products=[wtype], type="IfcMaterialLayerSet", material=lset)

        wall = root.create_entity(f, ifc_class="IfcWall", name=w["name"], predefined_type=ptype)
        ifctype.assign_type(f, related_objects=[wall], relating_type=wtype)
        material.assign_material(f, products=[wall], type="IfcMaterialLayerSetUsage")
        # the wall's own frame: x along the wall, y across it (face A to face B), z up
        m = np.eye(4)
        m[:3, 0], m[:3, 1] = w["x_axis"], w["y_axis"]
        m[:3, 2], m[:3, 3] = np.cross(w["x_axis"], w["y_axis"]), w["origin"]
        geometry.edit_object_placement(f, product=wall, matrix=m)
        rep = geometry.add_wall_representation(f, context=body, length=w["length"],
                                               height=w["height"], thickness=w["thickness"])
        geometry.assign_representation(f, product=wall, representation=rep)
        spatial.assign_container(f, products=[wall], relating_structure=storey)
        q = pset.add_qto(f, product=wall, name="Qto_WallBaseQuantities")
        pset.edit_qto(f, qto=q, properties={
            "Length": w["length"], "Width": w["thickness"], "Height": w["height"],
            "GrossSideArea": w["length"] * w["height"]})
        if w.get("rooms_both"):
            common = pset.add_pset(f, product=wall, name="Pset_WallCommon")
            pset.edit_pset(f, pset=common, properties={"IsExternal": False})
        ev = pset.add_pset(f, product=wall, name="Scan_Evidence")
        pset.edit_pset(f, pset=ev, properties={
            "ThicknessSource": "measured, two scanned faces compared cell by cell",
            "MeasuredThickness": w["thickness"],
            "ThicknessCI95Low": w["ci"][0], "ThicknessCI95High": w["ci"][1],
            "ThicknessP5Cells": w["t_p5"], "ThicknessP95Cells": w["t_p95"],
            "MatchedCells": w["n_cells"], "PointsFaceA": w["n_a"], "PointsFaceB": w["n_b"],
            "FaceNoiseA": w["noise_a"], "FaceNoiseB": w["noise_b"],
            "FaceLeanA_deg": w["lean_a_deg"], "FaceLeanB_deg": w["lean_b_deg"],
            "SeenFromBothSides": w["covis"], "RoomOnBothSides": bool(w.get("rooms_both"))})
    # stable GlobalIds: the same scan gives the same file, so a BIM tool sees an update, not a copy
    for e in f.by_type("IfcRoot"):
        key = f"{scan_key}/{e.is_a()}/{getattr(e, 'Name', '')}/{e.id()}"
        e.GlobalId = ifcopenshell.guid.compress(uuid.uuid5(uuid.NAMESPACE_URL, key).hex)
    f.write(path)
    return [e.GlobalId for e in f.by_type("IfcWall")]


# ============================================================ Step 9: read back
def layer_thickness(wall):
    import ifcopenshell.util.element as uel
    mat = uel.get_material(wall)
    if mat.is_a("IfcMaterialLayerSetUsage"):
        mat = mat.ForLayerSet
    return sum(l.LayerThickness for l in mat.MaterialLayers)


def read_back(path, walls, pts_by_wall):
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as uel
    import ifcopenshell.validate

    f = ifcopenshell.open(path)                       # a fresh read, nothing kept in memory
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, logger)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    report = {"schema": f.schema, "entities": len(list(f)), "errors": len(logger.statements),
              "ifcopenshell": ifcopenshell.version, "walls": []}
    meshes = []
    by_name = {e.Name: e for e in f.by_type("IfcWall")}
    for w, pts in zip(walls, pts_by_wall):
        ent = by_name[w["name"]]
        shape = ifcopenshell.geom.create_shape(settings, ent)
        v = np.array(shape.geometry.verts).reshape(-1, 3)
        tri = np.array(shape.geometry.faces).reshape(-1, 3)
        meshes.append((v, tri))
        across = v @ w["y_axis"]                      # the solid, measured across the wall
        geo_t = float(across.max() - across.min())
        psets = uel.get_psets(ent)
        # fit: every face point against the nearest side of the reopened solid
        o = pts @ w["y_axis"]
        d = np.minimum(np.abs(o - across.min()), np.abs(o - across.max()))
        report["walls"].append({
            "name": ent.Name, "guid": ent.GlobalId, "predefined": ent.PredefinedType,
            "geometry_t": geo_t, "qto_width": float(psets["Qto_WallBaseQuantities"]["Width"]),
            "layer_t": float(layer_thickness(ent)),
            "source": psets["Scan_Evidence"]["ThicknessSource"],
            "delta_mm": 1000 * max(abs(geo_t - w["thickness"]),
                                   abs(float(psets["Qto_WallBaseQuantities"]["Width"]) - w["thickness"]),
                                   abs(float(layer_thickness(ent)) - w["thickness"])),
            "fit_mean_cm": 100 * float(d.mean()), "fit_median_cm": 100 * float(np.median(d)),
            "fit_p95_cm": 100 * float(np.percentile(d, 95)),
            "fit_within_2cm": float((d < 0.02).mean()), "n_pts": int(len(pts))})
    return report, meshes


# ============================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()

    say("Step 1  load the classified scan")
    xyz, rgb, cls, names, ids = load_scan(DATA, SIDECAR)
    counts = {names[c]: int((cls == c).sum()) for c in sorted(names) if (cls == c).any()}
    say(f"  {len(xyz):,} points, {len(counts)} classes present")
    say("  " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    LOG["n_points"], LOG["classes"] = len(xyz), counts

    res = measure(xyz, cls, names, ids)
    lev, wall_mask, fams, walls = res["lev"], res["wall_mask"], res["families"], res["walls"]
    wall = lev[wall_mask]
    info = res["level"]
    LOG.update({k: v for k, v in info.items()})
    LOG.update(n_wall=int(wall_mask.sum()), n_good=int(res["good"].sum()),
               axis_deg=float(np.degrees(res["phi"])), within10=float(np.mean(res["dev"] < 10)),
               dev_median=float(np.median(res["dev"])))

    say("Step 4  project wall points onto each wall normal")
    for F in fams:
        say(f"  family {'AB'[F['k']]}: {F['member'].sum():,} points, profile "
            f"{len(F['centres'])} bins of 1 cm over {F['centres'][-1]-F['centres'][0]:.2f} m")
    say("Step 5  find the faces as density peaks")
    for F in fams:
        say(f"  family {'AB'[F['k']]}: {len(F['peaks'])} peaks -> {len(F['faces'])} face runs")
        for i, fc in enumerate(F["faces"]):
            say(f"    face {i}: offset {fc['peak']:+.3f} m, along {fc['s0']:+.2f}..{fc['s1']:+.2f} m, "
                f"{fc['n']} pts")
    say("Step 6  pair the two faces of a wall")
    for F in fams:
        for c in F["cands"]:
            say(f"  family {'AB'[F['k']]} faces {c['i']}+{c['j']}: gap {100*c['gap']:.1f} cm, "
                f"shared {c['s1']-c['s0']:.2f} m, valley {c['valley_ratio']:.2f}, seen from both "
                f"sides {100*c['covis']:.0f}% of {min(c['cells_a'], c['cells_b'])} cells | room beyond "
                f"{c['room_a']:.0f} / {c['room_b']:.0f} pts/m2, scan reaches {100*c['reach_a']:.0f} / "
                f"{100*c['reach_b']:.0f} cm beyond -> {c['verdict']}")

    say("Step 7  measure the thickness, cell by cell")
    pts_by_wall = []
    for w in walls:
        pts_by_wall.append(wall[np.r_[w["idx_a"], w["idx_b"]]])
        say(f"  {w['name']} (family {w['family']}): t = {100*w['thickness']:.2f} cm "
            f"[95% {100*w['ci'][0]:.2f}..{100*w['ci'][1]:.2f}] over {w['n_cells']} matched cells, "
            f"cells 5-95% {100*w['t_p5']:.1f}..{100*w['t_p95']:.1f} cm, length {w['length']:.2f} m")
        say(f"      face lines: {100*w['t_floor']:.1f} cm at the floor, {100*w['t_ceiling']:.1f} cm at the "
            f"ceiling; median of the cells {100*w['t_median_cells']:.2f} cm; clean cells "
            f"{w['clean_a']}/{w['cells_seen_a']} (A) and {w['clean_b']}/{w['cells_seen_b']} (B)")
        say(f"      lean of face A {w['lean_a_deg']:+.2f} deg, face B {w['lean_b_deg']:+.2f} deg; "
            f"face noise around its line {100*w['noise_a']:.2f} / {100*w['noise_b']:.2f} cm")
        for h0, h1, th, nc, _ in w["per_height"]:
            say(f"      height {h0:.2f}..{h1:.2f} m: {100*th:.2f} cm ({nc} cells)")
    if not walls:
        say("  nothing measured: " + res["reason"])

    say("Step 8  write the IFC4 walls")
    ifc_path = os.path.join(OUT, "measured_partitions.ifc")
    guids = write_ifc(walls, ifc_path)
    say(f"  {len(guids)} IfcWall written to {os.path.basename(ifc_path)} "
        f"({os.path.getsize(ifc_path)/1024:.1f} KB)")

    say("Step 9  open the IFC back and prove it")
    report, meshes = read_back(ifc_path, walls, pts_by_wall)
    say(f"  schema {report['schema']}, {report['entities']} entities, "
        f"{report['errors']} validation errors (IfcOpenShell {report['ifcopenshell']})")
    for r in report["walls"]:
        say(f"  {r['name']}: geometry {100*r['geometry_t']:.2f} cm, Qto Width "
            f"{100*r['qto_width']:.2f} cm, layer {100*r['layer_t']:.2f} cm, "
            f"worst delta {r['delta_mm']:.3f} mm")
        say(f"    face points to the reopened solid: mean {r['fit_mean_cm']:.2f} cm, "
            f"median {r['fit_median_cm']:.2f} cm, p95 {r['fit_p95_cm']:.2f} cm, "
            f"{100*r['fit_within_2cm']:.1f}% within 2 cm ({r['n_pts']} pts)")

    secs = time.perf_counter() - t_all
    say(f"Done in {secs:.2f} s")
    save_outputs(res, rgb, cls, names, report, meshes, secs)
    say(f"Outputs in {OUT}")


def save_outputs(res, rgb, cls, names, report, meshes, secs):
    lev, fams, walls = res["lev"], res["families"], res["walls"]
    np.savez_compressed(
        os.path.join(OUT, "state.npz"),
        lev=lev.astype(np.float32), rgb=rgb.astype(np.float32), cls=cls,
        wall_mask=res["wall_mask"], normals=res["normals"].astype(np.float32), good=res["good"],
        phi=res["phi"], dev=res["dev"].astype(np.float32), R=res["R"])
    fam_dump = []
    for F in fams:
        fam_dump.append({
            "k": F["k"], "dir": F["dir"].tolist(), "along": F["along"].tolist(),
            "centres": F.get("centres", np.array([])).tolist(),
            "hist": F.get("hist", np.array([])).tolist(),
            "smooth": F.get("smooth", np.array([])).tolist(), "peaks": np.asarray(F["peaks"]).tolist(),
            "member_idx": np.flatnonzero(F["member"]).tolist(),
            "faces": [{k: v for k, v in fc.items() if k != "idx"} |
                      {"idx": fc["idx"].tolist()} for fc in F["faces"]],
            "cands": F["cands"], "pairs": F["pairs"],
            "boot": [m["boot"].tolist() for m in F.get("measures", [])],
            "cells": [{"t": m["t_cells"].tolist(), "z": m["z_cells"].tolist(),
                       "keys": m["cells"], "lines": m["lines"]} for m in F.get("measures", [])]})
    walls_dump = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
                  for w in walls]
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"log": LOG, "families": fam_dump, "walls": walls_dump,
                   "ifc": report, "seconds": secs, "names": names}, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "ifc_meshes.npz"),
                        **{f"v{i}": v for i, (v, _) in enumerate(meshes)},
                        **{f"f{i}": fc for i, (_, fc) in enumerate(meshes)})
    # every number the article quotes, in one flat file
    metrics = {"n_points": LOG["n_points"], "classes": LOG["classes"], "seconds": round(secs, 2)}
    metrics.update({k: v for k, v in res["level"].items()})
    metrics.update(n_wall=LOG["n_wall"], n_good=LOG["n_good"], axis_deg=LOG["axis_deg"],
                   within10=LOG["within10"], dev_median=LOG["dev_median"])
    for F in fams:
        L = "AB"[F["k"]]
        metrics[f"fam{L}_points"] = int(F["member"].sum())
        metrics[f"fam{L}_peaks"] = int(len(F["peaks"]))
        metrics[f"fam{L}_faces"] = [{"peak": fc["peak"], "s0": fc["s0"], "s1": fc["s1"],
                                     "n": fc["n"]} for fc in F["faces"]]
        metrics[f"fam{L}_cands"] = [{k: c[k] for k in ("i", "j", "gap", "s0", "s1", "valley_ratio",
                                                       "covis", "cells_a", "cells_b", "cells_both",
                                                       "room_a", "room_b", "reach_a", "reach_b",
                                                       "objects_a", "objects_b", "verdict")} for c in F["cands"]]
        if len(F.get("centres", [])):
            metrics[f"fam{L}_bins"] = len(F["centres"])
            metrics[f"fam{L}_span"] = float(F["centres"][-1] - F["centres"][0])
    metrics["n_faces"] = sum(len(F["faces"]) for F in fams)
    metrics["walls"] = [{k: w[k] for k in (
        "name", "family", "thickness", "ci", "length", "height", "n_cells", "t_p5", "t_p95",
        "lean_a_deg", "lean_b_deg", "noise_a", "noise_b", "n_a", "n_b", "covis", "room_a",
        "room_b", "reach_a", "reach_b", "rooms_both", "per_height", "gap_peak", "s0", "s1",
        "face_a", "face_b", "t_floor", "t_ceiling", "t_median_cells", "cells_seen_a",
        "cells_seen_b", "clean_a", "clean_b", "objects_a", "objects_b")} for w in walls]
    metrics["ifc"] = {k: report[k] for k in ("schema", "entities", "errors", "ifcopenshell")}
    metrics["ifc"]["walls"] = report["walls"]
    metrics["ifc"]["kb"] = os.path.getsize(os.path.join(OUT, "measured_partitions.ifc")) / 1024
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1, default=float)


if __name__ == "__main__":
    main()
