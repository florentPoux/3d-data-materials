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
"""How far to the way out: walking distance to the nearest exit on a scanned office floor.

The stage taught in P24 article 04 (v2). Input: the classified S3DIS Area 1 floor (code/data).
Output (code/out): the distance field, the escape route from every room, the dead-end corridors
and the common paths, the block-an-exit sweep, a per-point PLY and a JSON a dispatcher can read.

Run:  python evacuation_distance.py            (writes code/out/)
Seeded and deterministic: the only randomness is in prepare_s3dis.py (seed 0).
The result does not depend on the order of the points (tests/test_determinism.py checks it).
"""
import json
import os
import time

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import networkx as nx
from skimage.morphology import disk, remove_small_objects, skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.environ.get("EVAC_SCAN", os.path.join(HERE, "data", "s3dis_area1_open_5cm.ply"))
CLASSES = os.path.join(HERE, "data", "classes.json")
OUT = os.environ.get("EVAC_OUT", os.path.join(HERE, "out"))

# ---- configuration: the only block you should need to touch -----------------------------
CELL = 0.05                  # grid cell, metres
BAND = (0.10, 1.80)          # height above the local floor where something blocks a person, metres
BODY_R = 0.25                # half a shoulder width: a body's centre keeps this far from anything
DOOR_DEPTH = 0.45            # floor added on each side of a door leaf (no floor is scanned under a wall)
SPECK = 3                    # a blocked blob smaller than this many cells is a stray point, not furniture
NEIGHBOURS = 16              # moves of the walking graph: 8, 16 (default, knight moves) or 32
SPEED = 1.2                  # unimpeded walking speed, m/s (a common design value)
EXIT_R = 0.75                # walkable cells this close to an exit's door leaf are joined to it
LIMIT_M = 60.96              # exit access travel distance to compare with (IBC, B, no sprinklers: 200 ft)
DEAD_END_M = 6.10            # dead-end corridor length to compare with (IBC general: 20 ft)
COMMON_PATH_M = 22.86        # common path of egress travel to compare with (IBC, B, no sprinklers: 75 ft)
DOOR_MAX = 1.20              # wall gaps up to this wide are doorways when splitting rooms
MIN_ROOM_M2 = 2.0            # a piece of floor smaller than this is not a room
MIN_DEAD_END_M = 1.0         # one-way floor shallower than this is a skeleton spur, not a place
MIN_CORRIDOR_M = 2.0         # a dead-end corridor shorter than this is a door niche or an alcove
CORRIDOR_MAX_W = 2.0         # one-way circulation floor wider than this on average (wall to wall) is
                             # a bay of the open hall, a common path, not a corridor
PROBE_XY = (-18.5, 30.0)     # the "clicked" point for the single-route demo (Step 6)

# The exits, declared the way you would click them on the plan: doors that lead off the
# scanned floor, onto the stair cores, lobbies and street S3DIS never captured. (x, y) metres.
EXITS = {
    "A": (-9.59, -1.96),     # south entrance
    "B": (-13.39, 6.95),     # west door of the south core
    "C": (-5.85, 7.09),      # east door of the south core
    "D": (-5.81, 35.02),     # door of the north core, east corridor
    "E": (-13.82, 41.74),    # north-west core
    "F": (-5.37, 40.47),     # north-east core
    "G": (-15.77, 44.68),    # far north-west
    "H": (-3.30, 44.09),     # far north-east
}

ROLE_NAMES = {
    "floor": {"floor", "ground", "carpet"},
    "wall": {"wall", "column", "window", "board", "beam"},
    "door": {"door"},
    "overhead": {"ceiling"},
}
# everything else (table, chair, bookcase, sofa, clutter...) stands in the way


def say(step, msg):
    print(f"[{step}] {msg}", flush=True)


# ---- the moves of the walking graph ------------------------------------------------------
def crossed(dr, dc):
    """Every cell the straight segment between the two cell centres passes through or touches,
    endpoints excluded. A move is allowed only if all of them are walkable: no corner cutting."""
    n = 64 * max(abs(dr), abs(dc))
    t = np.linspace(0.0, 1.0, n + 1)[1:-1]
    cells = set()
    for e1 in (-1e-9, 1e-9):                       # a segment through a corner touches both sides
        for e2 in (-1e-9, 1e-9):
            r = np.floor(t * dr + 0.5 + e1).astype(int)
            c = np.floor(t * dc + 0.5 + e2).astype(int)
            cells |= set(zip(r.tolist(), c.tolist()))
    cells -= {(0, 0), (dr, dc)}
    return sorted(cells)


def moves(neighbours=NEIGHBOURS):
    """Half of the move set (the other half is the same moves backwards): (dr, dc, crossed, length)."""
    base = [(0, 1), (1, 0), (1, 1), (1, -1)]
    if neighbours >= 16:
        base += [(1, 2), (2, 1), (1, -2), (2, -1)]
    if neighbours >= 32:
        base += [(1, 3), (3, 1), (1, -3), (3, -1), (2, 3), (3, 2), (2, -3), (3, -2)]
    return [(dr, dc, crossed(dr, dc), float(np.hypot(dr, dc))) for dr, dc in base]


def all_moves(neighbours=NEIGHBOURS):
    out = []
    for dr, dc, cr, w in moves(neighbours):
        out.append((dr, dc, cr, w))
        out.append((-dr, -dc, [(-a, -b) for a, b in cr], w))
    return out


# ---- Step 1 ------------------------------------------------------------------------------
def load_floor(path=None, classes=CLASSES):
    """Read the classified floor and give every point a role from its class name."""
    v = PlyData.read(path or SCAN)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1)
    with open(classes, encoding="utf-8") as f:
        names = {int(k): n.lower() for k, n in json.load(f).items()}
    cls = np.rint(np.asarray(v["classification"], float)).astype(int)
    role = np.full(len(cls), "obstacle", dtype=object)
    for r, ok in ROLE_NAMES.items():
        role[np.isin(cls, [k for k, n in names.items() if n in ok])] = r
    floor_z = float(np.median(xyz[role == "floor", 2]))
    counts = {r: int((role == r).sum()) for r in ["floor", "wall", "door", "obstacle"]}
    say("Step 1", f"{len(xyz):,} points, {len(names)} classes, roles {counts}, floor at z = {floor_z:.3f} m")
    return xyz, rgb, role, floor_z, counts


# ---- Step 2 ------------------------------------------------------------------------------
def grid_origin(xy):
    """The grid's corner, snapped to a whole number of cells and 10 cells clear of the data, so
    the cells are the same whatever the file's first or lowest point is."""
    return np.floor(xy.min(0) / CELL) * CELL - 10 * CELL


def build_walkable(xyz, role, floor_z=None):
    """One short pass: the floor's own height per cell, floor cells, blocked cells in the body
    band, the doors opened, then the cells where a body's centre fits. (Neurones cuts doorways
    along each wall and splits rooms properly; that part is course material. This is the quick,
    tested version.)"""
    origin = grid_origin(xyz[:, :2])
    ij = np.floor((xyz[:, :2] - origin) / CELL).astype(int)
    W, H = ij.max(0) + 11
    flat = ij[:, 1] * W + ij[:, 0]

    def raster(mask):
        g = np.zeros(H * W, bool)
        g[flat[mask]] = True
        return g.reshape(H, W)

    # the floor's height in every cell (mean of its floor points), borrowed from the nearest
    # floor cell where none was scanned: a slab that sags does not become a step
    fl = role == "floor"
    n_fl = np.bincount(flat[fl], minlength=H * W).reshape(H, W)
    z_fl = np.bincount(flat[fl], weights=xyz[fl, 2], minlength=H * W).reshape(H, W)
    seen = n_fl > 0
    _, (nr, nc) = ndi.distance_transform_edt(~seen, return_indices=True)
    height = np.where(seen, z_fl / np.maximum(n_fl, 1), 0.0)[nr, nc]
    hz = xyz[:, 2] - height.ravel()[flat]

    band = (hz >= BAND[0]) & (hz < BAND[1])
    floor = ndi.binary_closing(seen, disk(2))                                  # 10 cm pinholes
    holes = ndi.binary_fill_holes(floor) & ~floor
    lab, n = ndi.label(holes)
    small = np.flatnonzero(np.bincount(lab.ravel())[1:] * CELL ** 2 < 0.25) + 1
    floor |= np.isin(lab, small)                                               # enclosed holes < 0.25 m2
    blocked = raster(band & np.isin(role, ["wall", "obstacle"]))
    blocked = ndi.binary_closing(blocked, np.ones((3, 3), bool))                # one-cell holes in a table
    blocked = remove_small_objects(blocked, SPECK)                              # a stray point is not a desk
    door = ndi.binary_closing(raster(band & (role == "door")), np.ones((3, 3), bool))
    blocked &= ~door                                                            # the leaf opens; the jambs stay
    floor |= ndi.binary_dilation(door, disk(int(round(DOOR_DEPTH / CELL))))     # floor under the threshold
    free = floor & ~blocked
    # distance from a cell's centre to the EDGE of the nearest non-free cell = centre-to-centre - half a cell
    clearance = (ndi.distance_transform_edt(free) - 0.5) * CELL
    walk = clearance >= BODY_R                                                  # a body's centre fits
    wall = raster(band & (role == "wall"))
    grid = dict(origin=origin, shape=(H, W), free=free, walk=walk, door=door, wall=wall,
                blocked=blocked, floor=floor, clearance=clearance, height=height,
                scanned=raster(np.ones(len(xyz), bool)))
    say("Step 2", f"grid {H} x {W} at {CELL*100:.0f} cm: free {free.sum()*CELL**2:.1f} m2, "
                  f"walkable {walk.sum()*CELL**2:.1f} m2")
    return grid


def to_cell(grid, x, y):
    c = np.floor((np.array([x, y]) - grid["origin"]) / CELL).astype(int)
    return int(c[1]), int(c[0])


def to_xy(grid, rc):
    rc = np.atleast_2d(rc)
    return grid["origin"] + (rc[:, ::-1] + 0.5) * CELL


# ---- Step 3 ------------------------------------------------------------------------------
def place_exits(grid, exits=EXITS):
    """Each exit is its door leaf: the door blob under the clicked point. Every walkable cell
    within EXIT_R of the leaf is joined to the exit, charged its straight distance to the leaf,
    so nobody is credited with a walk they have not done."""
    H, W = grid["shape"]
    lab, _ = ndi.label(grid["door"], structure=np.ones((3, 3)))
    src = {}
    for name, (x, y) in exits.items():
        r0, c0 = to_cell(grid, x, y)
        leaf = np.zeros((H, W), bool)
        win = lab[max(0, r0 - 10):r0 + 11, max(0, c0 - 10):c0 + 11]        # a door blob within 50 cm
        ids = np.unique(win[win > 0])
        if len(ids):
            rr, cc = np.nonzero(np.isin(lab, ids))
            k = ids[np.argmin([np.min((rr[lab[rr, cc] == i] - r0) ** 2 + (cc[lab[rr, cc] == i] - c0) ** 2)
                               for i in ids])]
            leaf = lab == k
        else:                                                               # no door scanned: the point
            leaf[r0, c0] = True
        dist = ndi.distance_transform_edt(~leaf) * CELL
        cells = np.flatnonzero((grid["walk"] & (dist <= EXIT_R)).ravel())
        src[name] = dict(cells=cells, offset=dist.ravel()[cells], leaf=np.flatnonzero(leaf.ravel()))
        say("Step 3", f"exit {name} at ({x:.2f}, {y:.2f}): leaf {leaf.sum()} cells, {len(cells)} walkable cells "
                      f"joined, {dist.ravel()[cells].min():.2f} to {dist.ravel()[cells].max():.2f} m from the leaf")
    return src


# ---- Step 4 ------------------------------------------------------------------------------
def walking_graph(walk, neighbours=NEIGHBOURS):
    """Graph over the walkable cells, edge weight = step length in metres, both directions.
    A move is allowed only if every cell its straight line crosses is walkable."""
    H, W = walk.shape
    idx = np.full(walk.shape, -1, np.int64)
    idx[walk] = np.arange(walk.sum())
    pad = 3
    wp = np.pad(walk, pad)
    rows, cols, wts = [], [], []
    for dr, dc, cr, w in moves(neighbours):
        ok = walk & wp[pad + dr:pad + dr + H, pad + dc:pad + dc + W]
        for a, b in cr:
            ok &= wp[pad + a:pad + a + H, pad + b:pad + b + W]
        r, c = np.nonzero(ok)
        rows.append(idx[r, c])
        cols.append(idx[r + dr, c + dc])
        wts.append(np.full(len(r), w * CELL))
    n = int(walk.sum())
    a, b, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    G = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return G, idx


def attach_exits(G, idx, src):
    """The virtual source, made literal: one extra node per exit, with a one-way edge to each of
    its joined cells weighted by that cell's distance to the door leaf. One-way, so a blocked
    exit's node can never be used as a shortcut between two of its cells."""
    n = G.shape[0]
    names = list(src)
    G = G.tocoo()
    r, c, w = [G.row], [G.col], [G.data]
    for k, e in enumerate(names):
        nodes = idx.ravel()[src[e]["cells"]]
        r.append(np.full(len(nodes), n + k))
        c.append(nodes)
        w.append(np.asarray(src[e]["offset"], float))     # an explicit 0 is an edge in scipy.sparse.csgraph
    m = n + len(names)
    Ga = coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(m, m)).tocsr()
    return Ga, {e: n + k for k, e in enumerate(names)}


def distance_field(grid, G, idx, src, open_exits=None, exit_node=None):
    """Multi-source Dijkstra: every open exit's node is a source at distance 0, and each walkable
    cell gets the walking distance to whichever open exit is nearest, in ONE search."""
    open_exits = list(src) if open_exits is None else list(open_exits)
    if exit_node is None:
        G, exit_node = attach_exits(G, idx, src)
    t0 = time.perf_counter()
    dist, _, origin = dijkstra(G, directed=True, indices=[exit_node[e] for e in open_exits],
                               min_only=True, return_predecessors=True)
    ms = (time.perf_counter() - t0) * 1000
    n = int(grid["walk"].sum())
    D = np.full(grid["shape"], np.inf)
    D[grid["walk"]] = dist[:n]
    name_of = {v: k for k, v in exit_node.items()}
    serve = np.full(grid["shape"], "", dtype=object)
    serve[grid["walk"]] = np.array([name_of.get(int(o), "") for o in origin[:n]], dtype=object)
    # the last straight leg: from a joined cell to the leaf of an open exit
    leave = np.full(grid["shape"], np.inf).ravel()
    for e in open_exits:
        np.minimum.at(leave, src[e]["cells"], src[e]["offset"])
    return D, serve, ms, leave.reshape(grid["shape"])


# ---- Step 5 ------------------------------------------------------------------------------
def read_against_limit(D, walk, limit=LIMIT_M, speed=SPEED, quiet=False):
    finite = np.isfinite(D) & walk
    d = D[finite]
    pockets, npk = ndi.label(walk & ~np.isfinite(D), structure=np.ones((3, 3)))
    stats = dict(reach_m2=float(finite.sum() * CELL ** 2),
                 no_way_out_m2=float((walk & ~np.isfinite(D)).sum() * CELL ** 2), pockets=int(npk),
                 farthest_m=float(d.max()), mean_m=float(d.mean()), median_m=float(np.median(d)),
                 p95_m=float(np.percentile(d, 95)), farthest_s=float(d.max() / speed),
                 share_over_limit=float((d > limit).mean()),
                 share_over_half=float((d > limit / 2).mean()))
    if not quiet:
        say("Step 5", f"farthest {stats['farthest_m']:.1f} m ({stats['farthest_s']:.0f} s at {speed} m/s), "
                  f"mean {stats['mean_m']:.1f} m, over {limit:.1f} m: {100*stats['share_over_limit']:.1f} %, "
                  f"no way out {stats['no_way_out_m2']:.1f} m2 in {npk} pockets")
    return stats


# ---- Step 6 ------------------------------------------------------------------------------
def descend(D, walk, start, leave, neighbours=NEIGHBOURS):
    """The escape route from any cell: step to the neighbour that minimises D(n) + step length,
    or leave through the door when that is cheaper. On a Dijkstra field the route's length is
    exactly D(start). Returns the cells and the length (the last leg to the leaf included)."""
    H, W = D.shape
    r, c = start
    path, length = [(r, c)], 0.0
    if not np.isfinite(D[r, c]):
        return path, np.inf
    mv = all_moves(neighbours)
    for _ in range(H * W):
        best, bw, bn = leave[r, c], leave[r, c], None               # option 0: out through the door
        for dr, dc, cr, w in mv:
            rn, cn = r + dr, c + dc
            if not (0 <= rn < H and 0 <= cn < W) or not walk[rn, cn]:
                continue
            if any(not (0 <= r + a < H and 0 <= c + b < W) or not walk[r + a, c + b] for a, b in cr):
                continue
            v = D[rn, cn] + w * CELL
            if v < best - 1e-12:
                best, bw, bn = v, w * CELL, (rn, cn)
        length += bw
        if bn is None:                                               # through the door: done
            return path, length
        r, c = bn
        path.append(bn)
    raise RuntimeError("descent did not reach an exit")


# ---- Step 7 ------------------------------------------------------------------------------
def split_rooms(grid, src=None):
    """Quick rooms: close every gap in the walls up to DOOR_MAX along the two grid axes, only
    within 0.9 m of a door, so each doorway gets a virtual leaf; then one connected piece of
    walkable floor per space. A space is circulation (corridor, hall, lobby) when it is the
    largest, when an exit opens from it, or when four doors or more open onto it. (Neurones'
    room stage is finer: wall directions, corridor shapes, names. Course material.)"""
    L = int(round(DOOR_MAX / CELL))
    shut = grid["wall"] | grid["door"]
    closed = ndi.binary_closing(shut, np.ones((1, L), bool)) | ndi.binary_closing(shut, np.ones((L, 1), bool))
    near_door = ndi.binary_dilation(grid["door"], disk(int(round(0.75 * DOOR_MAX / CELL))))
    cut = grid["walk"] & ~(shut | (closed & near_door))
    lab, n = ndi.label(cut, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=n + 1) * CELL ** 2
    keep = [k for k in range(1, n + 1) if sizes[k] >= MIN_ROOM_M2]
    order = sorted(keep, key=lambda k: (-sizes[k], int(np.argmax(lab.ravel() == k))))
    rooms = np.zeros_like(lab)
    for i, k in enumerate(order, start=1):
        rooms[lab == k] = i
    # every walkable cell the cut left out (door throats, thin strips) joins the space it touches first
    _, (nr, nc) = ndi.distance_transform_edt(rooms == 0, return_indices=True)
    rooms = np.where(grid["walk"], rooms[nr, nc], 0)
    # doors per space: door blobs whose 30 cm neighbourhood touches it
    dl, nd = ndi.label(grid["door"], structure=np.ones((3, 3)))
    ring = ndi.grey_dilation(dl, footprint=disk(int(round(0.3 / CELL))))
    pairs = np.unique(np.c_[rooms.ravel(), ring.ravel()][(rooms.ravel() > 0) & (ring.ravel() > 0)], axis=0)
    doors = np.bincount(pairs[:, 0], minlength=len(order) + 1)
    at_exit = set()
    for e in (src or {}).values():
        at_exit |= set(np.unique(rooms.ravel()[e["cells"]]).tolist())
    kind = [""] + ["circulation" if (i == 1 or i in at_exit or doors[i] >= 4) else "room"
                   for i in range(1, len(order) + 1)]
    return rooms, [float(sizes[k]) for k in order], kind, doors


def route_every_room(D, walk, rooms, serve, grid, leave):
    out = []
    for k in range(1, rooms.max() + 1):
        m = (rooms == k) & np.isfinite(D)
        if not m.any():
            out.append(dict(room=k, reachable=False))
            continue
        flat = np.where(m, D, -1).argmax()
        start = np.unravel_index(flat, D.shape)
        path, length = descend(D, walk, start, leave)
        out.append(dict(room=k, reachable=True, area_m2=float(m.sum() * CELL ** 2),
                        worst_xy=[round(float(v), 2) for v in to_xy(grid, start)[0]],
                        exit=serve[start], dist_m=float(D[start]), route_m=float(length),
                        time_s=float(D[start] / SPEED), path=path))
    return out


# ---- Step 8 ------------------------------------------------------------------------------
def skeleton_frame(grid, G, idx):
    """The walkable floor's centre line as a graph (one node per skeleton pixel, 8-neighbour
    edges), plus, for every walkable cell, the skeleton pixel it belongs to (nearest along the
    floor, by one more multi-source Dijkstra). It depends on the scan only: build it once."""
    if "frame" in grid:
        return grid["frame"]
    walk = grid["walk"]
    sk = skeletonize(walk)
    pix = np.argwhere(sk)
    node = {tuple(p): i for i, p in enumerate(pix.tolist())}
    S = nx.Graph()
    S.add_nodes_from(range(len(pix)))
    for i, (r, c) in enumerate(pix.tolist()):
        for dr, dc in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            j = node.get((r + dr, c + dc))
            if j is not None:
                S.add_edge(i, j)
    n = int(walk.sum())
    sk_nodes = idx[pix[:, 0], pix[:, 1]]
    _, _, nearest = dijkstra(G[:n, :n], directed=True, indices=sk_nodes, min_only=True,
                             return_predecessors=True)
    owner = np.full(walk.shape, -1, np.int64)
    pos = np.full(n, -1, np.int64)
    pos[sk_nodes] = np.arange(len(sk_nodes))
    owner[walk] = np.where(nearest >= 0, pos[np.maximum(nearest, 0)], -1)
    grid["frame"] = (pix, S, owner)
    return grid["frame"]


def dead_ends(grid, G, idx, rooms, kind, D, src, open_exits):
    """Floor whose only way out goes back through one opening. On the centre-line graph, hang
    every open exit off one extra node, OUT. A passage whose removal cuts a piece off OUT is a
    bridge. Remove all bridges; the centre line still joined to OUT has two separate ways out,
    and so does every cell it owns (the TWO-WAY floor). Every other cell has one way out. Its
    depth, the walk back to the nearest two-way cell (or to the exit itself, when a single exit
    serves it), is its COMMON PATH of travel: one more multi-source Dijkstra. The part of the
    one-way floor that lies in a corridor, hall or lobby is a DEAD-END CORRIDOR, and its length
    is the walk back from its deepest circulation cell. An office behind one door is a common
    path, not a dead end, exactly as a fire engineer reads it. G is the graph with the exit
    nodes attached (attach_exits), so they are the last len(src) nodes, in src order."""
    pix, S, owner = skeleton_frame(grid, G, idx)
    S = S.copy()
    for e in open_exits:                           # the skeleton pixels that own the exit's cells
        hooks = np.unique(owner.ravel()[src[e]["cells"]])
        for i in hooks[hooks >= 0]:
            S.add_edge(int(i), "OUT")
    # an edge into OUT is the door itself, never a passage you can lose: two ways round a loop
    # to the same door are two ways out, so those edges are never counted as bridges
    bridges = [b for b in nx.bridges(S) if "OUT" not in b]
    S2 = S.copy()
    S2.remove_edges_from(bridges)
    core = nx.node_connected_component(S2, "OUT") if "OUT" in S2 else set()
    core_px = np.array(sorted(v for v in core if v != "OUT"), dtype=np.int64)
    walk = grid["walk"]
    reach = np.isfinite(D) & walk
    two_way = reach & np.isin(owner, core_px)
    n = int(walk.sum())
    exit_node = {e: n + k for k, e in enumerate(src)}
    starts = np.r_[idx[two_way], [exit_node[e] for e in open_exits]].astype(np.int64)
    back, _, origin = dijkstra(G, directed=True, indices=starts, min_only=True, return_predecessors=True)
    one_way = reach & ~two_way
    depth = np.full(grid["shape"], np.nan)
    depth[one_way] = back[idx[one_way]]
    circ = np.isin(rooms, [i for i, k in enumerate(kind) if k == "circulation"])
    lab, nlab = ndi.label(one_way, structure=np.ones((3, 3)))
    regions = []
    exit_of = {v: k for k, v in exit_node.items()}
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        cells = np.zeros(grid["shape"], bool)
        cells[sl] = lab[sl] == k
        dk = np.where(cells, depth, -1)
        tip = np.unravel_index(dk.argmax(), D.shape)
        cc = cells & circ
        corr = float(np.max(depth[cc])) if cc.any() else 0.0
        ctip = np.unravel_index(np.where(cc, depth, -1).argmax(), D.shape) if cc.any() else tip
        o = int(origin[idx[tip]])
        if o >= n:                                  # its common path runs all the way to an exit
            mouth = [round(float(v), 2) for v in EXITS.get(exit_of[o], (np.nan, np.nan))]
        else:
            mouth = [round(float(v), 2) for v in to_xy(grid, np.argwhere(idx == o)[0])[0]]
        regions.append(dict(depth_m=float(depth[tip]), corridor_m=corr, area_m2=float(cells.sum() * CELL ** 2),
                            corridor_area_m2=float(cc.sum() * CELL ** 2),
                            room=int(np.bincount(rooms[cells]).argmax()),
                            tip_xy=[round(float(v), 2) for v in to_xy(grid, tip)[0]],
                            corridor_tip_xy=[round(float(v), 2) for v in to_xy(grid, ctip)[0]],
                            mouth_xy=mouth, to_exit=bool(o >= n)))
    regions = [r for r in regions if r["depth_m"] >= MIN_DEAD_END_M]
    regions.sort(key=lambda r: (-r["depth_m"], r["tip_xy"]))
    share = float(one_way.sum() / max(1, reach.sum()))
    return depth, regions, share, len(bridges)


def corridor_dead_ends(regions):
    """Dead-end corridors: one-way circulation floor at least MIN_CORRIDOR_M deep and narrow like a
    corridor (mean walkable width = area / length, a body's width narrower than wall to wall)."""
    wmax = CORRIDOR_MAX_W - 2 * BODY_R
    return sorted((r for r in regions if r["corridor_m"] >= MIN_CORRIDOR_M
                   and r["corridor_area_m2"] / r["corridor_m"] <= wmax),
                  key=lambda r: (-r["corridor_m"], r["tip_xy"]))


# ---- Step 9 ------------------------------------------------------------------------------
def what_if(grid, G, idx, src, rooms, kind, open_exits, exit_node, base_routes=None):
    """Re-solve with a set of exits open and report what the building would feel."""
    D, serve, ms, leave = distance_field(grid, G, idx, src, open_exits, exit_node)
    s = read_against_limit(D, grid["walk"], quiet=True)
    rr = route_every_room(D, grid["walk"], rooms, serve, grid, leave)
    depth, regions, share, _ = dead_ends(grid, G, idx, rooms, kind, D, src, open_exits)
    corr = corridor_dead_ends(regions)
    row = dict(open=list(open_exits), farthest_m=s["farthest_m"], mean_m=s["mean_m"],
               share_over_limit=s["share_over_limit"], single_path_share=share,
               longest_common_path_m=max([r["depth_m"] for r in regions], default=0.0),
               longest_one_way_circulation_m=max([r["corridor_m"] for r in regions], default=0.0),
               corridor_dead_ends=len(corr),
               longest_dead_end_corridor_m=max([r["corridor_m"] for r in corr], default=0.0),
               solve_ms=ms)
    if base_routes is not None:
        row["rooms_rerouted"] = sum(1 for a, b in zip(rr, base_routes)
                                    if a.get("reachable") and a.get("exit") != b.get("exit"))
    return row, D, rr, depth


def block_each_exit(grid, G, idx, src, rooms, kind, exit_node, base_routes):
    rows = []
    for e in src:
        row, *_ = what_if(grid, G, idx, src, rooms, kind, [x for x in src if x != e], exit_node, base_routes)
        row["blocked"] = e
        rows.append(row)
        say("Step 9", f"block {e}: farthest {row['farthest_m']:.1f} m, mean {row['mean_m']:.1f} m, "
                      f"{row['rooms_rerouted']} spaces rerouted, longest one-way circulation "
                      f"{row['longest_one_way_circulation_m']:.1f} m, longest common path "
                      f"{row['longest_common_path_m']:.1f} m, solve {row['solve_ms']:.0f} ms")
    return rows


def block_worst_first(grid, G, idx, src, rooms, kind, exit_node):
    """The cascade: keep blocking whichever exit hurts the farthest point most, down to one."""
    open_ = list(src)
    rows = []
    while len(open_) > 1:
        trials = []
        for e in open_:
            row, *_ = what_if(grid, G, idx, src, rooms, kind, [x for x in open_ if x != e], exit_node)
            trials.append((round(row["farthest_m"], 9), round(row["mean_m"], 9), e, row))
        far, mean, e, row = max(trials, key=lambda t: (t[0], t[1], t[2]))
        open_.remove(e)
        row["blocked"] = e
        rows.append(row)
        say("Step 9", f"cascade: block {e} -> open {''.join(open_)}: farthest {far:.1f} m, mean {mean:.1f} m, "
                      f"over limit {100*row['share_over_limit']:.1f} %, longest one-way circulation "
                      f"{row['longest_one_way_circulation_m']:.1f} m")
    return rows


# ---- Step 10 -----------------------------------------------------------------------------
def export(grid, xyz, rgb, D, depth, routes, pockets):
    """Per-point distance for the 3D viewer, routes and dead ends for a dispatcher, grids for reuse."""
    os.makedirs(OUT, exist_ok=True)
    ij = np.floor((xyz[:, :2] - grid["origin"]) / CELL).astype(int)
    # a point off the walkable grid (a wall, a desk) takes the distance of the nearest walkable cell
    _, (nr, nc) = ndi.distance_transform_edt(~np.isfinite(D), return_indices=True)
    d_pt = D[nr[ij[:, 1], ij[:, 0]], nc[ij[:, 1], ij[:, 0]]]
    d_pt = np.where(np.isfinite(d_pt), d_pt, -1.0).astype(np.float32)
    v = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                                  ("green", "u1"), ("blue", "u1"), ("evac_distance_m", "f4")])
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    v["red"], v["green"], v["blue"] = rgb.T
    v["evac_distance_m"] = d_pt
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(OUT, "area1_evac_distance.ply"))
    plan = dict(cell_m=CELL, origin=grid["origin"].tolist(), speed_mps=SPEED, neighbours=NEIGHBOURS,
                exits=EXITS,
                spaces=[{k: (to_xy(grid, v).round(2).tolist() if k == "path" else v) for k, v in r.items()}
                        for r in routes], one_way_floor=pockets)
    with open(os.path.join(OUT, "evacuation_routes.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1)
    np.savez_compressed(os.path.join(OUT, "evacuation_grids.npz"), D=np.where(np.isfinite(D), D, -1),
                        depth=np.nan_to_num(depth, nan=-1), walk=grid["walk"], free=grid["free"],
                        blocked=grid["blocked"], door=grid["door"], wall=grid["wall"],
                        scanned=grid["scanned"], origin=grid["origin"], cell=CELL)
    sizes = {f: os.path.getsize(os.path.join(OUT, f)) for f in
             ["area1_evac_distance.ply", "evacuation_routes.json", "evacuation_grids.npz"]}
    say("Step 10", "wrote " + ", ".join(f"{k} ({v/1e3:.0f} kB)" for k, v in sizes.items()))
    return sizes


def solve(xyz, rgb, role, floor_z, m=None, sweep=True, write=True):
    """The whole pipeline on one floor. Returns the metrics dict (and fills m if given)."""
    t0 = time.perf_counter()
    m = {} if m is None else m
    m["floor_z"] = floor_z
    grid = build_walkable(xyz, role, floor_z)                                 # Step 2
    m["grid"] = list(grid["shape"])
    m["free_m2"], m["walk_m2"] = float(grid["free"].sum() * CELL ** 2), float(grid["walk"].sum() * CELL ** 2)
    src = place_exits(grid)                                                   # Step 3
    m["exit_cells"] = {k: int(len(v["cells"])) for k, v in src.items()}
    m["exit_leaf_cells"] = {k: int(len(v["leaf"])) for k, v in src.items()}
    Gc, idx = walking_graph(grid["walk"])                                     # Step 4
    G, exit_node = attach_exits(Gc, idx, src)
    m["neighbours"] = NEIGHBOURS
    m["graph_nodes"], m["graph_edges"] = int(Gc.shape[0]), int(Gc.nnz // 2)
    say("Step 4", f"walking graph ({NEIGHBOURS} neighbours): {Gc.shape[0]:,} cells, {Gc.nnz // 2:,} moves")
    D, serve, ms, leave = distance_field(grid, G, idx, src, exit_node=exit_node)
    say("Step 4", f"multi-source Dijkstra from {len(src)} exits in {ms:.0f} ms")
    m["solve_ms"] = ms
    f = np.isfinite(D)
    m["metric_check"] = {}
    for nb in (8, 32):                                                        # how much the move set matters
        Gn, _ = walking_graph(grid["walk"], nb)
        Gn, en = attach_exits(Gn, idx, src)
        Dn, *_ = distance_field(grid, Gn, idx, src, exit_node=en)
        ok = f & np.isfinite(Dn) & (D > 5)
        m["metric_check"][str(nb)] = dict(farthest_m=float(Dn[np.isfinite(Dn)].max()),
                                          mean_m=float(Dn[np.isfinite(Dn)].mean()),
                                          mean_vs16_pct=float(100 * np.mean(Dn[ok] / D[ok] - 1)),
                                          max_vs16_pct=float(100 * np.max(Dn[ok] / D[ok] - 1)),
                                          min_vs16_pct=float(100 * np.min(Dn[ok] / D[ok] - 1)))
        say("Step 4", f"{nb}-neighbour check: farthest {m['metric_check'][str(nb)]['farthest_m']:.2f} m, "
                      f"vs 16: mean {m['metric_check'][str(nb)]['mean_vs16_pct']:+.2f} %, "
                      f"range {m['metric_check'][str(nb)]['min_vs16_pct']:+.2f} to "
                      f"{m['metric_check'][str(nb)]['max_vs16_pct']:+.2f} % (cells over 5 m)")
    m["field"] = read_against_limit(D, grid["walk"])                          # Step 5
    m["catchment_m2"] = {e: float((serve == e).sum() * CELL ** 2) for e in src}
    probe = to_cell(grid, *PROBE_XY)                                          # Step 6
    rr, cc = np.nonzero(f)
    k = np.argmin((rr - probe[0]) ** 2 + (cc - probe[1]) ** 2)
    probe = (int(rr[k]), int(cc[k]))
    path, length = descend(D, grid["walk"], probe, leave)
    pick = np.random.default_rng(0).choice(len(rr), 200, replace=False)
    err = max(abs(descend(D, grid["walk"], (rr[i], cc[i]), leave)[1] - D[rr[i], cc[i]]) for i in pick)
    say("Step 6", f"route from {to_xy(grid, probe)[0].round(2)}: {len(path)} cells, {length:.2f} m "
                  f"(field says {D[probe]:.2f} m), exit {serve[probe]}; "
                  f"max |route - field| over 200 random cells {err:.1e} m")
    m["probe"] = dict(xy=to_xy(grid, probe)[0].round(2).tolist(), cells=len(path), route_m=length,
                      field_m=float(D[probe]), exit=serve[probe], check_err_m=float(err))
    rooms, room_sizes, kind, doors = split_rooms(grid, src)                        # Step 7
    routes = route_every_room(D, grid["walk"], rooms, serve, grid, leave)
    ok = [r for r in routes if r["reachable"]]
    worst = max(ok, key=lambda r: r["dist_m"])
    offices = [r for r in ok if kind[r["room"]] == "room"]
    say("Step 7", f"{len(routes)} spaces ({kind.count('circulation')} circulation), {len(ok)} routed, worst: "
                  f"space {worst['room']} at {worst['worst_xy']}, {worst['dist_m']:.1f} m to exit {worst['exit']} "
                  f"({worst['time_s']:.0f} s); median space worst {np.median([r['dist_m'] for r in ok]):.1f} m")
    m["rooms"] = dict(n=len(routes), routed=len(ok), circulation=kind.count("circulation"),
                      largest_m2=room_sizes[0], sizes_m2=room_sizes, kinds=kind[1:], doors=doors[1:].tolist(),
                      worst=dict(room=worst["room"], kind=kind[worst["room"]], xy=worst["worst_xy"],
                                 dist_m=worst["dist_m"], exit=worst["exit"], time_s=worst["time_s"]),
                      worst_office=max((r["dist_m"] for r in offices), default=0.0),
                      exits_used=sorted({r["exit"] for r in ok}),
                      median_worst_m=float(np.median([r["dist_m"] for r in ok])),
                      route_cells=int(sum(len(r["path"]) for r in ok)),
                      route_check_m=float(max(abs(r["route_m"] - r["dist_m"]) for r in ok)))
    depth, regions, share, nb = dead_ends(grid, G, idx, rooms, kind, D, src, list(src))  # Step 8
    pix, S, _ = skeleton_frame(grid, G, idx)
    corr = corridor_dead_ends(regions)
    say("Step 8", f"centre line {len(pix):,} pixels, {S.number_of_edges():,} links, {nb:,} bridges; "
                  f"one-way floor {100*share:.0f} % in {len(regions)} pieces deeper than {MIN_DEAD_END_M} m, "
                  f"longest common path {regions[0]['depth_m']:.1f} m at {regions[0]['tip_xy']}; "
                  f"dead-end corridors {len(corr)}, longest "
                  f"{(corr[0]['corridor_m'] if corr else 0):.1f} m, over {DEAD_END_M} m: "
                  f"{sum(r['corridor_m'] > DEAD_END_M for r in corr)}")
    m["dead_ends"] = dict(single_path_share=share, bridges=nb, skeleton_px=len(pix), n_one_way=len(regions),
                          longest_common_path_m=regions[0]["depth_m"] if regions else 0.0,
                          common_path_over_limit=int(sum(r["depth_m"] > COMMON_PATH_M for r in regions)),
                          corridor=[dict(length_m=r["corridor_m"], tip_xy=r["corridor_tip_xy"],
                                         mouth_xy=r["mouth_xy"], room=r["room"],
                                         walk_width_m=r["corridor_area_m2"] / r["corridor_m"]) for r in corr],
                          bays=[dict(length_m=r["corridor_m"], tip_xy=r["corridor_tip_xy"], room=r["room"],
                                     walk_width_m=r["corridor_area_m2"] / r["corridor_m"])
                                for r in regions if r["corridor_m"] >= MIN_CORRIDOR_M and r not in corr],
                          longest_one_way_circulation_m=max([r["corridor_m"] for r in regions], default=0.0),
                          corridor_over_limit=int(sum(r["corridor_m"] > DEAD_END_M for r in corr)),
                          regions=regions)
    m["total_core_s"] = time.perf_counter() - t0
    state = dict(grid=grid, src=src, G=G, Gc=Gc, idx=idx, exit_node=exit_node, D=D, serve=serve, leave=leave,
                 rooms=rooms, kind=kind, routes=routes, depth=depth, regions=regions, probe=probe, path=path)
    if not sweep:
        return m, state
    m["sweep"] = block_each_exit(grid, G, idx, src, rooms, kind, exit_node, routes)   # Step 9
    m["cascade"] = block_worst_first(grid, G, idx, src, rooms, kind, exit_node)
    Db, serve_b, _, leave_b = distance_field(grid, G, idx, src, [x for x in src if x != m["probe"]["exit"]],
                                             exit_node)
    pb, lb = descend(Db, grid["walk"], probe, leave_b)
    m["probe_blocked"] = dict(blocked=m["probe"]["exit"], route_m=lb, exit=serve_b[probe])
    say("Step 9", f"the clicked point with exit {m['probe']['exit']} blocked: {lb:.2f} m to exit {serve_b[probe]}")
    row, *_ = what_if(grid, G, idx, src, rooms, kind, ["A"], exit_node)
    m["only_A"] = row
    say("Step 9", f"only the south entrance A open: farthest {row['farthest_m']:.1f} m, mean {row['mean_m']:.1f} m, "
                  f"over {LIMIT_M} m: {100*row['share_over_limit']:.1f} %, longest one-way circulation "
                  f"{row['longest_one_way_circulation_m']:.1f} m")
    state.update(probe_blocked_path=pb)
    if write:
        os.makedirs(OUT, exist_ok=True)
        np.save(os.path.join(OUT, "probe_blocked_path.npy"), np.array(pb))
        m["export"] = export(grid, xyz, rgb, D, depth, routes, regions)      # Step 10
        np.savez_compressed(os.path.join(OUT, "figure_arrays.npz"), rooms=rooms, kind=np.array(kind),
                            serve=serve.astype(str), probe_path=np.array(path))
        with open(os.path.join(OUT, "routes_full.json"), "w", encoding="utf-8") as fh:
            json.dump([{**r, "path": [list(map(int, p)) for p in r.get("path", [])]} for r in routes], fh)
    return m, state


def main():
    t0 = time.perf_counter()
    m = {}
    xyz, rgb, role, floor_z, counts = load_floor()                            # Step 1
    m["points"], m["roles"] = len(xyz), counts
    m, _ = solve(xyz, rgb, role, floor_z, m)
    m["total_s"] = time.perf_counter() - t0
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, default=float)
    say("Done", f"whole run {m['total_s']:.1f} s")


if __name__ == "__main__":
    main()
