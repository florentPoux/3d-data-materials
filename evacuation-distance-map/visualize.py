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
"""Stepped 3D replay of the evacuation pipeline in Open3D. Close each window to advance.

Run after evacuation_distance.py (run.bat does both). Set HEADLESS=1 to build every scene
without opening a window (a smoke test).
"""
import os

import numpy as np
import open3d as o3d
from matplotlib import colormaps

import evacuation_distance as ev

HEADLESS = os.environ.get("HEADLESS") == "1"
ROLE_RGB = {"floor": (0.72, 0.80, 0.90), "wall": (0.55, 0.57, 0.60), "door": (0.95, 0.55, 0.10),
            "obstacle": (0.55, 0.40, 0.75)}


def cloud(xyz, rgb):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    p.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    return p


def grid_cloud(grid, mask, rgb, z):
    rc = np.argwhere(mask)
    xy = ev.to_xy(grid, rc)
    xyz = np.c_[xy, np.full(len(xy), z)]
    col = np.broadcast_to(rgb, (len(xy), 3)) if np.ndim(rgb) == 1 else rgb
    return cloud(xyz, np.asarray(col, float))


def polyline(grid, path, z, rgb):
    xy = ev.to_xy(grid, np.array(path))
    pts = np.c_[xy, np.full(len(xy), z)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                              o3d.utility.Vector2iVector(np.c_[np.arange(len(pts) - 1), np.arange(1, len(pts))]))
    ls.paint_uniform_color(rgb)
    return ls


def field_colors(D, vmax):
    cm = colormaps["viridis_r"]
    return cm(np.clip(D / vmax, 0, 1))[:, :3]


def show(title, geoms):
    print(f"-> {title}")
    if HEADLESS:
        return
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1400, height=900)


def main():
    xyz, rgb, role, floor_z, _ = ev.load_floor()
    sub = np.random.default_rng(0).choice(len(xyz), min(len(xyz), 600_000), replace=False)
    base_rgb = np.array([ROLE_RGB[r] for r in role[sub]])
    show("Step 1 - every point has a role (floor, wall, door, in the way)", [cloud(xyz[sub], base_rgb)])

    grid = ev.build_walkable(xyz, role, floor_z)
    z0 = floor_z + 0.02
    faint = cloud(xyz[sub], 0.35 + 0.6 * base_rgb)
    walk = grid_cloud(grid, grid["walk"], np.array([0.20, 0.65, 0.35]), z0)
    show("Step 2 - the walkable grid (green): where a body's centre fits", [faint, walk])

    src = ev.place_exits(grid)
    marks = []
    for name, (x, y) in ev.EXITS.items():
        s = o3d.geometry.TriangleMesh.create_sphere(0.35)
        s.translate((x, y, floor_z + 0.4))
        s.paint_uniform_color((0.10, 0.70, 0.30))
        marks.append(s)
    show("Step 3 - the exits, the sources of the distance field", [faint, walk] + marks)

    Gc, idx = ev.walking_graph(grid["walk"])
    G, node = ev.attach_exits(Gc, idx, src)
    D, serve, _, leave = ev.distance_field(grid, G, idx, src, None, node)
    fin = np.isfinite(D)
    field = grid_cloud(grid, fin, field_colors(D[fin], 30.0), z0)
    show("Step 4 - walking distance to the nearest exit (yellow near, purple far)", [faint, field] + marks)

    band = grid_cloud(grid, fin, np.where((D[fin] > 15)[:, None], [0.85, 0.35, 0.20], [0.80, 0.85, 0.80]), z0)
    show("Step 5 - cells farther than 15 m from any exit (red)", [faint, band] + marks)

    probe = ev.to_cell(grid, *ev.PROBE_XY)
    rr, cc = np.nonzero(fin)
    k = np.argmin((rr - probe[0]) ** 2 + (cc - probe[1]) ** 2)
    path, _ = ev.descend(D, grid["walk"], (int(rr[k]), int(cc[k])), leave)
    show("Step 6 - one escape route, by descending the field", [faint, field, polyline(grid, path, z0 + 0.05, (0.9, 0.1, 0.1))] + marks)

    rooms, _, kind, _ = ev.split_rooms(grid, src)
    routes = ev.route_every_room(D, grid["walk"], rooms, serve, grid, leave)
    lines = [polyline(grid, r["path"], z0 + 0.05, (0.85, 0.10, 0.10)) for r in routes if r.get("reachable")]
    show("Step 7 - the way out of every space", [faint, field] + lines + marks)

    depth, _, _, _ = ev.dead_ends(grid, G, idx, rooms, kind, D, src, list(src))
    dm = np.isfinite(depth)
    dcol = colormaps["magma_r"](np.clip(depth[dm] / 10.0, 0, 1))[:, :3]
    show("Step 8 - one-way floor: the walk back to where you can choose (dark = long)",
         [faint, grid_cloud(grid, dm, dcol, z0)] + marks)

    Da, *_ = ev.distance_field(grid, G, idx, src, ["A"], node)
    fa = np.isfinite(Da)
    keep = [m for m, n in zip(marks, ev.EXITS) if n == "A"]
    show("Step 9 - what if only the south entrance A is open", [faint, grid_cloud(grid, fa, field_colors(Da[fa], 60.0), z0)] + keep)

    out_ply = os.path.join(ev.OUT, "area1_evac_distance.ply")
    if os.path.exists(out_ply):
        pc = o3d.io.read_point_cloud(out_ply)
        from plyfile import PlyData
        d = PlyData.read(out_ply)["vertex"]["evac_distance_m"]
        pc.colors = o3d.utility.Vector3dVector(np.where((d < 0)[:, None], 0.8, field_colors(d, 30.0)))
        show("Step 10 - the exported cloud, every point carrying evac_distance_m", [pc])


if __name__ == "__main__":
    main()
