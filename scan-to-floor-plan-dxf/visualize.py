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
# Stepped 3D replay of walls_to_vectors.py: one Open3D window per stage.
# Close the window (or press Q) to move to the next stage.
#
# Headless check (builds every geometry, opens nothing):
#   python visualize.py --headless      or   set P24_HEADLESS=1
#
import os
import sys

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "out", "stages.npz"), allow_pickle=True)
HEADLESS = "--headless" in sys.argv or os.environ.get("P24_HEADLESS") == "1"

O, C, FZ = S["origin"], float(S["cell"]), float(S["floor_z"])
CLASS_COL = {0: (0.79, 0.82, 0.85), 1: (0.85, 0.81, 0.75), 2: (0.36, 0.61, 0.84),
             3: (0.91, 0.66, 0.0), 4: (0.51, 0.31, 0.87), 5: (0.18, 0.62, 0.36),
             6: (0.82, 0.14, 0.18)}


def cloud(no_ceiling=True, dim=False):
    P, K = S["cloud"], S["cloud_cls"]
    keep = K != 0 if no_ceiling else np.ones(len(K), bool)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[keep].astype(float)))
    col = np.array([CLASS_COL[int(k)] for k in K[keep]])
    if dim:
        col = 0.35 * col + 0.65
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc


def cells(mask, z, color):
    r, c = np.nonzero(mask)
    P = np.stack([O[0] + (c + 0.5) * C, O[1] + (r + 0.5) * C, np.full(len(r), z)], axis=1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.paint_uniform_color(color)
    return pc


def lines(segs, z, color):
    segs = np.asarray(segs)[:, :4]
    pts = np.concatenate([np.c_[segs[:, 0:2], np.full(len(segs), z)],
                          np.c_[segs[:, 2:4], np.full(len(segs), z)]])
    n = len(segs)
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                              o3d.utility.Vector2iVector([[i, i + n] for i in range(n)]))
    ls.paint_uniform_color(color)
    return ls


def wall_boxes(walls, height=2.3):
    out = []
    for w in walls:
        x0, y0, x1, y1, seen = w
        pts = [[x0, y0, FZ], [x1, y1, FZ], [x1, y1, FZ + height], [x0, y0, FZ + height]]
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                  o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]))
        ls.paint_uniform_color((0.88, 0.48, 0.0) if seen < 0.5 else (0.0, 0.47, 0.83))
        out.append(ls)
    return out


z = FZ + 0.02
STAGES = [
    ("1. The classified scan (ceiling hidden)", lambda: [cloud()]),
    ("1. Wall-evidence grid at floor height", lambda: [cloud(dim=True), cells(S["grid"], z, (0.1, 0.1, 0.1))]),
    ("2. Closed grid (gold: filled cells)", lambda: [cells(S["grid"], z, (0.1, 0.1, 0.1)),
                                                     cells(S["closed"] & ~S["grid"], z + 0.01, (0.91, 0.66, 0.0))]),
    ("3. Skeleton (centre line)", lambda: [cells(S["closed"], z, (0.8, 0.84, 0.88)),
                                           cells(S["skel"], z + 0.01, (0.76, 0.09, 0.36))]),
    ("5. Straight runs", lambda: [cells(S["skel"], z, (0.7, 0.7, 0.7)), lines(S["runs"], z + 0.01, (0.0, 0.47, 0.83))]),
    ("6. Merged walls", lambda: [cells(S["closed"], z, (0.8, 0.84, 0.88)), lines(S["merged"], z + 0.01, (0.18, 0.62, 0.36))]),
    ("7-8. Squared, snapped onto the points, joined, measured (orange: assumed)",
     lambda: [cloud(dim=True), lines(S["walls"], z, (0.1, 0.1, 0.1))] + wall_boxes(S["walls"])),
]

for name, build in STAGES:
    geoms = build()
    print(f"{name}: {len(geoms)} geometries")
    if not HEADLESS:
        o3d.visualization.draw_geometries(geoms, window_name=name, width=1280, height=820)
print("replay done" + (" (headless)" if HEADLESS else ""))
