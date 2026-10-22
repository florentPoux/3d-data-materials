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
Stepped 3D replay of the accessibility audit (run accessibility_audit.py first).

One Open3D window per step: close it (or press Q) to advance.
Set VIZ_SMOKE=1 to build every step's geometry without opening a window (a headless check).
"""
import json
import os
from pathlib import Path

import numpy as np

import accessibility_audit as aa

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("VIZ_SMOKE") == "1"

CLASS_RGB = {1: (0.79, 0.84, 0.89), 2: (0.54, 0.59, 0.64), 6: (0.18, 0.62, 0.36), 7: (0.70, 0.53, 0.40),
             8: (0.88, 0.64, 0.35), 10: (0.55, 0.34, 0.29), 12: (0.85, 0.42, 0.66), 13: (0.69, 0.72, 0.75)}
GREEN, AMBER, RED, GREY = (0.12, 0.54, 0.30), (0.95, 0.62, 0.04), (0.78, 0.16, 0.16), (0.77, 0.80, 0.82)


def load():
    v = aa.read_ply(aa.DATA)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(v), min(len(v), 1_500_000), replace=False)   # enough for a smooth view
    xyz = np.column_stack([v["x"][keep], v["y"][keep], v["z"][keep]]).astype(np.float64)
    rgb = np.column_stack([v["red"][keep], v["green"][keep], v["blue"][keep]]) / 255.0
    cls = v["classification"][keep].astype(int)
    G = dict(np.load(HERE / "out" / "audit_grids.npz"))
    S = json.loads((HERE / "out" / "summary.json").read_text(encoding="utf-8"))
    return xyz, rgb, cls, G, S


def grid_lookup(xyz, G):
    lo, cell = G["lo"], float(G["cell"])
    ij = ((xyz[:, :2] - lo) / cell).astype(int)
    i = np.clip(ij[:, 0], 0, G["H"].shape[0] - 1)
    j = np.clip(ij[:, 1], 0, G["H"].shape[1] - 1)
    return i, j


def ramp(values, lo, hi, c0, c1):
    t = np.clip((values - lo) / (hi - lo), 0, 1)[:, None]
    return (1 - t) * np.array(c0) + t * np.array(c1)


def stages(xyz, rgb, cls, G, S):
    i, j = grid_lookup(xyz, G)
    low = (xyz[:, 2] - G["Hnear"][i, j]) < 1.4          # walls cut at 1.40 m so rooms read from above
    fl = cls == 1
    base = np.array([CLASS_RGB.get(c, (0.7, 0.7, 0.7)) for c in range(14)])[cls]
    out = []

    out.append(("1. The scan in its true colours", xyz, rgb))
    out.append(("1b. The 13 classes, walls cut at 1.40 m", xyz[low], base[low]))

    c = base.copy()
    h = G["Hnear"][i, j]
    c[fl] = ramp(h[fl], -1.0, 0.9, (0.05, 0.20, 0.45), (0.80, 0.90, 1.00))
    c[G["jump"][i, j] & fl] = RED
    out.append(("2. The walkable grid: floor by height, jumps in red", xyz[low], c[low]))

    c = base.copy()
    W = G["W"][i, j]
    c[fl] = np.where((W[fl] >= 0.90)[:, None], ramp(W[fl], 0.9, 1.6, (0.66, 0.86, 0.72), GREEN),
                     ramp(W[fl], 0.0, 0.9, (0.48, 0.06, 0.12), AMBER))
    out.append(("3. Clear width: red narrow, green 0.90 m and more", xyz[low], c[low]))

    c = base.copy()
    c[fl] = GREY
    c[fl & G["free_chair"][i, j] & (G["W"][i, j] > 0)] = (0.66, 0.86, 0.72)
    c[fl & G["C"][i, j]] = GREEN
    out.append(("4. Where the wheelchair's centre can be", xyz[low], c[low]))

    c = base.copy()
    widths = dict(zip(G["door_ids"].tolist(), G["door_widths"].tolist()))
    ok = dict(zip(G["door_ids"].tolist(), G["door_measurable"].tolist()))
    dm = G["door_map"][i, j]
    for k, w in widths.items():
        c[(dm == k) & (cls == 6)] = GREY if not ok[k] else (RED if w < 0.80 else GREEN)
    out.append(("5. Doors: red under 0.80 m, green wide enough", xyz[low], c[low]))

    c = base.copy()
    c[fl & G["turn"][i, j]] = (0.21, 0.72, 0.91)
    out.append(("6. Turning floor: where a 1.50 m circle fits", xyz[low], c[low]))

    c = base.copy()
    c[fl & G["free_foot"][i, j]] = (0.89, 0.60, 0.60)
    c[fl & G["cov_f"][i, j]] = AMBER
    c[fl & G["cov_c"][i, j]] = GREEN
    out.append(("7. Reach: green wheelchair, amber on foot only", xyz[low], c[low]))

    out.append(("8. The issue list, numbered (spheres)", xyz[low], c[low]))
    return out, S["issue_list"], G


def show(title, pts, cols, issues=None, G=None):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    geoms = [pc]
    if issues is not None:
        sev = {"high": RED, "medium": AMBER, "low": GREY}
        for s in issues:
            x, y = s["pos"]
            i, j = grid_lookup(np.array([[x, y, 0.0]]), G)
            z = float(G["Hnear"][i[0], j[0]]) + 1.6
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.35)
            sp.translate((x, y, z))
            sp.paint_uniform_color(sev[s["severity"]])
            sp.compute_vertex_normals()
            geoms.append(sp)
    if SMOKE:
        print(f"[smoke] {title}: {len(pts):,} points, {len(geoms) - 1} markers")
        return
    print(f"-> {title}  (close the window to continue)")
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1400, height=900)


def main():
    xyz, rgb, cls, G, S = load()
    steps, issues, G = stages(xyz, rgb, cls, G, S)
    for k, (title, pts, cols) in enumerate(steps):
        last = k == len(steps) - 1
        show(title, pts, cols, issues if last else None, G)
    png = HERE / "out" / "issues.csv"
    print(f"Issue list: {png}")


if __name__ == "__main__":
    main()
