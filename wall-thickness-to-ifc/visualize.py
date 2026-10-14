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
# Stepped 3D replay of wall_thickness_to_ifc.py. Run it AFTER the pipeline (run.bat does both).
# One Open3D window per stage; close the window (or press Q) to move to the next one:
#
#   1  the classified scan, in its class colours
#   2  the scan levelled between its floor and ceiling, coloured by height
#   3  the wall points, coloured by the direction their normal votes for
#   5  the face runs found as density peaks, one colour per face
#   6  the pairs: accepted walls (green) and rejected pairs (amber: not seen from both sides, red: not empty)
#   7  the two faces of each measured wall, in its matched cells (blue face A, gold face B)
#   9  the IFC walls read back from the file, over the scan
#
# Headless check: set VIZ_SMOKE=1 (or pass --smoke) to build every stage and print a summary
# without opening any window.
#
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SMOKE = "--smoke" in sys.argv or os.environ.get("VIZ_SMOKE") == "1"

CLASS_COL = {"ceiling": "#C9CED6", "floor": "#D9C7A7", "wall": "#6E9BC9", "beam": "#8C6BB1",
             "column": "#B07AA1", "window": "#57C4E5", "door": "#E07B39", "table": "#9C755F",
             "chair": "#59A14F", "bookcase": "#EDC948", "board": "#F28E2B", "clutter": "#BAB0AC"}
FACE_COLS = ["#0078D4", "#E8A800", "#2E9E5B", "#D64545", "#8C6BB1", "#17BECF", "#E07B39",
             "#59A14F", "#B07AA1", "#9C755F"]


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def pcd(points, colors):
    import open3d as o3d
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    c = np.asarray(colors, float)
    if c.ndim == 1:
        c = np.tile(c, (len(points), 1))
    p.colors = o3d.utility.Vector3dVector(c)
    return p


def show(title, geoms):
    n = sum(len(g.points) if hasattr(g, "points") else len(g.vertices) for g in geoms)
    print(f"[view] {title}: {len(geoms)} geometries, {n:,} points/vertices")
    if SMOKE:
        return
    import open3d as o3d
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1400, height=900)


def main():
    if not os.path.exists(os.path.join(OUT, "state.npz")):
        sys.exit("Run wall_thickness_to_ifc.py first (run.bat does it for you).")
    S = np.load(os.path.join(OUT, "state.npz"))
    R = json.load(open(os.path.join(OUT, "result.json"), encoding="utf-8"))
    M = np.load(os.path.join(OUT, "ifc_meshes.npz"))
    names = {int(k): v for k, v in R["names"].items()}
    lev, cls, wm = S["lev"].astype(float), S["cls"], S["wall_mask"]
    wall = lev[wm]
    ceiling = [k for k, v in names.items() if v == "ceiling"][0]
    keep = cls != ceiling            # the ceiling hides everything from above

    import wall_thickness_to_ifc as P
    xyz, _, _, _, _ = P.load_scan(P.DATA, P.SIDECAR)
    cc = np.array([hex2rgb(CLASS_COL.get(names.get(int(c), ""), "#999999")) for c in cls])
    show("Step 1: the classified scan (ceiling hidden)", [pcd(xyz[keep], cc[keep])])

    z = lev[keep, 2]
    t = np.clip(z / 3.1, 0, 1)[:, None]
    show("Step 2: levelled between floor and ceiling, coloured by height",
         [pcd(lev[keep], (1 - t) * np.array(hex2rgb("#D9C7A7")) + t * np.array(hex2rgb("#005a9e")))])

    col = np.tile(hex2rgb("#C8CDD3"), (len(wall), 1))
    for F, h in zip(R["families"], ("#E8A800", "#0078D4")):
        col[np.asarray(F["member_idx"], int)] = hex2rgb(h)
    show("Step 3: wall points by direction (gold A, blue B, grey untrusted)", [pcd(wall, col)])

    col = np.tile(hex2rgb("#DADFE4"), (len(wall), 1))
    k = 0
    for F in R["families"]:
        mem = np.asarray(F["member_idx"], int)
        for fc in F["faces"]:
            col[mem[np.asarray(fc["idx"], int)]] = hex2rgb(FACE_COLS[k % len(FACE_COLS)])
            k += 1
    show(f"Steps 4-5: {k} face runs found as density peaks", [pcd(wall, col)])

    col = np.tile(hex2rgb("#DADFE4"), (len(wall), 1))
    for F in R["families"]:
        mem = np.asarray(F["member_idx"], int)
        for c in F["cands"]:
            h = {"WALL": "#2E9E5B", "not empty between": "#D64545"}.get(c["verdict"], "#E07B00")
            for i in (c["i"], c["j"]):
                col[mem[np.asarray(F["faces"][i]["idx"], int)]] = hex2rgb(h)
    show("Step 6: pairs (green wall, amber not seen from both sides, red not empty)", [pcd(wall, col)])

    geoms = [pcd(lev[keep], np.tile(hex2rgb("#E3E6EA"), (keep.sum(), 1)))]
    for w in R["walls"]:
        geoms.append(pcd(wall[np.asarray(w["idx_a"], int)], hex2rgb("#0078D4")))
        geoms.append(pcd(wall[np.asarray(w["idx_b"], int)], hex2rgb("#E8A800")))
    show("Step 7: the two faces of each measured wall", geoms)

    import open3d as o3d
    geoms = [pcd(lev[keep], np.tile(hex2rgb("#C8CDD3"), (keep.sum(), 1)))]
    for i, h in zip(range(len(R["walls"])), ("#2E9E5B", "#E07B00", "#0078D4", "#8C6BB1")):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(M[f"v{i}"]),
                                      o3d.utility.Vector3iVector(M[f"f{i}"].astype(np.int32)))
        m.compute_vertex_normals()
        m.paint_uniform_color(hex2rgb(h))
        geoms.append(m)
    show("Steps 8-9: the IFC walls, read back from the file", geoms)
    print("Replay done.")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
