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
"""Build this article's input scan from your own copy of S3DIS.

S3DIS, the Stanford Large-Scale 3D Indoor Spaces dataset (Armeni et al., CVPR 2016),
is free for research and teaching, but its terms say it is not redistributed in any
form. So this kit cannot hand you the scan. This script rebuilds it from the official
download instead, with the same recipe the article's copy was made with:

    office        articles 02 and 05   data/s3dis_office.ply (+ its .n3d.json class names)
    area1_evac    article 04           data/s3dis_area1_open_5cm.ply + data/classes.json
    area2_access  article 03           data/s3dis_area2_access.ply

1. Request S3DIS through its registration form, which carries the terms of use:
   https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform
   The confirmation page gives the download link. Take
   Stanford3dDataset_v1.2_Aligned_Version.zip (4.1 GB) and unzip it anywhere.
2. Run, from this folder:

       python prepare_s3dis.py --s3dis "path/to/Stanford3dDataset_v1.2_Aligned_Version"

   or set the S3DIS_ROOT environment variable and run it with no argument.

It reads the per-object files in every room's Annotations folder (the file name is the
class), so it only needs the area it uses: Area_1 for 02, 04 and 05, Area_2 for 03.
Reading a whole area takes a few minutes and about 4 GB of RAM.

At the end it prints the point count and a fingerprint of the file it wrote, next to
the ones the article ran on. The article's copy came from a 2020 conversion of S3DIS
to LAS; fed the points of that file in its order, this script rebuilds the article's
input bit for bit. Read from Stanford's text files, the order differs, so expect the
same point count with a handful of points picked differently inside their cells, and
the article's numbers to within a small margin. Which of Stanford's two v1.2 archives
that conversion started from was not recorded: start with the Aligned version, and if
the point count is off, try Stanford3dDataset_v1.2.zip instead.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# The recipe this copy of the script runs by default. The other two are here too:
# pass --recipe to build them.
RECIPE = "office"

# S3DIS names each annotation file after its class ("chair_3.txt"). The article files
# use the 13 classes of the S3DIS benchmark, plus stairs for the accessibility audit.
# Some rooms spell the stairs file "staris"; both are read as stairs.
CLASS14 = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table",
           "chair", "sofa", "bookcase", "board", "stairs", "clutter"]
CODE14 = {name: i for i, name in enumerate(CLASS14)}
CODE14["staris"] = CODE14["stairs"]
CLASS13 = CLASS14[:12] + ["clutter"]

# What each recipe should give, measured on the files the articles ran on.
# The fingerprint is a SHA-256 of x, y, z and the class of every point, in file order.
EXPECTED = {
    "office": {"points": 191353,
               "sha256": "2786fba63fb7706c23fffa337b6364a34306b685316e0ef2b3fae9315708d3c1"},
    "area1_evac": {"points": 1378028,
                   "sha256": "30f9657496829cff72158f69dfaa1430289c588d93f0cf28dcda24a742ffce3a"},
    "area2_access": {"points": 4031139,
                     "sha256": "ad3407e1eced5adc37e3ba303d76924855a95022a7ba7a75495b497c22b73158"},
}

# Area 1, as the benchmark grid: one point per 4 cm cell (see grid_majority).
GRID = 0.04
# The office used by articles 02 and 05: the east edge of Area 1, in the area's metres.
OFFICE_X = (-5.1215, 2.027)
OFFICE_Y = (18.302, 27.3)
# Article 04 thins the open floor once more, to one point per 5 cm voxel (seed 0).
EVAC_VOXEL = 0.05
EVAC_SEED = 0
# Article 03: the north half of Area 2 [x0, y0, x1, y1], no ceiling, the floor and the
# stairs at 2 cm (heights are measured on them), every other class at 3 cm.
ACCESS_CROP = (-8.3, -10.2, 22.8, 25.0)
ACCESS_FINE, ACCESS_COARSE = 0.02, 0.03


# ----------------------------------------------------------------------------- read
def find_area(root, area):
    """The Area_N folder, whether root is the unzipped folder or the folder above it."""
    for cand in (os.path.join(root, area), *glob.glob(os.path.join(root, "*", area))):
        if os.path.isdir(cand):
            return cand
    sys.exit(f"No {area} folder under {root}. Point --s3dis at the unzipped "
             f"Stanford3dDataset_v1.2_Aligned_Version folder.")


def read_txt(path):
    """One annotation file: x y z r g b per line, as a float64 array."""
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, usecols=range(6),
                           dtype=np.float64, engine="c").to_numpy()
    except (ValueError, pd.errors.ParserError):
        # A few S3DIS files carry a stray non-numeric byte in one line. Drop it.
        rows = []
        with open(path, "rb") as f:
            for line in f:
                clean = re.sub(rb"[^0-9eE+\-. \t]", b"", line).split()
                if len(clean) >= 6:
                    rows.append([float(v) for v in clean[:6]])
        print(f"    cleaned {os.path.basename(path)}: {len(rows):,} lines kept")
        return np.asarray(rows, dtype=np.float64)


def read_area(area_dir, xyz_dtype):
    """Every annotated point of one area: xyz, rgb (uint8) and a 14-class code.

    Rooms and files are read in sorted order, so two runs read the same sequence."""
    rooms = sorted(r for r in os.listdir(area_dir)
                   if os.path.isdir(os.path.join(area_dir, r, "Annotations")))
    if not rooms:
        sys.exit(f"No room with an Annotations folder in {area_dir}.")
    xyz, rgb, cls = [], [], []
    unknown = set()
    t0 = time.perf_counter()
    for i, room in enumerate(rooms):
        for path in sorted(glob.glob(os.path.join(area_dir, room, "Annotations", "*.txt"))):
            # "chair_3.txt" is a chair; a few files carry no number ("floor.txt").
            name = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[0].lower()
            code = CODE14.get(name)
            if code is None:
                unknown.add(name)
                code = CODE14["clutter"]
            a = read_txt(path)
            # The article's copy went through a LAS file first, which stores every
            # coordinate as a whole number of 0.01 mm. S3DIS coordinates are in mm, and
            # a point that sits exactly on a voxel boundary lands in one voxel or the
            # next depending on the last bit, so snap to the same grid.
            xyz.append((np.rint(a[:, :3] / 1e-5) * 1e-5).astype(xyz_dtype))
            rgb.append(np.clip(a[:, 3:6], 0, 255).astype(np.uint8))
            cls.append(np.full(len(a), code, dtype=np.uint8))
        print(f"  {os.path.basename(area_dir)}: room {i + 1}/{len(rooms)} {room}", end="\r")
    print()
    if unknown:
        print(f"  read as clutter (unknown class names): {sorted(unknown)}")
    xyz, rgb, cls = np.concatenate(xyz), np.concatenate(rgb), np.concatenate(cls)
    print(f"  {len(xyz):,} points in {len(rooms)} rooms, read in {time.perf_counter() - t0:.0f} s")
    return xyz, rgb, cls


# ---------------------------------------------------------------------- the recipes
def grid_majority(xyz, labels, voxel):
    """One real point per occupied cell of a grid anchored at the area's corner.

    The point kept is the one nearest the centroid of its cell, and it carries the
    label the majority of the cell's points vote for (a tie goes to the lower class
    id). Nothing here depends on the order the points were read in, which is what
    lets your copy match the article's point for point. Returns (indices, labels).
    """
    ijk = np.floor((xyz - xyz.min(axis=0)) / np.float32(voxel)).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    del ijk
    cells, inverse = np.unique(flat, return_inverse=True)
    del flat
    n = len(cells)
    counts = np.bincount(inverse, minlength=n).astype(np.float64)
    centroid = np.empty((n, 3))
    for a in range(3):
        centroid[:, a] = np.bincount(inverse, weights=xyz[:, a], minlength=n) / counts
    d2 = ((xyz - centroid[inverse]) ** 2).sum(axis=1)
    order = np.lexsort((d2, inverse))
    keep = order[np.searchsorted(inverse[order], np.arange(n))]
    del d2, order
    k = int(labels.max()) + 1
    votes = np.bincount(inverse * k + labels, minlength=n * k).reshape(n, k)
    return keep, votes.argmax(axis=1)


def area1_grid(root):
    """Area 1 on the 4 cm benchmark grid, 13 classes (stairs folded into clutter)."""
    xyz, rgb, cls = read_area(find_area(root, "Area_1"), np.float32)
    labels = np.where(cls >= CODE14["stairs"], CLASS13.index("clutter"), cls).astype(np.int64)
    del cls
    keep, voted = grid_majority(xyz, labels, GRID)
    print(f"  4 cm grid: {len(keep):,} points (the article's copy: 2,991,588)")
    # The article's copy went through a 16-bit colour field and back, which left every
    # colour channel one step darker (255 became 254). Kept so the file matches byte
    # for byte; the article's code draws colour and never measures it.
    col = np.clip((rgb[keep].astype(np.float32) * 256.0) / 257.0, 0, 255).astype(np.uint8)
    return xyz[keep], col, voted.astype(np.int64)


def make_office(root):
    xyz, rgb, cls = area1_grid(root)
    x, y = xyz[:, 0], xyz[:, 1]
    m = ((x >= np.float32(OFFICE_X[0])) & (x <= np.float32(OFFICE_X[1]))
         & (y >= np.float32(OFFICE_Y[0])) & (y <= np.float32(OFFICE_Y[1])))
    out = os.path.join(DATA, "s3dis_office.ply")
    arr = write_ply(out, xyz[m], rgb[m], cls[m])
    names = {str(c): CLASS13[c] for c in np.unique(cls[m])}
    with open(out + ".n3d.json", "w", encoding="utf-8") as f:
        json.dump({"version": 1, "source_file": "s3dis_office.ply", "class_labels": names},
                  f, indent=2)
    return out, arr


def make_area1_evac(root):
    xyz, rgb, cls = area1_grid(root)
    m = ~np.isin(cls, [CLASS13.index("ceiling"), CLASS13.index("beam")])
    xyz, rgb, cls = xyz[m], rgb[m], cls[m]
    print(f"  without ceiling and beams: {len(xyz):,} points (the article's copy: 2,313,429)")
    # One point per 5 cm voxel, picked at random inside each voxel with seed 0.
    p = xyz.astype(np.float64)
    order = np.random.default_rng(EVAC_SEED).permutation(len(p))
    key = np.floor((p[order] - p.min(axis=0)) / EVAC_VOXEL).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(order[first])
    out = os.path.join(DATA, "s3dis_area1_open_5cm.ply")
    arr = write_ply(out, xyz[keep], rgb[keep], cls[keep])
    names = {str(c): CLASS13[c] for c in np.unique(cls[keep])}
    with open(os.path.join(DATA, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    return out, arr


def make_area2_access(root):
    xyz, rgb, cls = read_area(find_area(root, "Area_2"), np.float64)
    x0, y0, x1, y1 = ACCESS_CROP
    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
         & (cls != CODE14["ceiling"]))
    xyz, rgb, cls = xyz[m], rgb[m], cls[m]
    fine = (cls == CODE14["floor"]) | (cls == CODE14["stairs"])
    parts = []
    for sel, voxel in ((fine, ACCESS_FINE), (~fine, ACCESS_COARSE)):
        idx = np.nonzero(sel)[0]
        ijk = np.floor(xyz[idx] / voxel).astype(np.int64)
        # One real point per voxel: the first one read, listed in voxel order.
        _, first = np.unique(ijk, axis=0, return_index=True)
        parts.append(idx[first])
    keep = np.concatenate(parts)
    print(f"  crop, no ceiling, 2 cm floor and stairs, 3 cm the rest: {len(keep):,} points")
    out = os.path.join(DATA, "s3dis_area2_access.ply")
    arr = write_ply(out, xyz[keep], rgb[keep], cls[keep])
    return out, arr


# ---------------------------------------------------------------------------- write
def write_ply(path, xyz, rgb, cls):
    """Binary PLY with the fields the article's scripts read."""
    arr = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                    ("classification", "<f4")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["classification"] = cls
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False, byte_order="<").write(path)
    return arr


def fingerprint(arr):
    h = hashlib.sha256()
    for field in ("x", "y", "z", "classification"):
        h.update(np.ascontiguousarray(arr[field]).tobytes())
    return h.hexdigest()


RECIPES = {"office": make_office, "area1_evac": make_area1_evac,
           "area2_access": make_area2_access}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Build this article's input scan from S3DIS.")
    ap.add_argument("--s3dis", default=os.environ.get("S3DIS_ROOT", ""),
                    help="the unzipped Stanford3dDataset_v1.2_Aligned_Version folder")
    ap.add_argument("--recipe", default=RECIPE, choices=sorted(RECIPES))
    args = ap.parse_args()
    if not args.s3dis or not os.path.isdir(args.s3dis):
        sys.exit("Pass --s3dis <folder> (or set S3DIS_ROOT): the unzipped "
                 "Stanford3dDataset_v1.2_Aligned_Version folder from Stanford.")
    t0 = time.perf_counter()
    print(f"Recipe {args.recipe}, from {args.s3dis}")
    out, arr = RECIPES[args.recipe](args.s3dis)
    fp = fingerprint(arr)
    want = EXPECTED[args.recipe]
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f} s")
    print(f"  points       {len(arr):,}   (the article's copy: {want['points']:,})")
    print(f"  fingerprint  {fp}")
    print(f"  article's    {want['sha256']}")
    if fp == want["sha256"]:
        print("Identical to the file the article ran on: every number will match.")
    elif abs(len(arr) - want["points"]) <= max(10, want["points"] // 10000):
        # Within 0.01 %: the 5 cm thinning of article 04 can gain or lose a few points
        # when the 4 cm pick before it differs.
        print("The same cells as the article's file, give or take a few, with some points\n"
              "picked differently inside them: the article's copy read S3DIS in another order,\n"
              "and the pick inside a cell can depend on it. Expect the article's numbers to\n"
              "within a small margin, not always to the last digit.")
    else:
        print("A different point count: your copy of S3DIS differs from the one the article\n"
              "used (another version, or files missing). The pipeline still runs; its numbers\n"
              "will differ.")


if __name__ == "__main__":
    main()
