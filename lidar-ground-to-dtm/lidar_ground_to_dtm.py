# -*- coding: utf-8 -*-
#
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/the-practical-field-guide-to-lidar-point-cloud-processing-f7c3b8ef6d50
# Dataset:  https://learngeodata.eu/materials/lidar-ground-to-dtm/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
#
# ---------------------------------------------------------------------------
# Airborne LiDAR tile -> ground points -> terrain model (DTM), on a CPU.
# ---------------------------------------------------------------------------
#
# The field-guide workflow, end to end:
#   .laz tile  ->  read it with laspy (scale, offset, CRS, classes)
#      ->  voxel downsample + KD-tree queries (Open3D)
#      ->  Cloth Simulation Filter: ground vs everything else
#      ->  score that ground against the survey's own ground class
#      ->  rasterise the ground to a DTM (GeoTIFF + hillshade PNG)
#      ->  height above ground for every point
#      ->  ground points written back out as .laz
#
# Dataset: a 300 x 300 m crop of one IGN LiDAR HD tile in France
# (LHD_FXX_0584_6264, Lambert-93, 3,830,133 points, ~21 MB). IGN classified
# every point, so the tile carries a ground class (2) we can grade our own
# filter against. Licence Ouverte / Open Licence 2.0, source: IGN.
#
# Run:
#   pip install -r requirements.txt
#   python lidar_ground_to_dtm.py
#
import os
import sys
import time

import numpy as np

# Keep prints readable on Windows where the console default is not UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import laspy
import open3d as o3d
import CSF                      # pip install cloth-simulation-filter
from scipy.spatial import cKDTree

# --- configuration ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# The tile. Point this at any .las or .laz of your own.
LAS_FILE = os.path.join(HERE, "lidar_hd_tile.laz")

# Voxel size in metres. The article's 0.1 m barely touches an aerial tile
# (40 points per square metre is already sparser than that grid), so 0.5 m
# is used here: big enough that you see the cloud thin out.
VOXEL = 0.5

# KD-tree questions: k nearest neighbours, and everything inside a radius.
K_NEIGHBOURS = 8
RADIUS = 1.0

# Cloth Simulation Filter, the article's three knobs.
CLOTH_RESOLUTION = 1.0   # cloth grid spacing in metres
RIGIDNESS = 2            # 1 steep terrain, 2 rolling, 3 flat
CLASS_THRESHOLD = 0.45   # cloth-to-point distance (m) still counted as ground

# Output raster cell size in metres, and the IDW neighbourhood.
DTM_CELL = 0.5
IDW_K = 8

ASPRS_GROUND = 2


def _say(step, msg):
    print(f"[{step}] {msg}", flush=True)


# --- Step 1: read the tile -------------------------------------------------
def read_tile(path=LAS_FILE):
    """laspy applies the header scale and offset, so x, y, z are real metres."""
    if not os.path.isfile(path):
        sys.exit(f"Dataset not found: {path}\n"
                 "Download lidar_hd_tile.laz from the kit page and put it next to "
                 "this script, or point LAS_FILE at a .las/.laz of your own.")
    las = laspy.read(path)
    h = las.header
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    cls = np.asarray(las.classification)
    crs = h.parse_crs()
    _say("Step 1", f"{len(xyz):,} points, LAS {h.version}, point format {h.point_format.id}")
    _say("Step 1", f"scale {[float(s) for s in h.scales]}  "
                   f"offset {[round(float(o), 2) for o in h.offsets]}")
    _say("Step 1", f"stored as integers: X[0] = {las.X[0]} -> x[0] = {las.x[0]:.2f} m")
    _say("Step 1", f"CRS {crs.name if crs else 'none in header'}")
    ext = xyz.max(0) - xyz.min(0)
    _say("Step 1", f"extent {ext[0]:.0f} x {ext[1]:.0f} m, z {xyz[:, 2].min():.1f} "
                   f"to {xyz[:, 2].max():.1f} m, {len(xyz) / (ext[0] * ext[1]):.1f} pts/m2")
    u, n = np.unique(cls, return_counts=True)
    _say("Step 1", "classes " + ", ".join(f"{a}: {b:,}" for a, b in zip(u, n)))
    return las, xyz, cls, crs


# --- Step 2: structure it --------------------------------------------------
def downsample_and_index(xyz):
    """The field guide's snippet: Open3D cloud, voxel grid, KD-tree queries."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    down = pcd.voxel_down_sample(voxel_size=VOXEL)
    n0, n1 = len(pcd.points), len(down.points)
    _say("Step 2", f"voxel {VOXEL} m: {n0:,} -> {n1:,} points ({100.0 * n1 / n0:.1f} % kept)")

    t = time.time()
    tree = o3d.geometry.KDTreeFlann(down)
    _say("Step 2", f"KD-tree built over {n1:,} points in {time.time() - t:.2f}s")

    probe = down.points[1000]
    k, knn_idx, knn_d2 = tree.search_knn_vector_3d(probe, K_NEIGHBOURS)
    _say("Step 2", f"{k} nearest neighbours of point 1000, farthest at "
                   f"{np.sqrt(np.max(knn_d2)):.2f} m")
    n, ball_idx, _ = tree.search_radius_vector_3d(probe, RADIUS)
    _say("Step 2", f"{n} points within {RADIUS} m of point 1000")
    return down, tree


# --- Step 3: ground filter -------------------------------------------------
def csf_ground(xyz):
    """Cloth Simulation Filter (Zhang et al. 2016): flip the cloud upside
    down, drop a cloth on it, and call ground whatever the cloth touches.

    The same filter in PDAL, if you have it installed (conda-forge only):
        {"type": "filters.csf", "resolution": 1.0, "rigidness": 2,
         "threshold": 0.45}
    """
    csf = CSF.CSF()
    # Slope post-processing. PDAL's filters.csf turns it on by default; this
    # package does not. On this tile it lifts recall from 0.74 to 0.92: without
    # it the cloth bridges over the steep banks and calls them off-ground.
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = CLOTH_RESOLUTION
    csf.params.rigidness = RIGIDNESS
    csf.params.class_threshold = CLASS_THRESHOLD
    csf.setPointCloud(xyz)
    ground, off = CSF.VecInt(), CSF.VecInt()
    t = time.time()
    csf.do_filtering(ground, off, False)
    is_ground = np.zeros(len(xyz), dtype=bool)
    is_ground[np.asarray(ground, dtype=np.int64)] = True
    _say("Step 3", f"CSF (res {CLOTH_RESOLUTION} m, rigidness {RIGIDNESS}, threshold "
                   f"{CLASS_THRESHOLD} m): {is_ground.sum():,} ground, "
                   f"{(~is_ground).sum():,} off-ground in {time.time() - t:.1f}s")
    return is_ground


# --- Step 4: check it against the survey -----------------------------------
def score_against_survey(is_ground, cls):
    """IGN already classified this tile. Treat their class 2 as the answer key."""
    truth = cls == ASPRS_GROUND
    tp = int(np.sum(is_ground & truth))
    fp = int(np.sum(is_ground & ~truth))
    fn = int(np.sum(~is_ground & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    _say("Step 4", f"survey ground {truth.sum():,} points, ours {is_ground.sum():,}")
    _say("Step 4", f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}")
    if fp:
        u, n = np.unique(cls[is_ground & ~truth], return_counts=True)
        worst = sorted(zip(n, u), reverse=True)[:3]
        _say("Step 4", "false ground comes from survey classes "
                       + ", ".join(f"{c}: {k:,}" for k, c in worst))
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# --- Step 5: terrain model -------------------------------------------------
def ground_to_dtm(xyz, is_ground):
    """Inverse-distance weighting from the ground points onto a regular grid.

    Every cell gets a height, including the ones under buildings, where the
    nearest ground points sit around the footprint.
    """
    g = xyz[is_ground]
    x0 = np.floor(xyz[:, 0].min() / DTM_CELL) * DTM_CELL
    y1 = np.ceil(xyz[:, 1].max() / DTM_CELL) * DTM_CELL
    nx = int(np.ceil((xyz[:, 0].max() - x0) / DTM_CELL))
    ny = int(np.ceil((y1 - xyz[:, 1].min()) / DTM_CELL))
    cx = x0 + (np.arange(nx) + 0.5) * DTM_CELL
    cy = y1 - (np.arange(ny) + 0.5) * DTM_CELL        # row 0 is the north edge
    gx, gy = np.meshgrid(cx, cy)

    tree = cKDTree(g[:, :2])
    d, i = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=IDW_K)
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    dtm = (np.sum(w * g[i, 2], axis=1) / np.sum(w, axis=1)).reshape(ny, nx)
    empty = (d[:, 0] > DTM_CELL).reshape(ny, nx)
    _say("Step 5", f"DTM {nx} x {ny} cells at {DTM_CELL} m, z {dtm.min():.1f} "
                   f"to {dtm.max():.1f} m, {100.0 * empty.mean():.1f} % of cells "
                   "filled from neighbours (no ground point inside)")
    return dtm, (x0, y1), empty


def write_dtm(dtm, origin, crs, path):
    import rasterio
    from rasterio.transform import from_origin
    x0, y1 = origin
    with rasterio.open(path, "w", driver="GTiff", height=dtm.shape[0],
                       width=dtm.shape[1], count=1, dtype="float32",
                       crs=crs.to_wkt() if crs else None,
                       transform=from_origin(x0, y1, DTM_CELL, DTM_CELL),
                       compress="deflate") as dst:
        dst.write(dtm.astype(np.float32), 1)
    _say("Step 5", f"wrote {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB)")


def write_hillshade(dtm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dtm, cmap=plt.cm.gist_earth, vert_exag=2.0,
                   dx=DTM_CELL, dy=DTM_CELL, blend_mode="soft")
    plt.imsave(path, rgb)
    _say("Step 5", f"wrote {os.path.basename(path)}")


# --- Step 6: height above ground -------------------------------------------
def height_above_ground(xyz, dtm, origin, cls):
    """Subtract the terrain under every point. A 4 m hedge on a slope and a
    4 m hedge on the flat now read the same."""
    x0, y1 = origin
    col = np.clip(((xyz[:, 0] - x0) / DTM_CELL).astype(int), 0, dtm.shape[1] - 1)
    row = np.clip(((y1 - xyz[:, 1]) / DTM_CELL).astype(int), 0, dtm.shape[0] - 1)
    hag = xyz[:, 2] - dtm[row, col]
    for c, name in [(2, "ground"), (5, "high vegetation"), (6, "building")]:
        m = cls == c
        if m.any():
            _say("Step 6", f"survey class {c} ({name}): median height above ground "
                           f"{np.median(hag[m]):.2f} m, 95th percentile "
                           f"{np.percentile(hag[m], 95):.2f} m")
    return hag


# --- Step 7: keep the ground -----------------------------------------------
def write_ground(las, is_ground, path):
    """The ground points with every original field, IGN's class included, so
    you can open them in CloudCompare and see where the two disagree."""
    out = laspy.LasData(las.header)
    out.points = las.points[is_ground]
    out.write(path)
    _say("Step 7", f"wrote {os.path.basename(path)}: {is_ground.sum():,} points, "
                   f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    las, xyz, cls, crs = read_tile()
    downsample_and_index(xyz)
    is_ground = csf_ground(xyz)
    score_against_survey(is_ground, cls)
    dtm, origin, _ = ground_to_dtm(xyz, is_ground)
    write_dtm(dtm, origin, crs, os.path.join(OUT, "dtm.tif"))
    write_hillshade(dtm, os.path.join(OUT, "dtm_hillshade.png"))
    height_above_ground(xyz, dtm, origin, cls)
    write_ground(las, is_ground, os.path.join(OUT, "ground.laz"))

    print(f"\nDone in {time.time() - t_start:.1f}s. Outputs in {OUT}")
