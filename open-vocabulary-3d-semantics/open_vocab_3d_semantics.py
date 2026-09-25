# -*- coding: utf-8 -*-
#
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/turn-video-into-smart-3d-models-the-python-guide-with-sam-clip-and-dino-f4878d4c37dc
# Dataset:  https://learngeodata.eu/materials/open-vocabulary-3d-semantics/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
#
# ---------------------------------------------------------------------------
# Open-vocabulary 3D semantics: lift 2D labels onto a real room scan, fuse them
# across cameras, vote out the flicker, refine with one click, slice a plan.
# ---------------------------------------------------------------------------
#
# READ THIS FIRST: WHAT IS REAL AND WHAT IS SIMULATED
#
# This is an honest CPU walkthrough of the article's pipeline. It runs with no
# GPU and no model weights, and it does NOT run SAM, CLIP or DINOv2.
#
#   Real:
#     * The scan. room.ply is a real handheld capture of a living room, with
#       colour and a per-point class for every point (the reference labels).
#     * The 3D logic, all of it: floor RANSAC and orientation, scale from the
#       ceiling, the pixel-to-point link with a depth test, lifting 2D label
#       maps and match masks onto the points, fusing them across cameras, the
#       k-NN consistency vote, the seed region grow, the floor-plan slice, and
#       the per-class queries.
#     * Every number the script prints. They are measured on the scan, against
#       its reference labels.
#
#   Simulated:
#     * The camera frames. The scan ships without its video frames and poses,
#       so 36 virtual cameras are placed inside the room and the link is
#       built by projecting the points into them (Step 1 and Step 4).
#     * The 2D model outputs. predict_2d_labels() stands in for SAM + CLIP: it
#       paints each frame with the reference labels, then flips 12 % of the
#       pixels to a wrong class from the vocabulary, per frame, independently.
#       predict_2d_match() stands in for DINOv2: it keeps 90 % of the target's
#       pixels in each frame and adds 2 % false positives.
#     * The human click in Step 8. The seed and the "stop at the boundary"
#       test come from the reference labels, standing in for a person's eye.
#
#   Plug in real models here (the only two functions to replace):
#     predict_2d_labels(scene) -> {frame_id: np.ndarray int32 (H, W)}
#         one class id per pixel, -1 where no mask covers the pixel. Class ids
#         are the keys of scene["names"] (map your CLIP prompt index to them).
#     predict_2d_match(scene, target_class) -> {frame_id: np.ndarray bool (H, W)}
#         True where the DINOv2 match lands in that frame.
#     (H, W) = scene["img_shape"]; frame ids are the keys of scene["link"].
#     For real frames, also replace the virtual cameras in load_scene() with
#     your poses: R (3x3, world to camera), position (3,), intrinsic K (3x3).
#     Everything downstream of those two functions is unchanged.
#
# Differences from the article's code blocks, on purpose:
#   * The article's `if not DEV:` branches called a private engine with
#     functions and signatures that do not exist, so they failed on any
#     machine. They are removed. There is no DEV switch: this is the DEV path.
#   * Step 5 and Step 6 now go through a 2D-to-3D lift (per-frame label maps
#     and masks, looked up through the Step 4 link), instead of adding noise
#     straight onto the points. Same idea, and it makes the plug-in point real.
#   * Step 8 runs the article's region grow from a seed by default
#     (method="region_grow"). The earlier script's largest-connected-component
#     variant is kept as method="components".
#   * Cameras move with the cloud when Steps 2 and 3 rotate and scale it.
#
# Run:
#   pip install -r requirements.txt
#   python open_vocab_3d_semantics.py
#
# Outputs land in out/: room_labeled.ply, floorplan_slice.ply,
# class_stats.csv, run_stats.json, run_log.txt, and one figure per step in
# out/viz/.

import csv
import json
import os
import sys
import time

import numpy as np

# Keep prints readable when the console is not UTF-8 (Windows default).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataviz as dv  # noqa: E402  (the kit's figure and loading helpers)

# --- configuration ---------------------------------------------------------
# The real scan: a living room, colour plus a per-point class.
DATA_FILE = os.path.join(HERE, "room.ply")
N_POINTS = 40000          # working subsample, as in the article
VIZ = True                # save one figure per step
OUT_DIR = os.path.join(HERE, "out")
VIZ_DIR = os.path.join(OUT_DIR, "viz")

# Names for the scan's class codes. The five small codes were not named when
# the scan was labeled, so they stay numbered. Edit it: this dict IS the
# open vocabulary (with real CLIP, these strings are your prompts).
CLASS_NAMES = {0: "wall", 1: "floor", 2: "ceiling", 5: "ceiling fixture",
               23: "furniture and clutter", 4: "object 4", 7: "object 7",
               9: "object 9", 15: "object 15", 19: "object 19"}

# The object we outline once and match across every frame (Step 6).
TARGET_CLASS = 9

# Ceiling reference used to scale the scene (Step 3). The video uses 2.5 m.
CEILING_REFERENCE_M = 2.5

# Simulated 2D model behaviour (see the header). Change these to stress Step 7.
LABEL_FLICKER = 0.12      # share of pixels per frame given a wrong class
MATCH_RECALL = 0.90       # share of the target's pixels the match keeps per frame
MATCH_FALSE_POS = 0.02    # share of other pixels the match wrongly keeps per frame

RNG = np.random.default_rng(42)
STATS = {"timings_s": {}}


def _v(name):
    """Path of one step figure."""
    return os.path.join(VIZ_DIR, name + ".png")


def _class_names(labels):
    """Display name per class code present in the scan."""
    return {int(u): CLASS_NAMES.get(int(u), f"class {int(u)}") for u in np.unique(labels)}


def _acc(pred, ref):
    """Share of points whose label matches the reference (unlabeled counts as wrong)."""
    return float((pred == ref).mean())


# ===========================================================================
# %% Step 1. Load the reconstruction and the cameras that saw it
# ===========================================================================
# A reconstruction is three things: the cloud, the frames the camera saw, and
# the provenance that ties each point to the pixel it came from. The scan here
# comes without its frames, so we place a ring of virtual cameras inside the
# room, the way a handheld walkthrough turns on the spot and sweeps up and down.

def load_scene():
    """Return a scene dict: xyz, rgb, reference labels, cameras, image size."""
    c = dv.load_cloud(DATA_FILE, max_points=N_POINTS, want_labels=True)
    xyz = c["xyz"].astype(np.float32)
    rgb = c["rgb"]
    labels = c["labels"]

    # 36 virtual frames from the middle of the room: 12 headings, 30 degrees
    # apart, each looking down, level and up, the way a handheld walkthrough
    # sweeps the floor, the walls and the ceiling. A 90 degree lens, like a
    # phone's wide camera.
    img_h, img_w = 270, 480
    K = np.array([[240, 0, img_w / 2], [0, 240, img_h / 2], [0, 0, 1]], float)
    headings, pitches = 12, (-50.0, -5.0, 40.0)
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    center = 0.5 * (lo + hi)
    eye_z = lo[2] + 0.6 * (hi[2] - lo[2])
    ring = 0.25 * float(min(hi[0] - lo[0], hi[1] - lo[1])) / 2
    cams = []
    for h in range(headings):
        ang = 2 * np.pi * h / headings
        pos = np.array([center[0] + ring * np.cos(ang), center[1] + ring * np.sin(ang), eye_z])
        for pitch in pitches:
            fwd = np.array([np.cos(ang), np.sin(ang), np.tan(np.radians(pitch))])
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            R = np.stack([right, -up, fwd])        # world -> camera
            cams.append({"frame": len(cams), "R": R, "position": pos, "intrinsic": K})
    n_frames = len(cams)

    names = _class_names(labels)
    print(f"[Step 1] Loaded {len(xyz):,} points from {c['source']} and {n_frames} virtual frames "
          f"({img_w} x {img_h}).")
    print("[Step 1] Classes: " + ", ".join(
        f"{u} {names[u]} ({int((labels == u).sum()):,})" for u in names))

    if VIZ:
        dv.scatter3d(xyz, _v("p13_step1_raw_rgb"),
                     title="Step 1: raw reconstruction (geometry, colour, no meaning yet)",
                     rgb=rgb)

    return {"xyz": xyz, "rgb": rgb, "labels": labels, "cams": cams, "K": K,
            "img_shape": (img_h, img_w), "names": names, "n_points": len(xyz)}


# ===========================================================================
# %% Step 2. Orient to gravity from the floor plane (RANSAC)
# ===========================================================================

def fit_plane_ransac(pts, thresh=0.03, iters=300):
    """Return (normal, d, inlier_mask) for n.x + d = 0, the largest plane found."""
    best_inliers, best_n, best_d = None, None, None
    n_pts = len(pts)
    for _ in range(iters):
        i, j, k = RNG.choice(n_pts, size=3, replace=False)
        v1, v2 = pts[j] - pts[i], pts[k] - pts[i]
        nrm = np.cross(v1, v2)
        norm = np.linalg.norm(nrm)
        if norm < 1e-8:
            continue
        nrm = nrm / norm
        d = -nrm @ pts[i]
        dist = np.abs(pts @ nrm + d)
        inliers = dist < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_n, best_d = inliers, nrm, d
    return best_n, best_d, best_inliers


def orient_to_floor(scene):
    """Rotate the cloud (and the cameras) so the floor normal points to +Z."""
    xyz = scene["xyz"]
    # Bias RANSAC toward the floor: sample from the lower third in current Z.
    z = xyz[:, 2]
    low = xyz[z < np.percentile(z, 35)]
    normal, _, inliers = fit_plane_ransac(low if len(low) > 500 else xyz)
    if normal is None:
        return scene
    if normal[2] < 0:                       # make it point up
        normal = -normal

    # Rotation that sends the plane normal to +Z (Rodrigues).
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        R = np.eye(3)
    else:
        axis /= s
        cos = float(np.clip(normal @ target, -1, 1))
        ang = np.arccos(cos)
        kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(ang) * kx + (1 - cos) * (kx @ kx)

    tilt_before = np.degrees(np.arccos(abs(float(normal @ target))))
    xyz_oriented = (R @ xyz.T).T.astype(np.float32)
    scene["xyz"] = xyz_oriented
    scene["floor_inliers"] = inliers
    scene["R_floor"] = R
    # The cameras move with the cloud: x' = R x, so R_cam' = R_cam R^T, pos' = R pos.
    for cam in scene["cams"]:
        cam["R"] = cam["R"] @ R.T
        cam["position"] = R @ cam["position"]
    print(f"[Step 2] Floor tilt corrected: {tilt_before:.1f} deg -> 0.0 deg "
          f"({int(inliers.sum()):,} floor inliers).")
    STATS["tilt_deg"] = round(float(tilt_before), 2)

    if VIZ:
        dv.compare3d(xyz, xyz_oriented, _v("p13_step2_orient_floor"),
                     title="Step 2: auto-orient from the floor plane (tilted vs levelled)",
                     name_a="as reconstructed (tilted)", name_b="levelled to gravity")
    return scene


# ===========================================================================
# %% Step 3. Identify an indoor room and scale it from the ceiling
# ===========================================================================

def identify_and_scale(scene):
    """Detect the floor and ceiling bands, scale the scene so ceiling = reference."""
    z = scene["xyz"][:, 2]
    hist, edges = np.histogram(z, bins=80)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Floor = lowest dense band, ceiling = highest dense band.
    dense = centers[hist > 0.25 * hist.max()]
    floor_z, ceil_z = float(dense.min()), float(dense.max())
    measured_height = ceil_z - floor_z
    scale = CEILING_REFERENCE_M / measured_height if measured_height > 1e-6 else 1.0

    scene["xyz"] = (scene["xyz"] * scale).astype(np.float32)
    scene["scale"] = scale
    scene["room_type"] = "indoor room"
    for cam in scene["cams"]:               # a uniform scale leaves every pixel in place
        cam["position"] = cam["position"] * scale
    print(f"[Step 3] Detected indoor room. Raw height {measured_height:.3f} units "
          f"-> scaled x{scale:.3f} so ceiling = {CEILING_REFERENCE_M} m.")
    STATS["scale"] = round(float(scale), 4)

    if VIZ:
        fig, ax = dv.new_fig(title="Step 3: height histogram (floor and ceiling bands set the scale)")
        ax.bar(centers, hist, width=(centers[1] - centers[0]), color=dv.CYAN)
        ax.axvline(floor_z, color=dv.GOLD, lw=2, label=f"floor z={floor_z:.3f}")
        ax.axvline(ceil_z, color=dv.RED, lw=2, label=f"ceiling z={ceil_z:.3f}")
        ax.set_xlabel("height (scan units)"); ax.set_ylabel("point count"); ax.legend()
        dv.save(fig, _v("p13_step3_scale_from_ceiling"))
    return scene


# ===========================================================================
# %% Step 4. Build the pixel-point link (project, with a depth test)
# ===========================================================================

def project_points_to_frame(cam, img_shape, points, depth_tol=0.04):
    """Return (visible_idx, uv) for points landing inside this frame, front surface only."""
    H, W = img_shape
    R = np.asarray(cam["R"], float)
    pos = np.asarray(cam["position"], float)
    K = np.asarray(cam["intrinsic"], float)
    Pc = (points - pos) @ R.T                     # world -> camera
    z = Pc[:, 2]
    front = z > 1e-6
    zf = np.where(front, z, 1.0)
    u = np.round(K[0, 0] * Pc[:, 0] / zf + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * Pc[:, 1] / zf + K[1, 2]).astype(np.int64)
    inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.flatnonzero(inb)
    if idx.size == 0:
        return idx, np.empty((0, 2), np.int64)
    # Per-pixel nearest-depth z-buffer, then drop occluded points.
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v[idx], u[idx]), z[idx])
    keep = z[idx] <= zbuf[v[idx], u[idx]] * (1.0 + depth_tol) + 1e-6
    idx = idx[keep]
    return idx, np.stack([u[idx], v[idx]], axis=1)


def build_pixel_link(scene):
    """Per-frame visible point indices and their (u, v): the Rosetta Stone."""
    xyz = scene["xyz"]
    link, coverage = {}, []
    views = np.zeros(len(xyz), np.int32)
    for cam in scene["cams"]:
        idx, uv = project_points_to_frame(cam, scene["img_shape"], xyz)
        link[cam["frame"]] = {"idx": idx, "uv": uv}
        views[idx] += 1
        coverage.append(len(idx) / len(xyz))
    scene["link"] = link
    scene["views"] = views
    seen = float((views > 0).mean())
    per = [len(fr["idx"]) for fr in link.values()]
    print(f"[Step 4] Built pixel-point link over {len(link)} frames; each frame sees "
          f"{min(per):,} to {max(per):,} points (median {int(np.median(per)):,}).")
    print(f"[Step 4] {seen:.1%} of points seen by at least one frame, "
          f"median {int(np.median(views[views > 0]))} views per seen point.")
    STATS["seen_share"] = round(seen, 4)
    STATS["median_views"] = int(np.median(views[views > 0]))

    if VIZ:
        H, W = scene["img_shape"]
        cam0 = scene["cams"][0]
        Pc = (xyz - cam0["position"]) @ np.asarray(cam0["R"]).T
        idx = link[0]["idx"]; uv = link[0]["uv"]
        depth_img = np.zeros((H, W))
        if len(idx):
            depth_img[uv[:, 1], uv[:, 0]] = Pc[idx, 2]
        dv.heatmap(depth_img, _v("p13_step4_pixel_link_depth"),
                   title="Step 4: frame 0 depth from the pixel-point link",
                   xlabel="pixel u", ylabel="pixel v", cbar_label="depth (m)")
    return scene


# ===========================================================================
# %% Step 5. Propose segments (SAM) and name them (CLIP), then lift to 3D
# ===========================================================================
# SAM proposes regions in each frame; CLIP names each region from a list of
# words. The output per frame is a label map: one class id per pixel. Lifting
# it to 3D is a lookup through the Step 4 link, and fusing it is a vote per
# point over every frame that saw the point.

def predict_2d_labels(scene):
    """PLUG-IN POINT for SAM + CLIP. SIMULATED in this kit.

    Must return {frame_id: label_map}, label_map an int32 array of shape
    scene["img_shape"] = (H, W): a class id (a key of scene["names"]) per pixel,
    -1 where no mask covers it. Replace the body with your real model calls."""
    H, W = scene["img_shape"]
    vocab = np.array(sorted(scene["names"]))
    ref = scene["labels"]
    maps = {}
    for fid, fr in scene["link"].items():
        lab = np.full((H, W), -1, np.int32)
        lab[fr["uv"][:, 1], fr["uv"][:, 0]] = ref[fr["idx"]]      # a perfect labeler...
        painted = np.flatnonzero(lab.ravel() >= 0)
        flip = painted[RNG.random(len(painted)) < LABEL_FLICKER]  # ...that flickers
        lab.ravel()[flip] = RNG.choice(vocab, size=len(flip))
        maps[fid] = lab
    return maps


def propose_and_label_segments(scene):
    """Return per-point class ids: 2D label maps lifted through the link and fused."""
    maps = predict_2d_labels(scene)
    vocab = np.array(sorted(scene["names"]))
    col = {int(c): i for i, c in enumerate(vocab)}
    n = scene["n_points"]
    votes = np.zeros((n, len(vocab)), np.int32)
    ref = scene["labels"]
    frame_acc = []
    for fid, fr in scene["link"].items():
        got = maps[fid][fr["uv"][:, 1], fr["uv"][:, 0]]            # the lift: a lookup
        ok = got >= 0
        frame_acc.append(float((got[ok] == ref[fr["idx"][ok]]).mean()))
        cols = np.array([col[int(g)] for g in got[ok]], np.int64)
        np.add.at(votes, (fr["idx"][ok], cols), 1)
    fused = np.where(votes.sum(1) > 0, vocab[votes.argmax(1)], -1).astype(np.int32)
    scene["labels_noisy"] = fused
    seen = fused >= 0
    STATS["acc_per_frame_2d"] = round(float(np.mean(frame_acc)), 4)
    STATS["acc_fused_seen"] = round(_acc(fused[seen], ref[seen]), 4)
    STATS["acc_fused_all"] = round(_acc(fused, ref), 4)
    print(f"[Step 5] SAM+CLIP (simulated) label maps: {np.mean(frame_acc):.1%} of lifted "
          f"pixels right per frame.")
    print(f"[Step 5] Fused over {len(maps)} frames: {STATS['acc_fused_seen']:.1%} right on the "
          f"{int(seen.sum()):,} seen points, {int((~seen).sum()):,} points unlabeled (no view).")

    if VIZ:
        names = dict(scene["names"]); names[-1] = "unlabeled (no view)"
        dv.scatter3d(scene["xyz"], _v("p13_step5_sam_clip_labels"),
                     title="Step 5: 2D labels lifted and fused (with flicker)",
                     labels=fused, legend_names=names)
    return fused


# ===========================================================================
# %% Step 6. In-context matching with DINOv2 (outline once, find everywhere)
# ===========================================================================

def predict_2d_match(scene, target_class):
    """PLUG-IN POINT for DINOv2 in-context matching. SIMULATED in this kit.

    Must return {frame_id: mask}, mask a bool array of shape scene["img_shape"],
    True where the match for the outlined object lands in that frame."""
    H, W = scene["img_shape"]
    ref = scene["labels"]
    masks = {}
    for fid, fr in scene["link"].items():
        m = np.zeros((H, W), bool)
        uv, is_t = fr["uv"], ref[fr["idx"]] == target_class
        hit = is_t & (RNG.random(len(uv)) < MATCH_RECALL)
        stray = ~is_t & (RNG.random(len(uv)) < MATCH_FALSE_POS)
        m[uv[hit | stray, 1], uv[hit | stray, 0]] = True
        masks[fid] = m
    return masks


def incontext_match(scene, target_class=TARGET_CLASS, min_share=0.5):
    """Lift one object's per-frame match masks to 3D point indices.

    Each point counts how many of the frames that see it put it inside the
    match. It is kept when that share reaches min_share. A plain union across
    frames (min_share close to 0, the article's version) keeps every stray
    false positive from every frame; the share vote keeps the object."""
    masks = predict_2d_match(scene, target_class)
    ref = scene["labels"]
    hits = np.zeros(scene["n_points"], np.int32)
    for fid, fr in scene["link"].items():
        inside = masks[fid][fr["uv"][:, 1], fr["uv"][:, 0]]
        hits[fr["idx"][inside]] += 1
    views = np.maximum(scene["views"], 1)
    matched = (hits > 0) & (hits / views >= min_share)
    union = hits > 0
    is_t0 = ref == target_class
    STATS["match_union_precision"] = round(float(is_t0[union].mean()), 4) if union.any() else 0.0

    is_t = ref == target_class
    recall = float(matched[is_t].mean()) if is_t.any() else 0.0
    precision = float(is_t[matched].mean()) if matched.any() else 0.0
    name = scene["names"].get(target_class, f"class {target_class}")
    print(f"[Step 6] DINOv2 match (simulated masks) for '{name}': lifted across {len(masks)} "
          f"frames, 3D recall {recall:.1%}, precision {precision:.1%} "
          f"(a plain union would give {STATS['match_union_precision']:.1%}).")
    STATS["match"] = {"target": name, "recall": round(recall, 4), "precision": round(precision, 4)}

    if VIZ:
        dv.scatter3d(scene["xyz"], _v("p13_step6_incontext_match"),
                     title=f"Step 6: one outline, matched everywhere ('{name}')",
                     labels=matched.astype(int),
                     legend_names={0: "rest of scene", 1: f"matched: {name}"})
    scene["matched_target"] = matched
    return matched


# ===========================================================================
# %% Step 7. Spatial-consistency vote (kill the flicker across cameras)
# ===========================================================================

def spatial_consistency_vote(scene, labels, k=16, min_cluster=20, min_agree=0.75):
    """Return labels after a k-NN majority vote that resolves cross-camera flicker.

    A point is relabeled only when at least min_agree of its labeled neighbours
    back one other class. min_agree=0 is the article's plain majority vote. On
    this scan the per-point fusion of Step 5 has already removed the bulk of the
    flicker, and a plain vote then erodes small objects and edges (the run log
    prints both, so you can see it). Points no camera saw (-1) take the vote of
    their labeled neighbours, then the nearest labeled point.

    min_cluster is kept for the article's signature: the production split into
    structure and objects uses it; this single vote does not."""
    from scipy.spatial import cKDTree
    xyz = scene["xyz"]
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=k, workers=-1)        # (n, k) neighbour indices
    vocab = np.array(sorted(scene["names"]))
    lut = np.full(int(vocab.max()) + 2, -1, np.int64)
    lut[vocab] = np.arange(len(vocab))
    neigh = labels[nn]                              # (n, k), -1 = unlabeled
    counts = np.zeros((len(xyz), len(vocab)), np.int32)
    for j in range(neigh.shape[1]):
        ok = neigh[:, j] >= 0
        counts[np.flatnonzero(ok), lut[neigh[ok, j]]] += 1
    total = counts.sum(1)
    top = vocab[counts.argmax(1)]
    share = counts.max(1) / np.maximum(total, 1)

    ref = scene["labels"]
    seen = labels >= 0
    plain = np.where(total > 0, top, -1).astype(np.int32)       # the article's vote
    voted = labels.copy().astype(np.int32)
    strong = seen & (total > 0) & (share >= min_agree)
    voted[strong] = top[strong]
    voted[~seen & (total > 0)] = top[~seen & (total > 0)]
    STATS["acc_seen_before_vote"] = round(_acc(labels[seen], ref[seen]), 4)
    STATS["acc_seen_plain_vote"] = round(_acc(plain[seen], ref[seen]), 4)
    STATS["acc_seen_after_vote"] = round(_acc(voted[seen], ref[seen]), 4)

    hole = voted < 0
    if hole.any() and (~hole).any():
        _, near = cKDTree(xyz[~hole]).query(xyz[hole], k=1, workers=-1)
        voted[hole] = voted[~hole][near]
    changed = int((voted != labels).sum())
    before, after = _acc(labels, ref), _acc(voted, ref)
    STATS["acc_before_vote"] = round(before, 4)
    STATS["acc_after_vote"] = round(after, 4)
    print(f"[Step 7] Consistency vote (k={k}, min_agree={min_agree}): {changed:,} points changed.")
    print(f"[Step 7] On the points the cameras saw: {STATS['acc_seen_before_vote']:.1%} -> "
          f"{STATS['acc_seen_after_vote']:.1%} (a plain majority vote would give "
          f"{STATS['acc_seen_plain_vote']:.1%}).")
    print(f"[Step 7] Whole cloud, unseen points filled from neighbours: {before:.1%} -> {after:.1%}.")

    if VIZ:
        dv.bars(["before vote", "after vote"], [before * 100, after * 100],
                _v("p13_step7_consistency_vote"),
                title="Step 7: agreement with the reference labels before and after the vote",
                color=[dv.GOLD, dv.CYAN], ylabel="accuracy (%)")
    return voted


# ===========================================================================
# %% Step 8. Human in the loop: correct one segment, propagate the fix
# ===========================================================================
# Click one point, say "this is floor", and the label grows over the k-NN graph
# through points that agree, stopping at boundaries. Here the click and the
# "agree" test are simulated from the reference labels (see the header).

def human_refine(scene, labels, seed_class=None, n_seeds=1, method="region_grow"):
    """Region-grow a corrected label from a seed point. Returns updated labels.

    method="region_grow" (default, the article): flood fill from the point
    nearest the class centroid. method="components": the largest connected
    same-class component, the variant the earlier script used."""
    from scipy.spatial import cKDTree
    xyz = scene["xyz"]
    gt = scene["labels"]
    if seed_class is None:
        seed_class = int(np.bincount(gt).argmax())

    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=12, workers=-1)        # (n, 12)

    if method == "region_grow":
        cls_idx = np.flatnonzero(gt == seed_class)
        centroid = xyz[cls_idx].mean(axis=0)
        seed = int(cls_idx[np.argmin(np.linalg.norm(xyz[cls_idx] - centroid, axis=1))])
        region = np.zeros(len(xyz), bool)
        region[seed] = True
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            for nb in nn[cur]:
                if not region[nb] and gt[nb] == seed_class:
                    region[nb] = True
                    frontier.append(int(nb))
    elif method == "components":
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        src = np.repeat(np.arange(len(xyz)), nn.shape[1])
        dst = nn.ravel()
        same = (gt[src] == seed_class) & (gt[dst] == seed_class)
        adj = coo_matrix((np.ones(int(same.sum())), (src[same], dst[same])),
                         shape=(len(xyz), len(xyz)))
        _, comp = connected_components(adj, directed=False)
        in_cls = gt == seed_class
        biggest = int(np.bincount(comp[in_cls]).argmax())
        region = in_cls & (comp == biggest)
    else:
        raise ValueError(f"unknown method {method!r}: use 'region_grow' or 'components'")

    out = labels.copy()
    fixed = int((labels[region] != seed_class).sum())
    out[region] = seed_class
    name = scene["names"].get(seed_class, f"class {seed_class}")
    acc = _acc(out, gt)
    STATS["refine"] = {"method": method, "class": name, "grown": int(region.sum()),
                       "fixed": fixed, "acc_after": round(acc, 4)}
    print(f"[Step 8] Seed click on '{name}' ({method}) grew to {int(region.sum()):,} points, "
          f"fixing {fixed} mislabeled ones; agreement now {acc:.1%}.")

    if VIZ:
        dv.scatter3d(scene["xyz"], _v("p13_step8_seed_propagate"),
                     title=f"Step 8: one click grew over the whole surface ('{name}')",
                     labels=region.astype(int),
                     legend_names={0: "rest of scene", 1: "selected by 1 seed click"})
    return out


# ===========================================================================
# %% Step 9. Slice to a measurable floor plan
# ===========================================================================

def export_floorplan_slice(scene, thickness_m=0.30, out_path=None):
    """Cut a horizontal slab, report the footprint, write a slice cloud."""
    xyz = scene["xyz"]
    z = xyz[:, 2]
    # Slice 1 m above the floor (a typical plan cut height).
    cut = float(np.percentile(z, 5)) + 1.0
    slab = (z >= cut - thickness_m / 2) & (z <= cut + thickness_m / 2)
    pts2d = xyz[slab][:, :2]
    if out_path is None:
        out_path = os.path.join(OUT_DIR, "floorplan_slice.ply")
    ext_x = float(pts2d[:, 0].max() - pts2d[:, 0].min()) if len(pts2d) else 0.0
    ext_y = float(pts2d[:, 1].max() - pts2d[:, 1].min()) if len(pts2d) else 0.0
    # The scan's X and Y do not follow the walls, so the box above is too big
    # for a room turned on the grid. Turn the slice (0.5 degree steps) until its
    # box is smallest: that box runs along the walls.
    wall_deg, wall_x, wall_y = 0.0, ext_x, ext_y
    if len(pts2d):
        c = pts2d - pts2d.mean(0)
        best = None
        for deg in np.arange(0.0, 90.0, 0.5):
            t = np.radians(deg)
            r = c @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            lo, hi = np.percentile(r, 0.5, axis=0), np.percentile(r, 99.5, axis=0)
            area = float(np.prod(hi - lo))
            if best is None or area < best[0]:
                best = (area, deg, float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        _, wall_deg, wall_x, wall_y = best

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts2d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y in pts2d:
            f.write(f"{x:.4f} {y:.4f} {cut:.4f}\n")

    STATS["slice"] = {"cut_z_m": round(cut, 3), "points": int(len(pts2d)),
                      "footprint_axis_m": [round(ext_x, 2), round(ext_y, 2)],
                      "footprint_walls_m": [round(wall_x, 2), round(wall_y, 2)],
                      "walls_turned_deg": float(wall_deg)}
    scene["slice_mask"] = slab
    scene["walls_deg"] = float(wall_deg)
    print(f"[Step 9] Floor-plan slice at z={cut:.2f} m, thickness {thickness_m*1000:.0f} mm: "
          f"{len(pts2d):,} points -> {os.path.basename(out_path)}")
    print(f"[Step 9] Footprint along the scan axes {ext_x:.2f} x {ext_y:.2f} m; along the walls "
          f"(turned {wall_deg:.1f} deg, 0.5 % trimmed per side) {wall_x:.2f} x {wall_y:.2f} m. "
          f"Metres assume the {CEILING_REFERENCE_M} m ceiling of Step 3.")

    if VIZ and len(pts2d):
        fig, ax = dv.new_fig(title=f"Step 9: floor-plan slice ({wall_x:.1f} x {wall_y:.1f} m along the walls)")
        ax.scatter(pts2d[:, 0], pts2d[:, 1], s=2, c=dv.TEXT)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        dv.save(fig, _v("p13_step9_floorplan_slice"))
    return out_path


# ===========================================================================
# %% Query the labeled scan (the deliverable is a table, not a picture)
# ===========================================================================

def query_class(scene, labels, name):
    """Everything the labeled cloud knows about one class, in metres.

    Footprints run along the walls found in Step 9, 0.5 % trimmed per side."""
    code = {v: k for k, v in scene["names"].items()}[name]
    m = labels == code
    if not m.any():
        return {"class": name, "code": code, "points": 0}
    p = scene["xyz"][m]
    t = np.radians(scene.get("walls_deg", 0.0))
    xy = p[:, :2] @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    return {"class": name, "code": code, "points": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "z_min_m": round(float(p[:, 2].min()), 2), "z_max_m": round(float(p[:, 2].max()), 2),
            "footprint_a_m": round(float(hi[0] - lo[0]), 2),
            "footprint_b_m": round(float(hi[1] - lo[1]), 2),
            "agree_with_reference": round(float((scene["labels"][m] == code).mean()), 4)}


def write_outputs(scene, labels):
    """room_labeled.ply, class_stats.csv and run_stats.json in out/."""
    from plyfile import PlyData, PlyElement
    n = scene["n_points"]
    v = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                           ("label", "i4"), ("reference", "i4")])
    for i, a in enumerate("xyz"):
        v[a] = scene["xyz"][:, i]
    rgb8 = np.clip(np.round(scene["rgb"] * 255), 0, 255).astype(np.uint8)
    v["red"], v["green"], v["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
    v["label"], v["reference"] = labels, scene["labels"]
    ply_path = os.path.join(OUT_DIR, "room_labeled.ply")
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(ply_path)

    rows = [query_class(scene, labels, nm) for nm in scene["names"].values()]
    keys = ["class", "code", "points", "share", "z_min_m", "z_max_m",
            "footprint_a_m", "footprint_b_m", "agree_with_reference"]
    with open(os.path.join(OUT_DIR, "class_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["points"]):
            w.writerow({k: r.get(k, "") for k in keys})

    STATS["classes"] = rows
    STATS["R_floor"] = np.round(scene["R_floor"], 6).tolist()
    STATS["class_names"] = {str(k): v for k, v in scene["names"].items()}
    with open(os.path.join(OUT_DIR, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(STATS, f, indent=2)
    floor = query_class(scene, labels, "floor")
    print(f"[Query] 'floor': {floor['points']:,} points, {floor['footprint_a_m']} x "
          f"{floor['footprint_b_m']} m along the walls, {floor['agree_with_reference']:.1%} agree with the reference.")
    print(f"[Output] room_labeled.ply ({os.path.getsize(ply_path)/1e6:.1f} MB), class_stats.csv, "
          f"run_stats.json")


class _Tee:
    """Print to the console and to out/run_log.txt at once."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.s = sys.stdout

    def write(self, t):
        self.s.write(t); self.f.write(t)

    def flush(self):
        self.s.flush(); self.f.flush()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs(VIZ_DIR, exist_ok=True)
    sys.stdout = _Tee(os.path.join(OUT_DIR, "run_log.txt"))
    print("[pipeline] CPU walkthrough: real scan and real 3D logic; SAM, CLIP and DINOv2 "
          "outputs are SIMULATED (see the header).")
    t_all = time.time()
    steps = [
        ("1_load", lambda s: load_scene()),
        ("2_orient", orient_to_floor),
        ("3_scale", identify_and_scale),
        ("4_link", build_pixel_link),
    ]
    scene = None
    for key, fn in steps:
        t = time.time(); scene = fn(scene); STATS["timings_s"][key] = round(time.time() - t, 2)

    t = time.time(); labels = propose_and_label_segments(scene)
    STATS["timings_s"]["5_label"] = round(time.time() - t, 2)
    t = time.time(); incontext_match(scene)
    STATS["timings_s"]["6_match"] = round(time.time() - t, 2)
    t = time.time(); labels = spatial_consistency_vote(scene, labels)
    STATS["timings_s"]["7_vote"] = round(time.time() - t, 2)
    t = time.time(); labels = human_refine(scene, labels, seed_class=1, n_seeds=1)
    STATS["timings_s"]["8_refine"] = round(time.time() - t, 2)
    t = time.time(); export_floorplan_slice(scene)
    STATS["timings_s"]["9_slice"] = round(time.time() - t, 2)

    STATS["timings_s"]["total"] = round(time.time() - t_all, 2)
    write_outputs(scene, labels)
    print(f"\n[pipeline] done in {time.time() - t_all:.1f}s (figures included). Outputs in out/.")
