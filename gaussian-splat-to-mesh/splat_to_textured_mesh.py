# -*- coding: utf-8 -*-
#
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://learngeodata.eu/materials/gaussian-splat-to-mesh/
# Dataset:  https://learngeodata.eu/materials/gaussian-splat-to-mesh/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
#
# ---------------------------------------------------------------------------
# 3D Gaussian splat -> clean TEXTURED mesh, via a point cloud, 100% local.
# ---------------------------------------------------------------------------
#
# The strategy this script answers, end to end and on a CPU:
#   trained splat .ply  ->  read its anatomy  ->  flip it upright
#      ->  filtered point cloud (opacity + scale, SH DC -> RGB)
#      ->  densified cloud (sample each Gaussian ellipsoid)
#      ->  cleaned cloud + oriented normals
#      ->  Poisson surface + cleanup
#      ->  vertex colors  ->  UV unwrap (xatlas)  ->  baked texture image
#      ->  textured .glb you open in Blender.
#
# Dataset: a real trained 3D Gaussian splat of a cabin in a forest
# (84,528 Gaussians, ~21 MB). Trained with a COLMAP-initialised splat
# pipeline, so it comes out y-down: upside down in every y-up viewer.
# Step 2 fixes that explicitly because every reader hits it.
#
# Two ways to run it:
#   * The cabin splat present  -> the full real pipeline, every number printed.
#   * File missing (DEV)       -> a small synthetic splat is generated with the
#                                 same fields, so the whole script still runs.
#
# Run:
#   python -m py_compile splat_to_textured_mesh.py
#   python splat_to_textured_mesh.py          # CPU-only, a few minutes
#
import os
import sys
import time
import numpy as np

# Keep prints readable on Windows where the console default is not UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import open3d as o3d
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from PIL import Image

# --- configuration ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# The trained splat. Point this at your own scene's .ply export.
SPLAT_FILE = os.path.join(HERE, "cabin_in_the_forest.ply")

# SH DC -> RGB: rgb = 0.5 + C0 * f_dc, with C0 the zeroth SH basis constant.
SH_C0 = 0.28209479177387814

# Filters. Opacity is stored as a logit; sigmoid it, then keep >= OPACITY_MIN.
# Scales are stored as logs; exp them, then drop any Gaussian whose largest
# axis exceeds SCALE_MAX_FRAC of the scene's bounding-box diagonal (those are
# the huge sky/fog floaters, not surface detail).
OPACITY_MIN = 0.5
SCALE_MAX_FRAC = 0.01

# Densification: sample this many extra points inside each surviving Gaussian
# ellipsoid (plus the center itself), so Poisson sees a dense skin, not 66k dots.
SAMPLES_PER_SPLAT = 6

# Cloud cleanup + meshing knobs.
VOXEL_TARGET = 220000          # downsample the densified cloud to about this
SOR_NEIGHBORS, SOR_STD = 20, 2.0
POISSON_DEPTH = 10
DENSITY_QUANTILE = 0.055       # drop the lowest-density Poisson vertices

# Texturing knobs.
BAKE_TRIANGLES = 80000         # decimate to this before UV unwrap
TEXTURE_SIZE = 2048

RNG = np.random.default_rng(7)


def _say(step, msg):
    print(f"[{step}] {msg}")


# ===========================================================================
# %% Step 1. Read the splat and lay out its anatomy
# ===========================================================================
# A trained 3DGS .ply is not a normal point cloud. Every "vertex" is one
# anisotropic 3D Gaussian carrying 62 floats: a position (where it sits),
# three log-scales (how far it stretches on each axis), a quaternion (how the
# ellipsoid is rotated), an opacity logit (how solid it is), and 48 spherical
# harmonics coefficients (its color, including how that color shifts with the
# viewing angle). We parse it with plyfile and report what is actually inside.

def read_splat(path=SPLAT_FILE):
    if not os.path.exists(path):
        _say("Step 1", f"{path} not found -> generating a synthetic DEV splat")
        return _synth_splat()
    el = PlyData.read(path).elements[0]
    d = el.data
    n = el.count
    names = d.dtype.names
    n_rest = sum(1 for k in names if k.startswith("f_rest_"))
    splat = {
        "xyz": np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64),
        "scales_log": np.column_stack([d["scale_0"], d["scale_1"],
                                       d["scale_2"]]).astype(np.float64),
        "quat": np.column_stack([d["rot_0"], d["rot_1"], d["rot_2"],
                                 d["rot_3"]]).astype(np.float64),
        "opacity_logit": np.asarray(d["opacity"], dtype=np.float64),
        "f_dc": np.column_stack([d["f_dc_0"], d["f_dc_1"],
                                 d["f_dc_2"]]).astype(np.float64),
    }
    ext = splat["xyz"].max(0) - splat["xyz"].min(0)
    _say("Step 1", f"{n} Gaussians, {len(names)} floats each "
                   f"(3 pos + 3 log-scale + 4 quat + 1 opacity + 3 SH DC + {n_rest} SH rest)")
    _say("Step 1", f"scene extent {np.round(ext, 2)} units, "
                   f"file {os.path.getsize(path)/1e6:.1f} MB")
    return splat


def _synth_splat(n=20000):
    """DEV fallback: a box-with-roof point set dressed up with splat fields so
    every later step still runs when the cabin file is absent."""
    xyz = RNG.uniform([-2, -1, -2], [2, 0.6, 2], (n, 3))
    splat = {
        "xyz": xyz,
        "scales_log": np.log(RNG.uniform(0.01, 0.05, (n, 3))),
        "quat": np.tile([1.0, 0, 0, 0], (n, 1)),
        "opacity_logit": RNG.normal(2.0, 1.5, n),
        "f_dc": RNG.normal(0.0, 0.6, (n, 3)),
    }
    return splat


# ===========================================================================
# %% Step 2. Flip the scene upright (the fix everyone needs)
# ===========================================================================
# COLMAP's camera convention points y DOWN, and almost every splat trainer
# inherits it. So the exported .ply is upside down in any y-up viewer or DCC
# tool. The fix is one rigid rotation: 180 degrees about the x axis, which
# negates y and z. Positions rotate, and each Gaussian's own orientation
# quaternion must be premultiplied by that same rotation, or every ellipsoid
# ends up tilted against its neighbours. Verify the result VISUALLY: render
# before and after, and only trust the version where the trees point up.

def flip_upright(splat):
    s = {k: v.copy() for k, v in splat.items()}
    s["xyz"][:, 1] *= -1.0
    s["xyz"][:, 2] *= -1.0
    # 180-degree rotation about x as a quaternion is (w=0, x=1, y=0, z=0).
    # Premultiply: q' = r * q  with Hamilton product, r = (0, 1, 0, 0).
    w, x, y, z = (s["quat"][:, 0], s["quat"][:, 1],
                  s["quat"][:, 2], s["quat"][:, 3])
    s["quat"] = np.column_stack([-x, w, z, -y])
    lo, hi = s["xyz"].min(0), s["xyz"].max(0)
    _say("Step 2", f"rotated 180 deg about x; y now spans "
                   f"{lo[1]:.2f} to {hi[1]:.2f} (ground low, canopy high)")
    return s


# ===========================================================================
# %% Step 3. Splat -> point cloud: filter, then decode the color
# ===========================================================================
# Three moves turn Gaussians into trustworthy points. First, opacity: the file
# stores a logit, so sigmoid it back to 0..1 and drop the ghosts below 0.5.
# Second, scale: exp the stored logs and drop any Gaussian whose largest axis
# is wider than 1% of the scene diagonal, because those are the sky and fog
# balloons, not geometry. Third, color: the SH DC term is not RGB, but the
# zeroth harmonic is just a constant, so rgb = 0.5 + C0 * f_dc recovers the
# view-independent base color each Gaussian shows from every direction.

def splat_to_points(splat, opacity_min=OPACITY_MIN, scale_max_frac=SCALE_MAX_FRAC):
    n = len(splat["xyz"])
    opacity = 1.0 / (1.0 + np.exp(-splat["opacity_logit"]))
    keep_op = opacity >= opacity_min
    sigma = np.exp(splat["scales_log"])
    diag = np.linalg.norm(splat["xyz"].max(0) - splat["xyz"].min(0))
    keep_sc = sigma.max(axis=1) <= scale_max_frac * diag
    keep = keep_op & keep_sc
    rgb = np.clip(0.5 + SH_C0 * splat["f_dc"], 0.0, 1.0)
    _say("Step 3", f"opacity >= {opacity_min}: {keep_op.sum()} / {n} survive; "
                   f"scale <= {scale_max_frac:.0%} of diag ({scale_max_frac*diag:.2f} u): "
                   f"{keep_sc.sum()} / {n} survive; both: {keep.sum()}")
    return keep, rgb, sigma, opacity


# ===========================================================================
# %% Step 3b. Outdoor scenes only: peel off the sky shell
# ===========================================================================
# An outdoor splat does not stop at the geometry. The trainer explains the sky
# by wrapping the scene in a dome of blue and white Gaussians, and many of them
# are fully OPAQUE, so the opacity filter keeps them. Left in, Poisson fuses
# that dome into a blue crust over your trees. The fix reads like a weather
# report: a point is sky if it sits above the scene's median height AND its
# color says sky (a saturated blue hue, or bright with almost no saturation,
# which is a cloud). Indoor scenes skip this step entirely.

def remove_sky(splat, keep, rgb):
    from matplotlib.colors import rgb_to_hsv
    y = splat["xyz"][:, 1]
    high = y > np.median(y[keep])
    hsv = rgb_to_hsv(rgb)
    blue = (hsv[:, 0] > 0.45) & (hsv[:, 0] < 0.75) & (hsv[:, 2] > 0.45)
    cloud = (hsv[:, 1] < 0.18) & (hsv[:, 2] > 0.75)
    sky = keep & high & (blue | cloud)
    _say("Step 3b", f"sky shell: {sky.sum()} opaque sky/cloud Gaussians removed "
                    f"-> {(keep & ~sky).sum()} remain")
    return keep & ~sky


# ===========================================================================
# %% Step 4. Densify: sample each Gaussian, not just its center
# ===========================================================================
# 66k centers are too sparse for a clean surface, and they ignore what a
# Gaussian IS: a little ellipsoid of density, not a dot. So we draw a few
# samples from each surviving Gaussian's actual distribution, rotated by its
# quaternion and stretched by its scales. The splat itself tells us where its
# surface mass lives, and sampling it gives Poisson a dense, honest skin.

def densify_points(splat, keep, rgb, sigma, per_splat=SAMPLES_PER_SPLAT):
    xyz = splat["xyz"][keep]
    q = splat["quat"][keep]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    sig = sigma[keep]
    col = rgb[keep]
    n = len(xyz)
    # quaternion -> rotation matrices, vectorised
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    # samples in the Gaussian's own frame, clipped to 1.5 sigma so we thicken
    # the surface instead of blurring it
    local = np.clip(RNG.normal(0, 1, (n, per_splat, 3)), -1.5, 1.5) * sig[:, None, :]
    world = xyz[:, None, :] + np.einsum("nij,nkj->nki", R, local)
    pts = np.concatenate([xyz, world.reshape(-1, 3)], axis=0)
    cols = np.concatenate([col, np.repeat(col, per_splat, axis=0)], axis=0)
    _say("Step 4", f"{n} centers + {per_splat} samples each -> {len(pts)} points")
    return pts, cols


# ===========================================================================
# %% Step 5. Clean the cloud and give it oriented normals
# ===========================================================================
# The densified cloud still carries stragglers: samples from half-transparent
# Gaussians hovering off the surface. Statistical outlier removal measures each
# point's mean distance to its neighbours and drops the ones that sit far from
# everyone. A voxel downsample then evens the density, and normal estimation
# plus consistent orientation gives Poisson the oriented field it solves from.

def clean_cloud(pts, cols, target=VOXEL_TARGET):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    n0 = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=SOR_NEIGHBORS,
                                            std_ratio=SOR_STD)
    n1 = len(pcd.points)
    diag = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    voxel = diag / 900.0
    while len(pcd.points) > target:
        down = pcd.voxel_down_sample(voxel)
        if len(down.points) <= target:
            pcd = down
            break
        voxel *= 1.2
    radius = diag * 0.008
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    _say("Step 5", f"outliers: {n0} -> {n1}; voxel {voxel:.3f} u -> "
                   f"{len(pcd.points)} points with oriented normals (radius {radius:.3f})")
    return pcd


# ===========================================================================
# %% Step 6. Poisson surface + cleanup (with the honest alternative)
# ===========================================================================
# Screened Poisson solves one watertight surface through the oriented cloud.
# Its density field tells us which triangles the data actually supports, so we
# crop the low-density balloon, keep the largest connected component, and drop
# the degenerate leftovers. Ball pivoting is the honest alternative (faithful,
# never invents surface) but on a splat cloud, with its soft fuzzy edges, it
# leaves far too many holes to texture, and we measure that instead of claiming it.

def mesh_from_cloud(pcd, depth=POISSON_DEPTH, quantile=DENSITY_QUANTILE):
    t0 = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    dens = np.asarray(dens)
    raw = (len(mesh.vertices), len(mesh.triangles))
    keep = dens >= np.quantile(dens, quantile)
    mesh.remove_vertices_by_mask(~keep)
    cl, ntri, _ = mesh.cluster_connected_triangles()
    cl = np.asarray(cl); ntri = np.asarray(ntri)
    if len(ntri) > 0:
        mesh.remove_triangles_by_mask(cl != int(ntri.argmax()))
        mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    _say("Step 6", f"Poisson depth={depth}: {raw[0]} verts / {raw[1]} tris "
                   f"in {time.time()-t0:.1f}s -> cleaned to {len(mesh.vertices)} "
                   f"verts / {len(mesh.triangles)} tris (q={quantile})")
    return mesh


def ball_pivot_reference(pcd):
    """Measured, not asserted: how ball pivoting behaves on the same cloud."""
    d = np.mean(pcd.compute_nearest_neighbor_distance())
    radii = [d * 1.5, d * 3.0, d * 6.0]
    t0 = time.time()
    m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    _say("Step 6b", f"ball pivoting: {len(m.vertices)} verts / "
                    f"{len(m.triangles)} tris in {time.time()-t0:.1f}s")
    return m


def centers_only_reference(splat, keep, rgb):
    """Measured, not asserted: what Step 4 buys us. This runs the IDENTICAL
    clean + Poisson chain, but fed only the 56,563 Gaussian centers (no
    ellipsoid sampling). The mesh it produces is the one most quick splat
    converters ship, so meshing both clouds lets the cabin cast the verdict."""
    t0 = time.time()
    pcd = clean_cloud(splat["xyz"][keep], rgb[keep])
    mesh = mesh_from_cloud(pcd)
    _say("Step 6c", f"centers-only reference: {keep.sum()} centers -> "
                    f"{len(pcd.points)} cleaned points -> "
                    f"{len(mesh.triangles)} tris in {time.time()-t0:.1f}s total")
    return mesh


# ===========================================================================
# %% Step 7. Give the mesh its color back: vertex transfer from the cloud
# ===========================================================================
# Poisson invents vertices, so none of them carry color yet. The splat cloud
# does. A KD-tree finds each mesh vertex's three nearest cloud points and
# blends their colors by inverse distance, which reads as a clean bilinear
# paint job instead of the speckle a single-nearest lookup produces.

def transfer_colors(mesh, pcd, k=3):
    cloud = np.asarray(pcd.points)
    ccol = np.asarray(pcd.colors)
    v = np.asarray(mesh.vertices)
    dist, idx = cKDTree(cloud).query(v, k=k, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    vc = (ccol[idx] * w[..., None]).sum(axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vc, 0, 1))
    _say("Step 7", f"colors blended onto {len(v)} vertices "
                   f"(k={k} nearest, inverse-distance weights)")
    return mesh


# ===========================================================================
# %% Step 8. Bake a real texture: UV unwrap + rasterise the vertex colors
# ===========================================================================
# Vertex colors die the moment you decimate, and game engines expect a texture.
# So we unwrap the mesh with xatlas (it cuts the surface into flat islands and
# packs them into the 0..1 UV square), then rasterise every UV triangle into a
# texture image, interpolating the three corner colors barycentrically. A
# dilation pass bleeds each island's edge outward so mipmaps and bilinear
# filtering never sample the black gutter between islands.

def bake_texture(mesh, max_tris=BAKE_TRIANGLES, tex_size=TEXTURE_SIZE):
    import xatlas
    if len(mesh.triangles) > max_tris:
        mesh = mesh.simplify_quadric_decimation(max_tris)
        mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles).astype(np.uint32)
    vc = np.asarray(mesh.vertex_colors)
    t0 = time.time()
    vmap, faces, uvs = xatlas.parametrize(v.astype(np.float32), f)
    v2, c2 = v[vmap], vc[vmap]
    _say("Step 8", f"xatlas unwrap: {len(v)} -> {len(v2)} verts "
                   f"({len(faces)} tris) in {time.time()-t0:.1f}s")
    # rasterise UV triangles with barycentric color interpolation
    img = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    hit = np.zeros((tex_size, tex_size), dtype=bool)
    px = uvs * (tex_size - 1)
    for tri in faces:
        p = px[tri]                       # (3,2) in pixel space
        c = c2[tri]                       # (3,3) colors
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int) + 1
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        d = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(d) < 1e-12:
            continue
        w1 = ((xs - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (ys - p[0, 1])) / d
        w2 = ((p[1, 0] - p[0, 0]) * (ys - p[0, 1]) - (xs - p[0, 0]) * (p[1, 1] - p[0, 1])) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        col = (w0[..., None] * c[0] + w1[..., None] * c[1] + w2[..., None] * c[2])
        yy, xx = ys[inside], xs[inside]
        img[yy, xx] = col[inside]
        hit[yy, xx] = True
    # dilate island edges into the gutter (8 passes of nearest-filled copy)
    for _ in range(8):
        empty = ~hit
        if not empty.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_h = np.roll(hit, (dy, dx), axis=(0, 1))
            src_i = np.roll(img, (dy, dx), axis=(0, 1))
            fill = empty & src_h
            img[fill] = src_i[fill]
            hit |= fill
            empty = ~hit
    filled = hit.mean()
    tex = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)[::-1])
    _say("Step 8", f"baked {tex_size}x{tex_size} texture "
                   f"({filled:.0%} of texels covered after dilation)")
    return v2, np.asarray(faces, dtype=np.int64), uvs, tex


# ===========================================================================
# %% Step 9. Export the textured .glb (and what Blender does with it)
# ===========================================================================
# glTF wants y up (which the Step 2 flip already gave us), UVs with a top-left
# origin (we flipped the image instead, same result), and one material. We
# build a PBR material with the baked texture as base color, zero metallic,
# full roughness: a matte photo-textured surface. Blender's glTF importer
# converts y-up to its own z-up automatically and wires the texture into the
# Principled BSDF's Base Color, so the cabin arrives upright and painted.

def export_textured_glb(v, f, uvs, tex, out_glb):
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=1.0)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    tm = trimesh.Trimesh(vertices=v, faces=f, visual=visual, process=False)
    tm.export(out_glb)
    _say("Step 9", f"exported {os.path.basename(out_glb)} "
                   f"({len(f)} tris, {os.path.getsize(out_glb)/1e6:.2f} MB, "
                   f"{tex.size[0]}x{tex.size[1]} texture inside)")
    return out_glb


# --- helpers for the figure scripts (not article steps) ---------------------

def write_cloud(path, pts, cols):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
    o3d.io.write_point_cloud(path, pcd)


def write_flipped_splat(splat, src=SPLAT_FILE, dst=None):
    """A flipped copy of the raw splat file for the photoreal viewer renders.
    Higher-order SH terms are view-direction dependent and do not survive a
    rotation without a Wigner rotation of the coefficients, so we keep the
    (rotation-invariant) DC color and zero the rest: base colors stay exact."""
    if not os.path.exists(src):
        return None
    ply = PlyData.read(src)
    d = ply.elements[0].data.copy()
    d["y"] = -d["y"]; d["z"] = -d["z"]
    if "ny" in d.dtype.names:
        d["ny"] = -d["ny"]; d["nz"] = -d["nz"]
    q = splat["quat"]  # already flipped in Step 2
    d["rot_0"], d["rot_1"] = q[:, 0].astype(np.float32), q[:, 1].astype(np.float32)
    d["rot_2"], d["rot_3"] = q[:, 2].astype(np.float32), q[:, 3].astype(np.float32)
    for k in d.dtype.names:
        if k.startswith("f_rest_"):
            d[k] = 0.0
    el = PlyElement.describe(d, "vertex")
    PlyData([el], text=False).write(dst)
    _say("aux", f"wrote flipped splat copy for the viewer: {os.path.basename(dst)}")
    return dst


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()

    # Step 1: anatomy.
    splat = read_splat()

    # Step 2: upright.
    splat = flip_upright(splat)
    write_flipped_splat(splat, dst=os.path.join(OUT, "cabin_flipped_splat.ply"))

    # Step 3: filters + SH DC color.
    keep, rgb, sigma, opacity = splat_to_points(splat)

    # Step 3b: outdoor scene, so peel the opaque sky shell off the mask.
    keep = remove_sky(splat, keep, rgb)
    write_cloud(os.path.join(OUT, "cloud_centers.ply"),
                splat["xyz"][keep], rgb[keep])

    # Step 4: densify from the Gaussians themselves (timed as one branch:
    # sample -> clean -> Poisson, so the ellipsoid path has an honest cost).
    t_ell = time.time()
    pts, cols = densify_points(splat, keep, rgb, sigma)

    # Step 5: outliers out, normals on.
    pcd = clean_cloud(pts, cols)
    o3d.io.write_point_cloud(os.path.join(OUT, "cloud_dense_clean.ply"), pcd)

    # Step 6: surface (+ the measured ball-pivoting reference).
    mesh = mesh_from_cloud(pcd)
    ell_secs = time.time() - t_ell
    _say("Step 6", f"ellipsoid branch (densify + clean + Poisson): {ell_secs:.1f}s")
    o3d.io.write_triangle_mesh(os.path.join(OUT, "mesh_clean.ply"), mesh)
    try:
        bpa = ball_pivot_reference(pcd)
        o3d.io.write_triangle_mesh(os.path.join(OUT, "mesh_ballpivot.ply"), bpa)
    except Exception as e:
        _say("Step 6b", f"ball pivoting skipped ({str(e)[:60]})")

    # Step 7: color the vertices from the splat cloud.
    mesh = transfer_colors(mesh, pcd)
    o3d.io.write_triangle_mesh(os.path.join(OUT, "mesh_vertexcolor.ply"), mesh)

    # Step 8: unwrap + bake.
    v, f, uvs, tex = bake_texture(mesh)
    tex.save(os.path.join(OUT, "cabin_texture.png"))

    # Step 9: the asset.
    glb = export_textured_glb(v, f, uvs, tex,
                              os.path.join(OUT, "cabin_textured.glb"))

    # Fallback twin: a vertex-colored .glb for tools that skip textures.
    vc8 = None
    m2 = o3d.io.read_triangle_mesh(os.path.join(OUT, "mesh_vertexcolor.ply"))
    if m2.has_vertex_colors():
        vc8 = (np.asarray(m2.vertex_colors) * 255).astype(np.uint8)
    tm2 = trimesh.Trimesh(np.asarray(m2.vertices), np.asarray(m2.triangles),
                          vertex_colors=vc8, process=False)
    tm2.export(os.path.join(OUT, "cabin_vertexcolor.glb"))

    print(f"\n[pipeline] done in {time.time()-t_start:.0f}s. "
          f"{keep.sum()} of {len(keep)} Gaussians kept, "
          f"{len(np.asarray(mesh.vertices))} mesh verts, "
          f"{len(np.asarray(mesh.triangles))} tris, textured glb "
          f"{os.path.getsize(glb)/1e6:.2f} MB.")

    # After the pipeline proper: the centers-only reference, which measures
    # what Step 4's ellipsoid sampling buys us (same clean + Poisson chain,
    # fed only the Gaussian centers). Compare it against the ellipsoid branch.
    ref = centers_only_reference(splat, keep, rgb)
    o3d.io.write_triangle_mesh(os.path.join(OUT, "mesh_centers_only.ply"), ref)
    print(f"[compare] ellipsoid branch: {len(np.asarray(mesh.triangles))} tris "
          f"in {ell_secs:.1f}s (densify+clean+Poisson) vs centers-only: "
          f"{len(np.asarray(ref.triangles))} tris.")
