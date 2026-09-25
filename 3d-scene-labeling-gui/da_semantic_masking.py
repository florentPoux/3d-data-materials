# -*- coding: utf-8 -*-
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-build-a-python-gui-for-3d-scene-labeling-49dd43624a7f
# Dataset:  https://learngeodata.eu/materials/3d-scene-labeling-gui/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""
Tutorial 2: Interactive Semantic Masking & Label Fusion
=======================================================

This tutorial shows how to:
1. Load 3D reconstruction data exported by Tutorial 1
2. Interactively paint HD masks on images
3. Project 2D masks to 3D point clouds
4. Create multi-frame labels for full scenes
5. Propagate labels with smart KD-Tree fusion

Prerequisites:
- reconstruction_data.npz next to this script: download it from the kit page,
  or make your own with Tutorial 1 (da_3d_reconstruction.py)

Dependencies:
- numpy, open3d, matplotlib, scipy

Author: Florent Poux
License: MIT (see the SPDX line above)
"""

#%% Initialization: Environment Setup

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from types import SimpleNamespace
import trimesh

from interactive_painting import paint_mask_hd, mask_multiple_images_hd

#%% Step 1: Load Reconstruction Data from Tutorial 1

SCENE = "KIDS"
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
results_dir = os.path.join(HERE, "results", SCENE)

paths = {
    'data': os.path.join(HERE, "data", SCENE),
    'results': results_dir,
    'masks': os.path.join(results_dir, "masks"),
}
os.makedirs(paths['masks'], exist_ok=True)

data = np.load(os.path.join(HERE, "reconstruction_data.npz"))
prediction = SimpleNamespace(
    depth=data['depth'],
    conf=data['conf'],
    intrinsics=data['intrinsics'],
    extrinsics=data['extrinsics'],
    processed_images=data['processed_images'],
)
points_3d = data['points_3d']
colors_3d = data['colors_3d']

print(f"Loaded {len(points_3d)} points, {len(prediction.depth)} frames")
print(f"Image resolution: {prediction.processed_images[0].shape[:2]}")

#%% Step 2: Apply HD Masking on Single Image

sample_image_hd = prediction.processed_images[0]
hd_mask = paint_mask_hd(sample_image_hd, num_classes=5, target_height=1080)
print(f"HD mask created: {hd_mask.shape}, Classes: {np.unique(hd_mask)}")

def visualize_hd_mask(image, mask, title="HD Mask Overlay"):
    """Show the original image, mask classes, and mask overlay side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title('Mask Classes')
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(mask, alpha=0.5, cmap='tab10')
    axes[2].set_title('Mask Overlay')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

visualize_hd_mask(sample_image_hd, hd_mask, "Step 5b: HD Mask on Single Image")

#%% Step 3: Project to 3D Point Cloud

def project_mask_to_3d(mask, depth_map, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Project 2D mask to 3D points using camera geometry"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter by confidence
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, mask = u[valid_mask], v[valid_mask], depth_map[valid_mask], mask[valid_mask]
    else:
        u, v, depth_map, mask = u.flatten(), v.flatten(), depth_map.flatten(), mask.flatten()

    # Back-project to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    points_cam = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R

    return points_world, mask

def visualize_projected_mask_3d(points, mask_labels, window_name="Projected Mask 3D"):
    """Visualize 3D points colored by mask class labels"""
    class_colormap = np.array([
        [0.3, 0.3, 0.3],  # 0: Unlabeled (dark gray)
        [1.0, 0.0, 0.0],  # 1: Red
        [0.0, 1.0, 0.0],  # 2: Green
        [0.0, 0.0, 1.0],  # 3: Blue
        [1.0, 1.0, 0.0],  # 4: Yellow
        [1.0, 0.0, 1.0],  # 5: Magenta
    ])
    colors = class_colormap[np.clip(mask_labels, 0, len(class_colormap) - 1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

# Project single-frame mask to 3D and visualize
projected_pts, projected_labels = project_mask_to_3d(
    hd_mask, prediction.depth[0], prediction.intrinsics[0],
    prediction.extrinsics[0], prediction.conf[0], conf_thresh=0.4
)
visualize_projected_mask_3d(projected_pts, projected_labels, "Step 5: Single Frame Mask Projected to 3D")

#%% Step 4: Create Proper Multi-Frame Labels for Full Scene

def create_full_scene_labels(points_3d, prediction, masks, frame_indices, conf_thresh=0.5):
    """
    Create label array for full scene by projecting multi-frame masks
    Returns labels aligned with full points_3d array (handles multi-frame properly)
    """
    print(f"\n=== Creating Full Scene Labels (Multi-Frame) ===")

    # Initialize all labels as unlabeled (0)
    all_labels = np.zeros(len(points_3d), dtype=np.int32)

    # Build KD-tree for full scene
    kdtree = cKDTree(points_3d)

    for i, mask_idx in enumerate(frame_indices):
        if mask_idx >= len(masks):
            continue

        # Project this frame's mask to 3D
        frame_points, frame_labels = project_mask_to_3d(
            masks[mask_idx],
            prediction.depth[mask_idx],
            prediction.intrinsics[mask_idx],
            prediction.extrinsics[mask_idx],
            prediction.conf[mask_idx],
            conf_thresh
        )

        # Only keep labeled points
        labeled_mask = frame_labels > 0
        labeled_frame_points = frame_points[labeled_mask]
        labeled_frame_labels = frame_labels[labeled_mask]

        if len(labeled_frame_points) == 0:
            continue

        # Find nearest neighbors in full scene (within 1cm tolerance)
        distances, indices = kdtree.query(labeled_frame_points, k=1, distance_upper_bound=0.01)

        # Assign labels to matched points
        valid_matches = distances < 0.01
        matched_indices = indices[valid_matches]
        matched_labels = labeled_frame_labels[valid_matches]

        all_labels[matched_indices] = matched_labels

        print(f"  Frame {mask_idx}: Assigned {len(matched_indices)} labels")

    labeled_count = np.sum(all_labels > 0)
    print(f"Total labeled points: {labeled_count} ({100*labeled_count/len(points_3d):.2f}%)")
    return all_labels

def visualize_scene_labels(points_3d, labels, window_name="Scene Labels"):
    """Visualize full scene point cloud colored by class labels"""
    class_colormap = np.array([
        [0.7, 0.7, 0.7],  # 0: Unlabeled (light gray)
        [1.0, 0.0, 0.0],  # 1: Red
        [0.0, 1.0, 0.0],  # 2: Green
        [0.0, 0.0, 1.0],  # 3: Blue
        [1.0, 1.0, 0.0],  # 4: Yellow
        [1.0, 0.0, 1.0],  # 5: Magenta
    ])
    colors = class_colormap[np.clip(labels, 0, len(class_colormap) - 1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

#%% Step 5: Smart Label Fusion with KD-Tree

def compute_camera_distances(points_3d, camera_positions):
    """Calculate minimum distance from each point to any camera"""
    distances = np.full(len(points_3d), np.inf)

    for cam_pos in camera_positions:
        point_distances = np.linalg.norm(points_3d - cam_pos, axis=1)
        distances = np.minimum(distances, point_distances)

    return distances

def extract_camera_positions(extrinsics_array):
    """Extract camera positions from extrinsics matrices (w2c format)"""
    camera_positions = []

    for ext in extrinsics_array:
        R = ext[:3, :3]
        t = ext[:3, 3]
        # Camera position in world coordinates: -R^T @ t
        cam_pos = -R.T @ t
        camera_positions.append(cam_pos)

    return np.array(camera_positions)

def smart_label_fusion(points_3d, labels_3d, camera_positions,
                       max_distance=0.15, max_camera_dist=5.0, min_neighbors=3,
                       batch_size=50000):
    """
    Memory-efficient label propagation with batched processing
    """
    print(f"\n=== Smart Label Fusion (Memory-Efficient) ===")
    print(f"Input: {len(points_3d)} points, {np.sum(labels_3d > 0)} labeled")

    # Step 1: Filter labels by camera distance (remove noise)
    print("Step 1 - Filtering by camera distance...")
    camera_distances = compute_camera_distances(points_3d, camera_positions)
    valid_distance_mask = camera_distances <= max_camera_dist

    # Keep only labels close to cameras
    filtered_labels = labels_3d.copy()
    filtered_labels[~valid_distance_mask] = 0

    noise_removed = np.sum(labels_3d > 0) - np.sum(filtered_labels > 0)
    print(f"  Removed {noise_removed} distant labels")

    # Step 2: Build KD-tree only with LABELED points (much smaller tree)
    print("Step 2 - Building KD-tree with labeled points only...")
    labeled_mask = filtered_labels > 0
    labeled_points = points_3d[labeled_mask]
    labeled_values = filtered_labels[labeled_mask]

    if len(labeled_points) == 0:
        print("No labeled points to propagate from")
        return filtered_labels, camera_distances

    kdtree = cKDTree(labeled_points)
    print(f"  KD-tree built with {len(labeled_points)} labeled points")

    # Step 3: Find unlabeled points
    unlabeled_mask = filtered_labels == 0
    unlabeled_indices = np.where(unlabeled_mask)[0]

    print(f"Step 3 - Found {len(unlabeled_indices)} unlabeled points")

    if len(unlabeled_indices) == 0:
        print("No unlabeled points to propagate to")
        return filtered_labels, camera_distances

    # Step 4: Batched label propagation (prevents memory overflow)
    print(f"Step 4 - Label propagation in batches of {batch_size}...")
    fused_labels = filtered_labels.copy()
    propagated_count = 0

    num_batches = (len(unlabeled_indices) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(unlabeled_indices))

        batch_indices = unlabeled_indices[start_idx:end_idx]
        batch_points = points_3d[batch_indices]

        # Query neighbors for this batch (memory stays bounded)
        neighbor_indices_list = kdtree.query_ball_point(batch_points, r=max_distance)

        # Process each point in batch
        for i, neighbor_indices in enumerate(neighbor_indices_list):
            if len(neighbor_indices) < min_neighbors:
                continue

            # Get labels of neighbors (from labeled_values)
            neighbor_labels = labeled_values[neighbor_indices]

            # Majority vote using bincount
            label_counts = np.bincount(neighbor_labels)
            most_common_label = np.argmax(label_counts)

            fused_labels[batch_indices[i]] = most_common_label
            propagated_count += 1

        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            print(f"  Processed batch {batch_idx+1}/{num_batches} ({propagated_count} labels propagated)")

    print(f"Step 5 - Label propagation complete")
    print(f"  Added {propagated_count} new labels")
    print(f"  Output: {np.sum(fused_labels > 0)} total labeled points")

    return fused_labels, camera_distances

def visualize_fusion_comparison(points_3d, labels_before, labels_after, camera_positions):
    """Show before/after comparison of label fusion"""
    class_colormap = np.array([
        [0.8, 0.8, 0.8],  # 0: Dark gray (unlabeled)
        [1.0, 0.0, 0.0],  # 1: Red
        [0.0, 1.0, 0.0],  # 2: Green
        [0.0, 0.0, 1.0],  # 3: Blue
        [1.0, 1.0, 0.0],  # 4: Yellow
        [1.0, 0.0, 1.0],  # 5: Magenta
    ])

    # Before fusion
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(points_3d)
    pcd_before.colors = o3d.utility.Vector3dVector(class_colormap[labels_before])

    # After fusion
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(points_3d)
    pcd_after.colors = o3d.utility.Vector3dVector(class_colormap[labels_after])
    pcd_after.estimate_normals()

    # Camera positions as small spheres
    camera_spheres = []
    for cam_pos in camera_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(cam_pos)
        sphere.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
        camera_spheres.append(sphere)

    print("\n=== Visualization ===")
    print("Showing BEFORE fusion (close window to see AFTER)")
    o3d.visualization.draw_geometries([pcd_before] + camera_spheres,
                                     window_name="Before Smart Fusion")

    print("Showing AFTER fusion")
    o3d.visualization.draw_geometries([pcd_after] + camera_spheres,
                                     window_name="After Smart Fusion")

def analyze_fusion_statistics(labels_before, labels_after):
    """Print detailed statistics about fusion results"""
    print(f"\n=== Fusion Statistics ===")

    for class_id in range(1, 6):
        before_count = np.sum(labels_before == class_id)
        after_count = np.sum(labels_after == class_id)
        increase = after_count - before_count

        if before_count > 0:
            percent_increase = (increase / before_count) * 100
            print(f"Class {class_id}: {before_count} → {after_count} points (+{increase}, +{percent_increase:.1f}%)")

    total_before = np.sum(labels_before > 0)
    total_after = np.sum(labels_after > 0)
    coverage_before = (total_before / len(labels_before)) * 100
    coverage_after = (total_after / len(labels_after)) * 100

    print(f"\nTotal labeled: {total_before} → {total_after}")
    print(f"Scene coverage: {coverage_before:.1f}% → {coverage_after:.1f}%")

#%% Step 6: Complete Workflow: HD masking → multi-frame labels → smart fusion

print("\n=== Step 10 Complete Workflow ===")
num_frames_to_mask = min(5, len(prediction.processed_images))
print(f"Masking {num_frames_to_mask} frames in HD...")

hd_multi_masks = mask_multiple_images_hd(
    prediction.processed_images,
    num_images=num_frames_to_mask,
    target_height=1080
)

frame_indices = list(range(num_frames_to_mask))
all_labels_3d = create_full_scene_labels(
    points_3d,
    prediction,
    hd_multi_masks,
    frame_indices,
    conf_thresh=0.4
)

camera_positions = extract_camera_positions(prediction.extrinsics)
print(f"\nExtracted {len(camera_positions)} camera positions")

# Apply memory-efficient smart fusion
fused_labels, camera_distances = smart_label_fusion(
    points_3d,
    all_labels_3d,
    camera_positions,
    max_distance=0.05,      # 15cm neighborhood
    max_camera_dist=5.0,    # Filter labels beyond 5m
    min_neighbors=3,        # Need 3+ labeled neighbors
    batch_size=100000        # Process 50k points at once (reduce if memory issues)
)

# Visualize scene labels before and after fusion
visualize_scene_labels(points_3d, all_labels_3d, "Step 8: Scene Labels Before Fusion")
visualize_scene_labels(points_3d, fused_labels, "Step 8: Scene Labels After Fusion")

#%% Step 7: Visualize before/after comparison
visualize_fusion_comparison(points_3d, all_labels_3d, fused_labels, camera_positions)

#%% Step 8: Analyze fusion results
analyze_fusion_statistics(all_labels_3d, fused_labels)

#%% Step 9: Export fused results

def save_point_cloud_as_ply(points, colors, output_path, labels=None):
    """Export point cloud to PLY format with optional segment labels"""
    if labels is not None:
        n_points = len(points)
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int segment_label
end_header
"""
        colors_uint8 = (colors * 255).astype(np.uint8)

        with open(output_path, 'wb') as f:
            f.write(header.encode('ascii'))
            for i in range(n_points):
                f.write(points[i].astype(np.float32).tobytes())
                f.write(colors_uint8[i].tobytes())
                f.write(np.array([labels[i]], dtype=np.int32).tobytes())

        print(f"Saved labeled point cloud to: {output_path}")
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud to: {output_path}")


output_fused_ply = os.path.join(paths['results'], "smart_fused_labels-v2.ply")
save_point_cloud_as_ply(points_3d, colors_3d, output_fused_ply, labels=fused_labels)

# Final visualization of exported result
visualize_scene_labels(points_3d, fused_labels, "Export: Final Labeled Point Cloud")


#%% Step 10: GLB Export Utilities

def _as_homogeneous44(ext):
    """Accept (4,4) or (3,4) extrinsic matrix, return (4,4) homogeneous."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def _hsv_to_rgb(h, s, v):
    """Pure-Python HSV to RGB conversion (no colorsys dependency)."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    else:
        return v, p, q


def _estimate_normals(points, k=30, orient_toward=None):
    """Estimate point cloud normals using Open3D's KNN-based PCA.

    Args:
        points:        (N, 3) point positions.
        k:             Number of nearest neighbors for local covariance.
        orient_toward: If provided, orient all normals toward this 3D point.

    Returns:
        (N, 3) unit normals.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(k, len(points)))
    )
    if orient_toward is not None:
        pcd.orient_normals_towards_camera_location(camera_location=orient_toward)
    return np.asarray(pcd.normals).copy()


def _points_to_normal_mesh(points, normals, colors_rgba):
    """Convert an oriented point cloud into a Trimesh of micro-triangles.

    Each point becomes a tiny equilateral triangle (edge ≈ 2e-4 units)
    oriented along its normal.  This ensures vertex normals are written to
    the GLB NORMAL accessor while keeping a point-cloud-like appearance.
    """
    n = len(points)
    eps = 1e-4

    # Two tangent vectors per normal via cross product
    ref = np.tile([0.0, 0.0, 1.0], (n, 1))
    t1 = np.cross(normals, ref)
    degen = np.linalg.norm(t1, axis=1) < 1e-6
    if degen.any():
        t1[degen] = np.cross(normals[degen], [[1.0, 0.0, 0.0]])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)

    # Three vertices per point
    v0 = points + eps * t1
    v1 = points - 0.5 * eps * t1 + (0.866 * eps) * t2
    v2 = points - 0.5 * eps * t1 - (0.866 * eps) * t2

    vertices = np.empty((n * 3, 3), dtype=np.float64)
    vertices[0::3] = v0
    vertices[1::3] = v1
    vertices[2::3] = v2

    vert_normals = np.repeat(normals, 3, axis=0)
    vert_colors = np.repeat(colors_rgba, 3, axis=0)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vert_normals,
        process=False,
    )
    mesh.visual.vertex_colors = vert_colors
    return mesh


def export_labeled_glb(points_3d, labels, prediction, output_path,
                        camera_size=0.03, estimate_normals=True):
    """Export a labeled point cloud with camera frustums to a GLB file.

    Replicates the DA3 ``export_to_glb`` pipeline with semantic-class coloring
    instead of per-point RGB.  The scene is aligned to glTF conventions
    (X-right, Y-up, Z-backward) relative to the first camera.

    When *estimate_normals* is True (the default), normals are computed via
    Open3D's KNN-based PCA and the point cloud is emitted as micro-triangles
    so that the GLB file contains a proper NORMAL accessor.  This triples the
    vertex count but the triangles are invisible at normal zoom (~1e-4 units).

    Args:
        points_3d:        (N, 3) world-space point positions.
        labels:           (N,) int32 per-point class labels (0 = unlabeled).
        prediction:       SimpleNamespace with .intrinsics, .extrinsics, .depth.
        output_path:      Destination file path (e.g. ``semantic_scene.glb``).
        camera_size:      Frustum scale as a fraction of the scene diagonal.
        estimate_normals: Compute and bake vertex normals into the GLB (default True).
    """
    # ---- 1. Class colormap (RGBA uint8) ----
    class_colormap = np.array([
        [178, 178, 178, 255],  # 0: unlabeled, gray
        [255,   0,   0, 255],  # 1: red
        [  0, 255,   0, 255],  # 2: green
        [  0,   0, 255, 255],  # 3: blue
        [255, 255,   0, 255],  # 4: yellow
        [255,   0, 255, 255],  # 5: magenta
    ], dtype=np.uint8)
    rgba = class_colormap[np.clip(labels, 0, len(class_colormap) - 1)]

    # ---- 2. Alignment matrix: T_center @ M @ w2c0 ----
    w2c0 = _as_homogeneous44(prediction.extrinsics[0]).astype(np.float64)

    # CV → glTF axis flip (Y=-1, Z=-1)
    M = np.eye(4, dtype=np.float64)
    M[1, 1] = -1.0
    M[2, 2] = -1.0

    A_no_center = M @ w2c0

    # Center on the median of the transformed points
    if len(points_3d) > 0:
        pts_tmp = trimesh.transform_points(points_3d, A_no_center)
        center = np.median(pts_tmp, axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    T_center = np.eye(4, dtype=np.float64)
    T_center[:3, 3] = -center

    A = T_center @ A_no_center

    # ---- 3. Point cloud ----
    aligned_pts = trimesh.transform_points(points_3d, A) if len(points_3d) > 0 else points_3d
    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = A

    if len(aligned_pts) > 0:
        if estimate_normals:
            normals = _estimate_normals(aligned_pts, k=30, orient_toward=np.zeros(3))
            pc = _points_to_normal_mesh(aligned_pts, normals, rgba)
        else:
            pc = trimesh.points.PointCloud(vertices=aligned_pts, colors=rgba)
        scene.add_geometry(pc)

    # ---- 4. Camera frustums ----
    N_cams = len(prediction.extrinsics)
    H, W = prediction.depth.shape[1:]

    # Scene scale for frustum sizing
    if len(aligned_pts) >= 2:
        lo = np.percentile(aligned_pts, 5, axis=0)
        hi = np.percentile(aligned_pts, 95, axis=0)
        scene_scale = float(np.linalg.norm(hi - lo))
        if not (np.isfinite(scene_scale) and scene_scale > 0):
            scene_scale = 1.0
    else:
        scene_scale = 1.0
    frustum_scale = scene_scale * camera_size

    for i in range(N_cams):
        K_inv = np.linalg.inv(prediction.intrinsics[i])
        c2w = np.linalg.inv(_as_homogeneous44(prediction.extrinsics[i]))

        # Camera center in world
        Cw = (c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

        # Image corners → rays → scale to frustum depth
        corners = np.array([
            [0, 0, 1.0],
            [W - 1, 0, 1.0],
            [W - 1, H - 1, 1.0],
            [0, H - 1, 1.0],
        ], dtype=np.float64)

        rays = (K_inv @ corners.T).T
        z = rays[:, 2:3]
        z[z == 0] = 1.0
        plane_cam = (rays / z) * frustum_scale

        # To world coordinates
        plane_w = np.array([
            (c2w @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in plane_cam
        ])

        # 8 line segments: 4 center→corner + 4 rectangle edges
        segs = []
        for k in range(4):
            segs.append(np.stack([Cw, plane_w[k]], axis=0))
        order = [0, 1, 2, 3, 0]
        for a, b in zip(order[:-1], order[1:]):
            segs.append(np.stack([plane_w[a], plane_w[b]], axis=0))
        segs = np.stack(segs, axis=0)  # (8, 2, 3)

        # Apply alignment transform
        segs = trimesh.transform_points(segs.reshape(-1, 3), A).reshape(-1, 2, 3)

        # HSV-based per-camera color
        h = (i + 0.5) / max(N_cams, 1)
        r, g, b = _hsv_to_rgb(h, 0.85, 0.95)
        cam_color = (np.array([r, g, b]) * 255).astype(np.uint8)

        path = trimesh.load_path(segs)
        if hasattr(path, "colors"):
            path.colors = np.tile(cam_color, (len(path.entities), 1))
        scene.add_geometry(path)

    # ---- 5. Export ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scene.export(output_path)
    print(f"Labeled GLB exported: {output_path}")


#%% Step 11: Export Labeled Scene as GLB

glb_output = os.path.join(paths['results'], "semantic_scene.glb")
export_labeled_glb(points_3d, fused_labels, prediction, glb_output, camera_size=0.02, estimate_normals=False)
