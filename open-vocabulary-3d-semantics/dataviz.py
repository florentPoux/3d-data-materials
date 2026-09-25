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
# The loader and figure helpers that open_vocab_3d_semantics.py imports as `dv`.
# ---------------------------------------------------------------------------
#
# This is the kit's own copy of the shared pillar toolkit, cut down to what the
# script uses. One behaviour changed on purpose: the shared toolkit quietly
# swapped in a synthetic cloud when a file was missing or a library was not
# installed, so a reader could get plausible figures from fake data. Here a
# missing file or a missing library stops the run with the fix in the message.
#
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House palette (the article's figures).
CYAN, GOLD, TEXT, MUTED, SURFACE = "#0078D4", "#E8A800", "#1B1F23", "#9AA0A6", "#F6F8FA"
GREEN, RED = "#2E9E5B", "#E84040"
LABEL_CMAP = plt.get_cmap("tab20")
_RNG = np.random.default_rng(0)

plt.rcParams.update({"axes.edgecolor": "#D0D7DE", "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "figure.dpi": 110})


# ---- loading -------------------------------------------------------------

def load_cloud(path, max_points=40000, want_labels=False):
    """Load a .ply point cloud and subsample it to max_points (seeded, so every
    run reads the same points). Returns a dict with xyz (N,3 float), rgb (N,3 in
    0..1 or None), labels (N int or None) and source.

    Fails loudly: no file, no plyfile, or no label field when labels are asked
    for all stop the run. Nothing is ever replaced by synthetic data."""
    if not os.path.isfile(path):
        sys.exit(f"[dataviz] ERROR: point cloud not found: {path}\n"
                 f"          Download room.ply from the kit page and put it next to "
                 f"the script (or set DATA_FILE).")
    if not path.lower().endswith(".ply"):
        sys.exit(f"[dataviz] ERROR: {os.path.basename(path)} is not a .ply file.")
    out = _load_ply(path, want_labels)
    return _subsample(out, max_points, source=os.path.basename(path))


def _load_ply(path, want_labels):
    try:
        from plyfile import PlyData
    except ImportError:
        sys.exit("[dataviz] ERROR: the plyfile package is not installed.\n"
                 "          pip install plyfile   (or: pip install -r requirements.txt)")
    el = PlyData.read(path).elements[0].data
    names = el.dtype.names
    xyz = np.column_stack([el["x"], el["y"], el["z"]]).astype(np.float64)
    rgb = None
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.column_stack([el["red"], el["green"], el["blue"]]).astype(np.float64)
        rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
    labels = None
    if want_labels:
        for key in ("classification", "label", "scalar_semantic_label", "scalar_Classification"):
            if key in names:
                labels = np.asarray(el[key]).astype(int)
                break
        if labels is None:
            sys.exit(f"[dataviz] ERROR: {os.path.basename(path)} has no per-point label "
                     f"field (looked for classification, label). Fields: {list(names)}")
    return {"xyz": xyz, "rgb": rgb, "labels": labels}


def _subsample(d, max_points, source):
    n = len(d["xyz"])
    if n > max_points:
        idx = np.sort(_RNG.choice(n, max_points, replace=False))
        d = {"xyz": d["xyz"][idx],
             "rgb": d["rgb"][idx] if d["rgb"] is not None else None,
             "labels": d["labels"][idx] if d["labels"] is not None else None}
    d["source"] = source
    return d


# ---- figure helpers ------------------------------------------------------

def _true_aspect(xyz):
    """Box aspect proportional to real extents, so geometry is never stretched."""
    rng = xyz.max(0) - xyz.min(0)
    rng = np.where(rng < 1e-9, 1.0, rng)
    return tuple(rng / rng.max())


def _zoom_lims(xyz, pct=1.5):
    """Per-axis limits clipped to the [pct, 100-pct] percentile band, so a few
    stray points do not shrink the scene to a blob in the middle of the frame."""
    lo = np.percentile(xyz, pct, axis=0)
    hi = np.percentile(xyz, 100 - pct, axis=0)
    pad = (hi - lo) * 0.04 + 1e-6
    return lo - pad, hi + pad


def _style_ax3d(ax, title, xyz=None, zoom=1.32):
    if title:
        ax.set_title(title, color=TEXT, fontsize=12.5, pad=2, loc="left")
    ax.set_axis_off()
    if xyz is not None:
        lo, hi = _zoom_lims(xyz)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        try:                                   # zoom kwarg: matplotlib >= 3.6
            ax.set_box_aspect(_true_aspect(xyz), zoom=zoom)
        except TypeError:
            ax.set_box_aspect(_true_aspect(xyz))


def _fill(fig, is3d=False):
    if is3d:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96)
    else:
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.90)


def _say_saved(out_path):
    print(f"[dataviz] saved {os.path.basename(out_path)}")


def save(fig, out_path, is3d=False):
    """Save with a subtle brand mark and a tight crop."""
    _fill(fig, is3d=is3d)
    fig.text(0.99, 0.012, "learngeodata.eu", ha="right", va="bottom",
             fontsize=7, color=MUTED, alpha=0.8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    _say_saved(out_path)
    return out_path


def scatter3d(xyz, out_path, title="", rgb=None, labels=None, point_size=4,
              legend_names=None, elev=24, azim=-62, zoom=1.32):
    """3D scatter coloured by rgb, by labels, or by height."""
    fig = plt.figure(figsize=(8, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    kw = dict(s=point_size, linewidths=0, depthshade=False, rasterized=True)
    if rgb is not None:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.clip(rgb, 0, 1), **kw)
    elif labels is not None:
        for i, u in enumerate(np.unique(labels)):
            m = labels == u
            name = legend_names.get(int(u), f"class {u}") if legend_names else f"class {u}"
            ax.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2],
                       color=LABEL_CMAP(i % 20), label=name, **kw)
        ax.legend(loc="upper left", bbox_to_anchor=(0.98, 1.0), fontsize=8,
                  markerscale=2.5, framealpha=0.0)
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="viridis", **kw)
    _style_ax3d(ax, title, xyz, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    return save(fig, out_path, is3d=True)


def compare3d(xyz_a, xyz_b, out_path, title="", name_a="A", name_b="B",
              point_size=3, elev=24, azim=-62, zoom=1.28):
    """Two clouds side by side (before / after), each coloured by height."""
    fig = plt.figure(figsize=(10.5, 5.0))
    for k, (xyz, name) in enumerate([(xyz_a, name_a), (xyz_b, name_b)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="viridis",
                   s=point_size, linewidths=0, depthshade=False, rasterized=True)
        _style_ax3d(ax, name, xyz, zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, x=0.02, ha="left")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92, wspace=0.02)
    fig.text(0.99, 0.012, "learngeodata.eu", ha="right", va="bottom",
             fontsize=7, color=MUTED, alpha=0.8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    _say_saved(out_path)
    return out_path


def bars(labels, values, out_path, title="", color=CYAN, ylabel="", annotate=True):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = color if isinstance(color, list) else [color] * len(values)
    b = ax.bar(range(len(values)), values, color=cols, zorder=3)
    if annotate:
        for rect, v in zip(b, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=13, loc="left"); ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#E1E4E8", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, out_path)


def heatmap(mat, out_path, title="", cmap="plasma", xlabel="", ylabel="", cbar_label=""):
    """2D field: a depth map, a confusion matrix, an affinity grid."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(mat, cmap=cmap, origin="upper", aspect="auto")
    ax.set_title(title, color=TEXT, fontsize=13, loc="left")
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    return save(fig, out_path)


def new_fig(title="", figsize=(8, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, color=TEXT, fontsize=13, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
