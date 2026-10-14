# -*- coding: utf-8 -*-
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
# Synthetic scans with known answers for the tests: two rooms side by side, a partition of a
# known thickness between them, options for a leaning face, a lining, a crop, a tilt, a turn.
import numpy as np

NAMES = {0: "ceiling", 1: "floor", 2: "wall", 12: "clutter"}
IDS = {v: k for k, v in NAMES.items()}
STEP = 0.035          # point spacing, like the real office (about 3 to 4 cm)
H = 3.0               # storey height
L = 5.0               # rooms run 5 m along x
D = 4.0               # each room is 4 m deep along y


def _grid(u0, u1, v0, v1, step=STEP):
    u = np.arange(u0 + step / 2, u1, step)
    v = np.arange(v0 + step / 2, v1, step)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu.ravel(), vv.ravel()


def _wall_x(x, y0, y1):            # a wall standing at x = const, normal along x
    y, z = _grid(y0, y1, 0.0, H)
    return np.c_[np.full_like(y, x), y, z]


def _wall_y(y, x0, x1, lean_deg=0.0, x_skip=None, z_top=H, skip_below=H):
    """A wall face at y = const (+ a lean: y moves by tan(lean) per metre of height)."""
    x, z = _grid(x0, x1, 0.0, z_top)
    if x_skip is not None:                      # a stretch hidden behind something, below skip_below
        keep = (x < x_skip[0]) | (x > x_skip[1]) | (z > skip_below)
        x, z = x[keep], z[keep]
    return np.c_[x, y + np.tan(np.radians(lean_deg)) * z, z]


def _slab(z, y0, y1):
    x, y = _grid(0.0, L, y0, y1)
    return np.c_[x, y, np.full_like(x, z)]


def two_rooms(t=0.12, lean_a_deg=0.0, crop_room_b=False, lining=False, tilt_deg=0.0,
              turn_deg=0.0, noise=0.002, seed=0, no_ceiling=False):
    """Room A: y in [-D, 0]; the partition: face A at y = 0, face B at y = t; room B beyond it.

    lean_a_deg   face A leans (its offset grows with height), face B stays plumb
    crop_room_b  room B is cut away 5 cm behind face B (a scan cropped at the partition)
    lining       a 1.2 m high panel 8 cm in front of room A's back wall, over x in [1, 3];
                 the back wall is hidden behind it and seen only above it (a dado, a cabinet)
    tilt_deg     the whole scan tilted about the x axis; turn_deg turned about z
    """
    rng = np.random.default_rng(seed)
    parts, cls = [], []

    def add(p, c):
        parts.append(p)
        cls.append(np.full(len(p), IDS[c]))

    # room A
    add(_slab(0.0, -D, 0.0), "floor")
    if not no_ceiling:
        add(_slab(H, -D, 0.0), "ceiling")
    add(_wall_x(0.0, -D, 0.0), "wall")
    add(_wall_x(L, -D, 0.0), "wall")
    add(_wall_y(-D, 0.0, L, x_skip=(1.0, 3.0) if lining else None, skip_below=1.2), "wall")
    if lining:
        add(_wall_y(-D + 0.08, 1.0, 3.0, z_top=1.2), "wall")
    add(_wall_y(0.0, 0.0, L, lean_deg=lean_a_deg), "wall")          # face A of the partition
    # a few chairs' worth of clutter so a room has something standing in it
    add(np.c_[rng.uniform(0.5, 4.5, 800), rng.uniform(-3.5, -0.8, 800), rng.uniform(0.0, 0.9, 800)],
        "clutter")
    # room B
    add(_wall_y(t, 0.0, L), "wall")                                  # face B of the partition
    if not crop_room_b:
        add(_slab(0.0, t, t + D), "floor")
        if not no_ceiling:
            add(_slab(H, t, t + D), "ceiling")
        add(_wall_x(0.0, t, t + D), "wall")
        add(_wall_x(L, t, t + D), "wall")
        add(_wall_y(t + D, 0.0, L), "wall")
        add(np.c_[rng.uniform(0.5, 4.5, 800), rng.uniform(t + 0.8, t + 3.5, 800),
                  rng.uniform(0.0, 0.9, 800)], "clutter")
    xyz = np.concatenate(parts)
    cls = np.concatenate(cls)
    xyz = xyz + rng.normal(0.0, noise, xyz.shape)
    if tilt_deg:
        a = np.radians(tilt_deg)
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        xyz = xyz @ R.T
    if turn_deg:
        a = np.radians(turn_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        xyz = xyz @ R.T
    return xyz, cls, dict(NAMES), dict(IDS)
