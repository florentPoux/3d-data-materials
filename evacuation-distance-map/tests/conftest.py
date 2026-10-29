# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-build-a-2d-map-and-3d-evacuation-simulation-from-3d-scan-data-python-a4d7e71c0f0c
# Dataset:  https://learngeodata.eu/materials/evacuation-distance-map/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""Shared helpers for the tests: synthetic floors with answers known in advance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evacuation_distance as ev  # noqa: E402,F401


def make_grid(walk):
    """A grid dict around a hand-made walkable mask (cell (0, 0) centred at (0.025, 0.025) m)."""
    walk = np.asarray(walk, bool)
    return dict(shape=walk.shape, walk=walk, origin=np.zeros(2))


def make_src(grid, exits):
    """exits: {name: list of (row, col)}; each cell joined at distance 0 (the door is on the cell)."""
    W = grid["shape"][1]
    return {e: dict(cells=np.array([r * W + c for r, c in cells]), offset=np.zeros(len(cells)),
                    leaf=np.array([], int)) for e, cells in exits.items()}


def solve_field(walk, exits, neighbours=16, open_exits=None):
    grid = make_grid(walk)
    src = make_src(grid, exits)
    G, idx = ev.walking_graph(grid["walk"], neighbours)
    Ga, node = ev.attach_exits(G, idx, src)
    D, serve, _, leave = ev.distance_field(grid, Ga, idx, src, open_exits, node)
    return grid, src, Ga, idx, node, D, serve, leave
