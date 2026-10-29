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
"""The walkable grid: known widths in, known widths out."""
import numpy as np
from scipy import ndimage as ndi

from conftest import ev


def corridor_cloud(width, length=6.0, door=None, step=0.02):
    """A straight corridor along x, walls along both sides, floor between them.
    door=(x0, x1): the north wall has a door leaf there instead of wall, with a room behind it."""
    xs = np.arange(0.0, length, step)
    ys = np.arange(0.0, width, step) + step / 2
    fx, fy = np.meshgrid(xs, ys)
    pts, roles = [np.c_[fx.ravel(), fy.ravel(), np.zeros(fx.size)]], [np.full(fx.size, "floor", object)]
    zs = np.arange(0.3, 1.6, 0.1)
    for y0 in (-0.08, width + 0.08):
        wx, wz = np.meshgrid(xs, zs)
        wy = np.full(wx.size, y0)
        r = np.full(wx.size, "wall", object)
        if door is not None and y0 > 0:
            r[(wx.ravel() >= door[0]) & (wx.ravel() < door[1])] = "door"
        pts.append(np.c_[wx.ravel(), wy, wz.ravel()])
        roles.append(r)
    if door is not None:                                       # the room behind the door
        bx, by = np.meshgrid(np.arange(door[0] - 1, door[1] + 1, step), np.arange(width + 0.2, width + 2.5, step))
        pts.append(np.c_[bx.ravel(), by.ravel(), np.zeros(bx.size)])
        roles.append(np.full(bx.size, "floor", object))
    return np.vstack(pts), np.concatenate(roles)


def column(grid, x):
    return int(np.floor((x - grid["origin"][0]) / ev.CELL))


def walk_width(grid, x):
    """Walkable width (m) across the corridor at abscissa x."""
    return grid["walk"][:, column(grid, x)].sum() * ev.CELL


def test_corridor_walkable_width_is_width_minus_a_body():
    """A 1.60 m corridor leaves a 1.10 m strip where a body's centre fits: 1.60 - 2 x 0.25."""
    xyz, role = corridor_cloud(1.60)
    g = ev.build_walkable(xyz, role)
    assert abs(walk_width(g, 3.0) - (1.60 - 2 * ev.BODY_R)) <= ev.CELL + 1e-9


def test_clearance_is_measured_to_the_obstacle_edge():
    """Half a cell matters: the clearance in the middle of a 1.60 m corridor is 0.80 m."""
    xyz, role = corridor_cloud(1.60)
    g = ev.build_walkable(xyz, role)
    assert abs(g["clearance"][:, column(g, 3.0)].max() - 0.80) <= ev.CELL / 2 + 1e-9


def test_narrow_corridor_is_not_walkable():
    xyz, role = corridor_cloud(0.45)
    g = ev.build_walkable(xyz, role)
    assert walk_width(g, 3.0) == 0


def test_door_opens_and_a_too_narrow_door_does_not():
    """A 0.90 m door leaf in a wall lets a body through; a 0.40 m one does not."""
    for w, passes in ((0.90, True), (0.40, False)):
        xyz, role = corridor_cloud(1.60, door=(3.0, 3.0 + w))
        g = ev.build_walkable(xyz, role)
        lab, _ = ndi.label(g["walk"], structure=np.ones((3, 3)))
        r_in, c_in = ev.to_cell(g, 3.0 + w / 2, 0.8)            # middle of the corridor
        r_out, c_out = ev.to_cell(g, 3.0 + w / 2, 1.60 + 1.5)   # inside the room behind the door
        same = lab[r_in, c_in] > 0 and lab[r_in, c_in] == lab[r_out, c_out]
        assert same == passes, (w, passes)


def test_result_does_not_depend_on_point_order():
    xyz, role = corridor_cloud(1.60, door=(3.0, 3.9))
    g1 = ev.build_walkable(xyz, role)
    p = np.random.default_rng(1).permutation(len(xyz))
    g2 = ev.build_walkable(xyz[p], role[p])
    assert np.array_equal(g1["walk"], g2["walk"]) and np.array_equal(g1["free"], g2["free"])


def test_floor_height_per_cell_follows_a_sloping_slab():
    """A slab that rises 12 cm over 6 m does not turn its high end into an obstacle."""
    xyz, role = corridor_cloud(1.60)
    xyz = xyz.copy()
    xyz[:, 2] += 0.02 * xyz[:, 0]                              # 2 cm per metre
    g = ev.build_walkable(xyz, role)
    assert abs(walk_width(g, 5.5) - 1.10) <= ev.CELL + 1e-9
