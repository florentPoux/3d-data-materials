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
"""Dead-end corridors and common paths on a floor drawn by hand."""
import numpy as np

from conftest import ev, make_grid, make_src


def t_floor(stub_len_cells=100, office=True):
    """A corridor (1 m of walkable width) between exits A (west end) and B (east end), a dead-end
    corridor branching north from its middle, and an office behind one door further east."""
    H, W = 200, 400
    walk = np.zeros((H, W), bool)
    walk[0:21, :] = True                                         # main corridor, rows 0-20
    walk[21:21 + stub_len_cells, 190:211] = True                 # the stub, 1 m wide
    rooms = np.zeros((H, W), int)
    rooms[walk] = 1
    if office:
        walk[21:25, 300:311] = True                              # the door throat, 0.55 m wide
        walk[25:85, 280:341] = True                              # the office, 3 m x 3 m
        rooms[21:25, 300:311] = 1
        rooms[25:85, 280:341] = 2
    grid = make_grid(walk)
    src = make_src(grid, {"A": [(r, 0) for r in range(21)], "B": [(r, W - 1) for r in range(21)]})
    return grid, src, rooms


def run(grid, src, rooms, kind, open_exits):
    G, idx = ev.walking_graph(grid["walk"])
    Ga, node = ev.attach_exits(G, idx, src)
    D = ev.distance_field(grid, Ga, idx, src, open_exits, node)[0]
    return ev.dead_ends(grid, Ga, idx, rooms, kind, D, src, open_exits)


def test_stub_is_one_dead_end_corridor_of_its_length():
    """The stub is 5.0 m deep from the corridor's edge and 5.5 m from its centre line: the walk
    back to where a person can choose a direction lies between the two."""
    grid, src, rooms = t_floor(100)
    depth, regions, share, nb = run(grid, src, rooms, ["", "circulation", "room"], ["A", "B"])
    corr = ev.corridor_dead_ends(regions)
    assert len(corr) == 1
    assert 4.95 <= corr[0]["corridor_m"] <= 5.55, corr[0]


def test_office_behind_one_door_is_a_common_path_not_a_dead_end():
    grid, src, rooms = t_floor(100)
    depth, regions, share, nb = run(grid, src, rooms, ["", "circulation", "room"], ["A", "B"])
    office = [r for r in regions if r["room"] == 2]
    assert len(office) == 1 and office[0]["depth_m"] > 3.0      # one way out, several metres deep
    assert office[0]["corridor_m"] < ev.MIN_CORRIDOR_M          # but not a dead-end corridor
    assert all(r["room"] != 2 for r in ev.corridor_dead_ends(regions))


def test_blocking_one_end_turns_the_corridor_into_a_dead_end():
    grid, src, rooms = t_floor(100, office=False)
    regions2 = run(grid, src, rooms, ["", "circulation"], ["A", "B"])[1]
    regions1 = run(grid, src, rooms, ["", "circulation"], ["A"])[1]
    long2 = max(r["corridor_m"] for r in ev.corridor_dead_ends(regions2))
    long1 = max(r["corridor_m"] for r in ev.corridor_dead_ends(regions1))
    assert long1 > 15.0 > long2                                  # the whole east end is now one way


def test_a_loop_has_no_dead_end():
    H, W = 120, 120
    walk = np.zeros((H, W), bool)
    walk[0:21, :] = walk[-21:, :] = walk[:, 0:21] = walk[:, -21:] = True   # a ring corridor
    grid = make_grid(walk)
    rooms = walk.astype(int)
    src = make_src(grid, {"A": [(10, 0)]})
    regions = run(grid, src, rooms, ["", "circulation"], ["A"])[1]
    assert ev.corridor_dead_ends(regions) == []
