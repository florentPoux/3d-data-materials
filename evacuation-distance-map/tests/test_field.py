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
"""The distance field, the routes and the what-if, on grids whose answers are known."""
import numpy as np

from conftest import ev, solve_field


def test_move_sets():
    assert ev.crossed(1, 1) == [(0, 1), (1, 0)]
    assert ev.crossed(1, 2) == [(0, 1), (1, 1)]
    assert ev.crossed(0, 1) == []
    assert len(ev.all_moves(8)) == 8 and len(ev.all_moves(16)) == 16 and len(ev.all_moves(32)) == 32


def open_floor(neighbours, n=161):
    walk = np.ones((n, n), bool)
    c = n // 2
    D = solve_field(walk, {"X": [(c, c)]}, neighbours)[5]
    rr, cc = np.mgrid[0:n, 0:n]
    eu = np.hypot(rr - c, cc - c) * ev.CELL
    far = eu > 2.0
    return D, eu, D[far] / eu[far] - 1


def test_open_floor_exact_along_the_moves():
    """Along a move direction the graph distance is the straight distance, to float precision."""
    D, eu, _ = open_floor(16)
    c = D.shape[0] // 2
    for dr, dc in [(0, 1), (1, 1), (1, 2), (2, 1)]:
        for k in range(1, 30):
            assert abs(D[c + k * dr, c + k * dc] - eu[c + k * dr, c + k * dc]) < 1e-9


def test_open_floor_error_bounds():
    """The known price of a grid, never short: 8 neighbours reads up to 8.2 % long, 16 up to
    2.8 %, 32 up to 1.4 %."""
    worst = {}
    for nb in (8, 16, 32):
        _, _, err = open_floor(nb)
        assert err.min() > -1e-9
        worst[nb] = err.max()
    print("worst open-floor overestimate", {k: round(100 * v, 2) for k, v in worst.items()})
    assert 0.075 < worst[8] < 0.083
    assert worst[16] < 0.028
    assert worst[32] < 0.014


def test_wall_with_a_gap_known_geodesic():
    """A wall across the floor with a one-cell gap: the walk is two straight legs through the gap."""
    n = 121
    walk = np.ones((n, n), bool)
    walk[60, :] = False
    walk[60, 90] = True                                          # the gap
    D = solve_field(walk, {"X": [(20, 20)]}, 32)[5]
    exact = (np.hypot(60 - 20, 90 - 20) + np.hypot(100 - 60, 20 - 90)) * ev.CELL
    assert D[100, 20] >= exact - 1e-9
    assert D[100, 20] / exact - 1 < 0.012


def test_multi_source_is_the_minimum_of_single_sources():
    rng = np.random.default_rng(3)
    walk = rng.random((80, 90)) > 0.15
    exits = {"A": [(5, 5)], "B": [(70, 80)], "C": [(40, 10)]}
    for v in exits.values():
        walk[v[0]] = True
    D, serve = solve_field(walk, exits)[5:7]
    singles = np.stack([solve_field(walk, exits, open_exits=[e])[5] for e in exits])
    low = singles.min(0)
    f = np.isfinite(D)
    assert np.array_equal(f, np.isfinite(low))
    assert np.allclose(D[f], low[f], atol=1e-9)
    names = np.array(list(exits))                               # the serving exit is the nearest one
    best = names[np.argmin(singles, axis=0)]
    srt = np.sort(singles, axis=0)
    with np.errstate(invalid="ignore"):
        tie = (srt[1] - srt[0]) < 1e-9
    assert np.all((serve[f] == best[f]) | tie[f])


def test_descent_length_equals_the_field():
    rng = np.random.default_rng(4)
    walk = rng.random((70, 70)) > 0.2
    walk[2, 2] = walk[65, 60] = True
    grid, src, G, idx, node, D, serve, leave = solve_field(walk, {"A": [(2, 2)], "B": [(65, 60)]})
    rr, cc = np.nonzero(np.isfinite(D))
    for i in rng.choice(len(rr), 60, replace=False):
        path, length = ev.descend(D, walk, (rr[i], cc[i]), leave)
        assert abs(length - D[rr[i], cc[i]]) < 1e-9
        assert all(walk[p] for p in path)


def test_blocking_an_exit_only_lengthens_and_only_its_catchment():
    rng = np.random.default_rng(5)
    walk = rng.random((80, 80)) > 0.15
    ex = {"A": [(3, 3)], "B": [(75, 75)], "C": [(3, 75)]}
    for v in ex.values():
        walk[v[0]] = True
    D, serve = solve_field(walk, ex)[5:7]
    Db = solve_field(walk, ex, open_exits=["A", "C"])[5]
    f = np.isfinite(D)
    assert np.all(Db[f] >= D[f] - 1e-9)
    keep = f & (serve != "B")
    assert np.allclose(Db[keep], D[keep], atol=1e-9)


def test_a_blocked_exit_is_not_a_shortcut():
    """Two cells joined to the same blocked exit must not reach each other through its node."""
    walk = np.zeros((10, 40), bool)
    walk[5, :] = True
    walk[5, 20] = False                                          # the corridor is cut in two
    ex = {"A": [(5, 0)], "B": [(5, 19), (5, 21)]}                # B touches both halves
    D = solve_field(walk, ex, open_exits=["A"])[5]
    assert np.isinf(D[5, 30])


def test_exit_offset_charges_the_last_step():
    """A cell 0.60 m from the door leaf is 0.60 m from the exit, not 0: the leg to the door counts."""
    walk = np.ones((40, 40), bool)
    grid = dict(shape=walk.shape, walk=walk, origin=np.zeros(2))
    cells = np.array([20 * 40 + c for c in range(0, 16)])
    src = {"A": dict(cells=cells, offset=np.arange(16) * ev.CELL, leaf=np.array([], int))}
    G, idx = ev.walking_graph(walk)
    Ga, node = ev.attach_exits(G, idx, src)
    D = ev.distance_field(grid, Ga, idx, src, None, node)[0]
    assert abs(D[20, 12] - 0.60) < 1e-9
    assert abs(D[20, 30] - 1.50) < 1e-9
