# -*- coding: utf-8 -*-
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-turn-a-3d-scan-into-a-cad-floor-plan-f38562744833
# Dataset:  https://learngeodata.eu/materials/scan-to-floor-plan-dxf/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""Parameter sweeps and stability checks behind the numbers quoted in the article.

Runs the pipeline with one knob changed at a time, then checks that the answer
does not depend on things it should not depend on: the order of the points,
their rounding, where the 2 cm grid happens to fall, and the point density.
Writes out/sweeps.json.
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

import walls_to_vectors as W

quiet = io.StringIO()
v = W.read_ply(W.SCAN_FILE)
XYZ = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
CLS = np.rint(v["scalar_semantic_label"]).astype(int)


def summary(m):
    me = m["measure"]
    return dict(chains=m["chains"]["chains_kept"], pieces=m["runs"]["pieces"],
                runs=m["runs"]["runs"], walls=m["merge"]["walls"],
                fragments=m["merge"]["fragments"],
                corner_gap_cm=m["square"]["max_corner_gap_before_cm"],
                room_m2=me["room_area_m2"],
                lengths=[w["length_m"] for w in me["walls"]],
                seen=[w["seen"] for w in me["walls"]],
                gaps=[g for w in me["walls"] for g in w["gaps_m"]])


def run(xyz=XYZ, cls=CLS, **knobs):
    old = {k: getattr(W, k) for k in knobs}
    try:
        for k, val in knobs.items():
            setattr(W, k, val)
        with redirect_stdout(quiet):
            m, _ = W.run_pipeline(xyz, cls)
        return summary(m)
    except Exception as e:                       # a setting can break the chain
        return {"failed": f"{type(e).__name__}: {e}"}
    finally:
        for k, val in old.items():
            setattr(W, k, val)


res = {}
res["baseline"] = run()
res["no_snap"] = run(SNAP=False)
res["split_tol"] = {str(t): run(SPLIT_TOL=t) for t in (0.01, 0.02, 0.04, 0.08, 0.16)}
res["close_radius"] = {str(r): run(CLOSE_RADIUS=r) for r in (0, 1, 2, 4, 6)}
res["merge_offset"] = {str(o): run(MERGE_OFFSET=o) for o in (0.03, 0.05, 0.10, 0.20, 0.40)}

# the run angles and lengths quoted in step 6
with redirect_stdout(quiet):
    _, st = W.run_pipeline(XYZ, CLS)
res["run_angles_deg"] = [round(W.angle_of(r["p0"], r["p1"]), 2) for r in st["runs"]]
res["run_lengths_m"] = [round(float(np.linalg.norm(r["p1"] - r["p0"])), 2) for r in st["runs"]]

# stability: order, rounding, grid phase, density
rng = np.random.default_rng(W.SEED)
stab = {}
stab["rerun"] = run()
p = rng.permutation(len(XYZ))
stab["shuffled"] = run(XYZ[p], CLS[p])
stab["rounded_1mm"] = run(np.round(XYZ, 3))
# the grid is anchored to the data, so a translation changes nothing; a rotation
# moves every wall onto a different staircase of cells: that is the real test
for deg in (0.5, 1.0, 7.5, 30.0):
    th = np.radians(deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    X = XYZ @ Rz.T
    stab[f"rotated_{deg}_deg"] = run(X)
    stab[f"rotated_{deg}_deg_no_snap"] = run(X, SNAP=False)
stab["translated_1_cm"] = run(XYZ + np.array([0.01, 0.01, 0.0]))
keep = rng.random(len(XYZ)) < 0.5
stab["half_the_points"] = run(XYZ[keep], CLS[keep])
res["stability"] = stab


def spread(keys):
    """Largest spread of any wall length and of the room area over runs that
    found the same four walls; runs that found a different count are listed."""
    four = [k for k in keys if stab[k].get("walls") == 4 and stab[k].get("room_m2")]
    other = {k: stab[k].get("walls", stab[k].get("failed")) for k in keys if k not in four}
    L = np.array([sorted(stab[k]["lengths"]) for k in four])
    A = np.array([stab[k]["room_m2"] for k in four])
    return dict(runs=len(four), max_length_spread_mm=round(1000 * float((L.max(0) - L.min(0)).max()), 1),
                area_spread_m2=round(float(A.max() - A.min()), 3), different_wall_count=other)


stab["baseline"] = res["baseline"]
stab["baseline_no_snap"] = res["no_snap"]
rot = [k for k in stab if k.startswith("rotated")]
res["stability_summary"] = dict(
    identical_rerun=stab["rerun"] == res["baseline"],
    identical_shuffled=stab["shuffled"] == res["baseline"],
    identical_translated=stab["translated_1_cm"] == res["baseline"],
    rounded_1mm=spread(["baseline", "rounded_1mm"]),
    rotations_with_snap=spread(["baseline"] + [k for k in rot if not k.endswith("no_snap")]),
    rotations_without_snap=spread(["baseline_no_snap"] + [k for k in rot if k.endswith("no_snap")]),
    half_the_points=spread(["baseline", "half_the_points"]))
del stab["baseline"], stab["baseline_no_snap"]

os.makedirs(W.OUT, exist_ok=True)
with open(os.path.join(W.OUT, "sweeps.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res["stability_summary"], indent=1))
for k, val in res["stability"].items():
    print(k, val)
