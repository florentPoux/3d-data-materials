# -*- coding: utf-8 -*-
#
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
# How stable are the thicknesses? Runs the measurement on the real office again:
#   - twice, as delivered (must be identical)
#   - with the points shuffled into three random orders (must be identical)
#   - with every coordinate rounded to 1 mm (a re-export can do that)
#   - with the normal neighbourhood at 8, 12 and 16 points (a parameter, not the data)
# and writes out/robustness.json. Run after wall_thickness_to_ifc.py:  python check_determinism.py
#
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wall_thickness_to_ifc as W  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def summary(res):
    # key each wall by its family and where it stands, so a renaming cannot hide a change
    ws = sorted(res["walls"], key=lambda w: (w["family"], w["face_a"]))
    return {f"{w['family']}{i}": {"t_cm": round(100 * w["thickness"], 4),
                                                  "ci_cm": [round(100 * c, 4) for c in w["ci"]],
                                                  "lean_a": round(w["lean_a_deg"], 4),
                                                  "lean_b": round(w["lean_b_deg"], 4),
                                                  "cells": w["n_cells"]}
            for i, w in enumerate(ws)}


def main():
    xyz, _, cls, names, ids = W.load_scan(W.DATA, W.SIDECAR)
    quiet = lambda m: None
    runs = {"as delivered": summary(W.measure(xyz, cls, names, ids, log=quiet)),
            "as delivered, again": summary(W.measure(xyz, cls, names, ids, log=quiet))}
    for seed in (1, 2, 3):
        p = np.random.default_rng(seed).permutation(len(xyz))
        runs[f"shuffled (seed {seed})"] = summary(W.measure(xyz[p], cls[p], names, ids, log=quiet))
    runs["rounded to 1 mm"] = summary(W.measure(np.round(xyz, 3), cls, names, ids, log=quiet))
    k0 = W.NORMAL_K
    for k in (8, 12, 16):
        W.NORMAL_K = k
        runs[f"normals from {k} neighbours"] = summary(W.measure(xyz, cls, names, ids, log=quiet))
    W.NORMAL_K = k0

    base = runs["as delivered"]
    report = {"runs": runs, "max_change_cm": {}}
    for name, r in runs.items():
        common = set(r) & set(base)
        d = max((abs(r[k]["t_cm"] - base[k]["t_cm"]) for k in common), default=float("nan"))
        report["max_change_cm"][name] = {"walls": len(r), "matched": len(common),
                                         "max_thickness_change_cm": round(d, 4)}
        print(f"{name:28s} {len(r)} walls, largest thickness change {d:.4f} cm   "
              + "  ".join(f"{k} {v['t_cm']:.2f}" for k, v in sorted(r.items())))
    os.makedirs(W.OUT, exist_ok=True)
    with open(os.path.join(W.OUT, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
