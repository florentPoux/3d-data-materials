# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-automate-3d-deliverables-accessibility-map-from-3d-scan-3ecef96a5a6f
# Dataset:  https://learngeodata.eu/materials/wheelchair-accessibility-audit/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""
Is the audit stable? Run it on the same scan four ways and compare the headline numbers.

  as is       the scan as prepared
  again       the same file a second time                       -> must be identical
  shuffled    the same points in a random order (seed 1)        -> must be identical
  rounded     every coordinate rounded to the millimetre         -> within the stated tolerance

Writes out/stability/<case>/ and prints one table. About 15 seconds.
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import accessibility_audit as aa

HERE = Path(__file__).resolve().parent
ROOT = HERE / "out" / "stability"
KEYS = ["rooms", "rooms_wheelchair", "rooms_on_foot", "doors", "doors_under_required", "issues",
        "issues_high", "rooms_after_best_ramp", "reached_rooms_with_circle", "centre_m2",
        "width_ge_passage_m2", "turn_m2"]


def write_ply(path, v):
    names = v.dtype.names
    kind = {"<f4": "float", "|u1": "uchar", "<f8": "double", "<i4": "int"}
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(v)}"]
    head += [f"property {kind[v.dtype[n].str]} {n}" for n in names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        v.tofile(f)


def run(case, data):
    with contextlib.redirect_stdout(io.StringIO()):
        aa.main(data, ROOT / case)
    return json.loads((ROOT / case / "metrics.json").read_text(encoding="utf-8"))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    v = aa.read_ply(aa.DATA)
    shuffled = v[np.random.default_rng(1).permutation(len(v))]
    write_ply(ROOT / "shuffled.ply", shuffled)
    rounded = v.copy()
    for k in ("x", "y", "z"):
        rounded[k] = np.round(rounded[k].astype(np.float64), 3).astype(np.float32)
    write_ply(ROOT / "rounded.ply", rounded)
    res = {"as is": run("as_is", aa.DATA), "again": run("again", aa.DATA),
           "shuffled": run("shuffled", ROOT / "shuffled.ply"), "rounded": run("rounded", ROOT / "rounded.ply")}
    print(f"{'':28s}" + "".join(f"{c:>11s}" for c in res))
    for k in KEYS:
        print(f"{k:28s}" + "".join(f"{str(r[k]):>11s}" for r in res.values()))
    same = all(res["as is"][k] == res[c][k] for c in ("again", "shuffled") for k in KEYS)
    print("\nagain and shuffled identical to as is:", same)
    (ROOT / "stability.json").write_text(json.dumps({c: {k: r[k] for k in KEYS} for c, r in res.items()},
                                                    indent=1), encoding="utf-8")
    for f in ("shuffled.ply", "rounded.ply"):
        (ROOT / f).unlink()


if __name__ == "__main__":
    main()
