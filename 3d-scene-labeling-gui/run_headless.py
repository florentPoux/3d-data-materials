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
"""Run da_semantic_masking.py with no screen: recorded strokes instead of a mouse.

The tutorial is interactive on purpose: you paint the labels. That makes it
impossible to run on a server, in CI, or twice with the same result. This
runner keeps every line of the tutorial as it is and changes three things:

  1. The painter replays brush strokes from a JSON file (kids_strokes.json)
     through its own paint_mask(), with the class and brush size the 1-5 and
     +/- keys would set. The mask is resampled back to the depth resolution by
     the painter's own get_original_resolution_mask(). Nothing about the mask
     path differs from a hand-painted run, except that the hand is recorded.
  2. Open3D viewers return at once instead of opening a window.
  3. matplotlib figures are written to results/KIDS/figures/ instead of shown.

Each painted mask is also saved as .npy in results/KIDS/masks/, which is how
you move masks painted on a laptop to a headless machine.

Usage:
    python run_headless.py                       # replays kids_strokes.json
    python run_headless.py --strokes my.json     # your own recorded strokes
For the interactive tool, run the tutorial itself: python da_semantic_masking.py
"""
import argparse
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def install_replay(strokes_path, mask_dir):
    """Swap the painter's window loop for a replay of recorded strokes."""
    import numpy as np
    import interactive_painting as ip

    with open(strokes_path, encoding='utf-8') as f:
        spec = json.load(f)
    sequence = spec['sequence']
    state = {'call': 0}

    def replay_run(self, window_name='HD Mask Painter'):
        n = state['call']
        if n >= len(sequence):
            raise SystemExit('The strokes file covers %d painting calls; the script made '
                             'a %dth. Add a frame to "sequence".' % (len(sequence), n + 1))
        frame = sequence[n]
        state['call'] += 1
        h, w = self.display_shape
        for stroke in spec['frames'][str(frame)]:
            # What the keys do in the interactive loop: 1-5 set the class,
            # +/- set the brush radius.
            self.current_class = int(stroke['class'])
            self.brush_size = int(stroke['brush'])
            pts = [(u * (w - 1), v * (h - 1)) for u, v in stroke['points']]
            # Mouse down on the first point, then mouse moves along the path,
            # one event every third of a brush radius, as a steady hand would.
            step = max(1.0, self.brush_size / 3.0)
            x0, y0 = pts[0]
            self.paint_mask(int(round(x0)), int(round(y0)))
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                n_steps = max(1, int(np.hypot(xb - xa, yb - ya) / step))
                for t in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
                    self.paint_mask(int(round(xa + t * (xb - xa))),
                                    int(round(ya + t * (yb - ya))))
        mask = self.get_original_resolution_mask()
        os.makedirs(mask_dir, exist_ok=True)
        np.save(os.path.join(mask_dir, 'mask_call%02d_frame%02d.npy' % (n + 1, frame)), mask)
        labeled = 100.0 * (mask > 0).mean()
        print('[headless] painting call %d: frame %d, %d strokes replayed, %.1f%% of pixels labeled'
              % (n + 1, frame, len(spec['frames'][str(frame)]), labeled))
        return mask

    ip.HDMultiClassMaskPainter.run = replay_run


def close_viewers(fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import open3d as o3d

    o3d.visualization.draw_geometries = lambda *a, **k: None
    os.makedirs(fig_dir, exist_ok=True)
    count = {'n': 0}

    def save_instead(*a, **k):
        count['n'] += 1
        p = os.path.join(fig_dir, 'figure_%02d.png' % count['n'])
        plt.savefig(p, dpi=110, bbox_inches='tight')
        plt.close('all')
        print('[headless] figure written: %s' % p)

    plt.show = save_instead


def main():
    # The tutorial prints arrows; a redirected Windows console would choke on them.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--strokes', default=os.path.join(HERE, 'kids_strokes.json'))
    ap.add_argument('--script', default=os.path.join(HERE, 'da_semantic_masking.py'))
    ap.add_argument('--batch-size', type=int, default=0,
                    help='fusion batch size (the script uses 100000). Lower it on a '
                         'machine short of RAM: every point votes on its own '
                         'neighbours only, so the labels come out the same.')
    a = ap.parse_args()

    results = os.path.join(HERE, 'results', 'KIDS')
    sys.path.insert(0, HERE)
    close_viewers(os.path.join(results, 'figures'))
    install_replay(a.strokes, os.path.join(results, 'masks'))
    os.chdir(HERE)
    t0 = time.time()
    if a.batch_size:
        with open(a.script, encoding='utf-8') as f:
            src = f.read()
        old = 'batch_size=100000'
        if old not in src:
            raise SystemExit('--batch-size: the script no longer contains %r' % old)
        src = src.replace(old, 'batch_size=%d' % a.batch_size)
        print('[headless] fusion batch size %d instead of 100000' % a.batch_size)
        g = {'__name__': '__main__', '__file__': os.path.abspath(a.script)}
        exec(compile(src, os.path.abspath(a.script), 'exec'), g)
    else:
        g = runpy.run_path(a.script, run_name='__main__')
    # Keep the labels before and after fusion, so the two states can be
    # compared or plotted without running the tutorial again.
    import numpy as np
    np.savez_compressed(os.path.join(results, 'labels_before_after.npz'),
                        before=g['all_labels_3d'], after=g['fused_labels'])
    print('[headless] labels before and after fusion saved to %s'
          % os.path.join(results, 'labels_before_after.npz'))
    print('[headless] finished in %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
