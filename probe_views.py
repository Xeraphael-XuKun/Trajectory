"""How many cameras does WHU-MARS have, which are aerial, and how much data is
aerial RGB?  Read-only: touches filenames only, opens no image, builds no model.

    python probe_views.py [/mnt/cache/wanghanzhi/Datasets]

Two things have to be settled before direction two is worth implementing:

1. The camera -> view mapping.  `whu_mars.py`'s commented-out WHU-MARS-1000-GD
   filter keeps `camid <= 5` (one-based) as the ground subset, and the dataset
   paper describes "5 ground nodes + 2 UAVs", so c1-c5 should be ground and
   c6-c7 aerial.  That is an inference from a comment, never measured.

2. The share of images that are aerial *and* RGB.  Direction two supervises
   only those, and only on RGB (CLIP has never seen thermal).  If they are a
   few percent of the data, a batch of 192 carries a handful of supervised
   images and the loss will not move anything -- which is worth knowing before
   writing the loss rather than after training for two hours.

Nothing here decides the mapping; it reports the evidence.  Whether c6/c7 are
really the drones has to be confirmed by looking at a few images.
"""

import glob
import os
import os.path as osp
import re
import sys
from collections import Counter, defaultdict

PATTERN = re.compile(r'(\d+)_c(\d+)')
MODALITIES = ('RGB', 'IR', 'Thermal')
SPLITS = ('train', 'query', 'test')          # whu_mars.py maps gallery -> test/
AERIAL_GUESS = (6, 7)                        # one-based, to be confirmed


def main(data_root):
    root = osp.join(data_root, 'WHU-MARS')
    print('WHU-MARS:', root)
    if not osp.isdir(root):
        print('  not found'); return 1

    # counts[split][modality][camid(one-based)] = n
    counts = defaultdict(lambda: defaultdict(Counter))
    pids_per_cam = defaultdict(set)
    unparsed = []
    example = {}

    for split in SPLITS:
        for m in MODALITIES:
            d = osp.join(root, split, m)
            if not osp.isdir(d):
                print('  missing {}/{}'.format(split, m)); continue
            for p in glob.glob(osp.join(d, '*.jpg')):
                name = osp.basename(p)
                hit = PATTERN.search(name)
                if hit is None:
                    if len(unparsed) < 5:
                        unparsed.append(name)
                    continue
                pid, cam = int(hit.group(1)), int(hit.group(2))
                counts[split][m][cam] += 1
                pids_per_cam[cam].add(pid)
                example.setdefault((split, m, cam), name)

    if unparsed:
        print('  filenames that do not parse:', unparsed)

    cams = sorted({c for s in counts for m in counts[s] for c in counts[s][m]})
    print('\n=== 1. cameras present:', cams)
    print('  (paper: 5 ground nodes + 2 UAVs = 7)')

    print('\n=== 2. images per camera per modality')
    for split in SPLITS:
        if not counts[split]:
            continue
        total = sum(sum(counts[split][m].values()) for m in MODALITIES)
        print('\n  {}  ({} images)'.format(split, total))
        print('    cam ' + ''.join('%10s' % m for m in MODALITIES) + '%10s%9s  ids' % ('total', 'share'))
        for c in cams:
            row = [counts[split][m][c] for m in MODALITIES]
            sub = sum(row)
            print('    c%-3d' % c + ''.join('%10d' % v for v in row)
                  + '%10d%8.1f%%  %d' % (sub, 100.0 * sub / total if total else 0, len(pids_per_cam[c])))

    print('\n=== 3. one example filename per camera (train/RGB)')
    for c in cams:
        print('    c%-3d  %s' % (c, example.get(('train', 'RGB', c), '<none>')))

    print('\n=== 4. what direction two would actually supervise')
    print('    assuming aerial = c{} (INFERRED from the GD filter, not verified)'
          .format('/c'.join(map(str, AERIAL_GUESS))))
    tr = counts['train']
    tot = sum(sum(tr[m].values()) for m in MODALITIES)
    aer = sum(tr[m][c] for m in MODALITIES for c in AERIAL_GUESS)
    aer_rgb = sum(tr['RGB'][c] for c in AERIAL_GUESS)
    for label, n in (('train images, all', tot),
                     ('  of which aerial (any modality)', aer),
                     ('  of which aerial AND RGB  <-- supervised', aer_rgb)):
        print('    %-38s %8d %7.1f%%' % (label, n, 100.0 * n / tot if tot else 0))
    if tot:
        per_batch = 192.0 * aer_rgb / tot
        print('\n    a 192-image batch would carry ~%.0f supervised images' % per_batch)
        if per_batch < 8:
            print('    WARNING: that is very few -- the text loss would see almost no'
                  '\n    gradient.  Reconsider before implementing.')

    print('\n=== 5. still to confirm by eye')
    print('    Open one image from each camera and check which are shot from above.')
    print('    The GD filter in whu_mars.py:87 keeps camid<=5 as "ground", which is')
    print('    the only evidence for the mapping so far.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '/mnt/cache/wanghanzhi/Datasets'))
