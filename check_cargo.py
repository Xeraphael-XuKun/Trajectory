"""Integrity check for a freshly downloaded CARGO tree.

    python check_cargo.py [/mnt/cache/wanghanzhi/Datasets]

Structure is the easy part and the least likely to be wrong.  What actually
goes wrong with a multi-gigabyte download is quieter:

  * a truncated archive leaves a few hundred JPEGs that open but cannot decode.
    datasets/bases.py sets ImageFile.LOAD_TRUNCATED_IMAGES = True, so those
    train perfectly happily on grey mush -- this script turns that flag off on
    purpose so the bad files surface here instead;
  * an extra nesting level (CARGO/CARGO/train) makes every glob return nothing,
    which reads as "0 images" only if you look;
  * a filename that our parser cannot read is skipped silently by glob-based
    loaders, quietly shrinking the dataset.

Everything is checked with the same parser training will use, not a copy.
"""

import os
import sys
import types
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Bypass datasets/__init__.py (it imports timm via make_dataloader); we only
# want the dataset class itself.
_pkg = types.ModuleType('datasets')
_pkg.__path__ = [os.path.join(ROOT, 'datasets')]
sys.modules.setdefault('datasets', _pkg)

from datasets.cargo import CARGO  # noqa: E402

from PIL import Image, ImageFile  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = False   # the whole point -- see the docstring

PAPER_CAMS = 13
RAW_IMAGES = 108563       # what both papers quote for the dataset as collected
RAW_IDS = 5000

# ...but the *released* split is smaller by construction, and that is the number
# to check against.  VDT (CVPR'24, sec. 5.1): "we split it into the train
# (51,451 images with 2500 IDs) and test sets (51,024 images with the remaining
# 2500 IDs) with an almost 1:1 ratio."  Sec. 7.1 explains the shortfall: "we
# select 149 IDs as the query set ... For each identity selected as a query, we
# keep only its images from two random cameras" -- so a query identity's images
# from its other eleven cameras are dropped, 6,088 in total.
SPLIT_IMAGES = {'train': 51451, 'test': 51024}
SPLIT_IDS = {'train': 2500, 'test': 2500}
QUERY_IDS = 149

problems = []


def bad(msg):
    problems.append(msg)
    print('  FAIL   {}'.format(msg))


def ok(msg):
    print('  ok     {}'.format(msg))


def note(msg):
    print('  note   {}'.format(msg))


def main(data_root):
    root = os.path.join(data_root, 'CARGO')
    print('CARGO integrity check:', root)

    # ---- 1. layout ------------------------------------------------------
    print('\n=== 1. layout')
    if not os.path.isdir(root):
        bad('{} does not exist'.format(root))
        return 1
    if os.path.isdir(os.path.join(root, 'CARGO')):
        bad('found {}/CARGO -- the archive unpacked one level too deep; '
            'move the inner folder up'.format(root))
        return 1

    splits = ('train', 'query', 'gallery')
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            bad('missing {}/'.format(split))
            continue
        cams = sorted(x for x in os.listdir(d) if x.startswith('Cam'))
        if len(cams) == PAPER_CAMS:
            ok('{:<8} Cam1..Cam{}'.format(split, PAPER_CAMS))
        else:
            bad('{} has {} Cam* folders, expected {}: {}'.format(split, len(cams), PAPER_CAMS, cams))
    if problems:
        return 1

    # ---- 2. counts ------------------------------------------------------
    print('\n=== 2. counts')
    files = {}
    for split in splits:
        paths = []
        for cam in range(1, PAPER_CAMS + 1):
            d = os.path.join(root, split, 'Cam{}'.format(cam))
            paths += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.jpg')]
        files[split] = sorted(paths)
        print('  {:<8} {:>7} images'.format(split, len(paths)))
    total = sum(len(v) for v in files.values())

    got = {'train': len(files['train']),
           'test': len(files['query']) + len(files['gallery'])}
    for name, want in SPLIT_IMAGES.items():
        if got[name] == want:
            ok('{:<5} {} images, exactly VDT section 5.1'.format(name, got[name]))
        else:
            bad('{} has {} images, VDT section 5.1 says {} ({:+d})'.format(
                name, got[name], want, got[name] - want))
    print('  total {} = {} released.  ({} were collected; the {} difference is the '
          'query identities\'\n        images from their other eleven cameras, dropped by '
          'the protocol -- not missing.)'
          .format(total, sum(SPLIT_IMAGES.values()), RAW_IMAGES, RAW_IMAGES - sum(SPLIT_IMAGES.values())))

    # ---- 3. every filename must parse -----------------------------------
    print('\n=== 3. filename parsing (the parser training will use)')
    ids, cams_seen, unparsed = defaultdict(set), Counter(), []
    per_cam = Counter()
    for split in splits:
        for p in files[split]:
            try:
                pid, cam, aerial = CARGO.parse(p)
            except ValueError as e:
                if len(unparsed) < 5:
                    unparsed.append(str(e))
                continue
            ids[split].add(pid)
            cams_seen[cam] += 1
            per_cam[(split, cam)] += 1
    parsed = sum(cams_seen.values())
    if parsed == total:
        ok('all {} filenames parse'.format(total))
    else:
        bad('{} of {} filenames do not parse; first few: {}'.format(
            total - parsed, total, unparsed))

    print('  cameras present: {}'.format(sorted(cams_seen)))
    aerial = sum(n for c, n in cams_seen.items() if c <= CARGO.aerial_max_cam)
    print('  aerial (Cam1-5) {:>7} | ground (Cam6-13) {:>7}'.format(aerial, parsed - aerial))

    # ---- 4. identities --------------------------------------------------
    print('\n=== 4. identities')
    for split in splits:
        print('  {:<8} {:>5} identities'.format(split, len(ids[split])))
    test_ids = ids['query'] | ids['gallery']
    for name, got_ids in (('train', ids['train']), ('test', test_ids)):
        if len(got_ids) == SPLIT_IDS[name]:
            ok('{:<5} {} identities, exactly VDT section 5.1'.format(name, len(got_ids)))
        else:
            bad('{} has {} identities, VDT section 5.1 says {}'.format(
                name, len(got_ids), SPLIT_IDS[name]))
    if len(ids['query']) == QUERY_IDS:
        ok('query    {} identities, exactly VDT section 7.1'.format(QUERY_IDS))
    else:
        bad('query has {} identities, VDT section 7.1 says {}'.format(
            len(ids['query']), QUERY_IDS))
    all_ids = ids['train'] | test_ids
    if len(all_ids) != RAW_IDS:
        bad('{} identities overall, both papers say {}'.format(len(all_ids), RAW_IDS))
    leak = ids['train'] & (ids['query'] | ids['gallery'])
    if leak:
        bad('{} identities appear in BOTH train and test -- the split is wrong '
            '(e.g. {})'.format(len(leak), sorted(leak)[:5]))
    else:
        ok('train and test identities are disjoint')
    orphan = ids['query'] - ids['gallery']
    if orphan:
        note('{} query identities have no gallery entry (they score as invalid '
             'queries, which is normal in small protocols)'.format(len(orphan)))

    # ---- 5. decodability ------------------------------------------------
    # Every file, fully decoded.  Image.verify() only walks the header, and a
    # JPEG truncated part-way through its scan data passes that and then fails
    # at training time -- which is exactly the case this section exists for.
    # These are small person crops, so the whole set decodes in well under a
    # minute; sampling would trade the only guarantee here for nothing.
    print('\n=== 5. decodability  (all {} files, fully decoded; '
          'LOAD_TRUNCATED_IMAGES is off on purpose)'.format(total))
    empty, broken, done = [], [], 0
    for split in splits:
        for p in files[split]:
            done += 1
            if done % 20000 == 0:
                print('  ... {}/{}'.format(done, total))
            if os.path.getsize(p) == 0:
                empty.append(p)
                continue
            try:
                Image.open(p).load()
            except Exception as e:                # noqa: BLE001
                if len(broken) < 10:
                    broken.append('{}: {}'.format(os.path.relpath(p, root), e))
                else:
                    broken.append(None)
    if empty:
        bad('{} zero-byte files, e.g. {}'.format(
            len(empty), [os.path.relpath(x, root) for x in empty[:3]]))
    if broken:
        bad('{} unreadable images (truncated download), e.g.\n         {}'.format(
            len(broken), '\n         '.join(x for x in broken[:5] if x)))
    if not empty and not broken:
        ok('all {} images decode'.format(total))

    # ---- 6. the four protocols ------------------------------------------
    print('\n=== 6. protocols (built through the real dataset class)')
    for proto in ('ALL', 'AA', 'GG', 'AG'):
        try:
            ds = CARGO(root=data_root, verbose=False, modalities=['RGB'], protocol=proto)
            print('  {:<4} train {:>6} imgs / {:>4} ids | query {:>6} | gallery {:>6} | camids {}'
                  .format(proto, len(ds.train['RGB']), ds.num_train_pids,
                          len(ds.query['RGB']), len(ds.gallery['RGB']),
                          sorted({c for _, _, c, _ in ds.train['RGB']})))
        except Exception as e:                    # noqa: BLE001
            bad('protocol {} failed to build: {}: {}'.format(proto, type(e).__name__, e))

    print()
    if problems:
        print('{} PROBLEM(S) -- do not start training:'.format(len(problems)))
        for p in problems:
            print('  - {}'.format(p))
        return 1
    print('CARGO looks complete.  Next:  bash run_hihr.sh cargo_base')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '/mnt/cache/wanghanzhi/Datasets'))
