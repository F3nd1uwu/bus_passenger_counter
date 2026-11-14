"""Build a separate detection candidate from reviewed CountPeople and local data.

The source export, existing datasets, weights and active profile are never edited.
Review JSON must cover every source image and bind decisions to image/label hashes.
Only photometric augmentation is applied, and only after the split is frozen.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

SPLITS = ('train', 'val', 'test')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_boxes(path, width, height):
    result = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or values[0] != 0 or not np.isfinite(values).all():
            raise ValueError(f'Invalid person annotation: {path}')
        _, x, y, w, h = values
        box = np.array([x-w/2, y-h/2, x+w/2, y+h/2]) * [width, height, width, height]
        clipped = np.clip(box, 0, [width, height, width, height])
        if (w <= 0 or h <= 0 or np.max(np.abs(box-clipped)) > .02
                or np.any(clipped[2:] <= clipped[:2])):
            raise ValueError(f'Invalid box; only subpixel rounding may be clipped: {path}')
        result.append(clipped.tolist())
    return result


def write_labels(path, boxes, width, height):
    lines = []
    for x1, y1, x2, y2 in boxes:
        lines.append(f'0 {(x1+x2)/2/width:.9f} {(y1+y2)/2/height:.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def episode_splits(rows, seed=42):
    """Keep clips, adjacent recordings (<=120 seconds), and hash-neighbours together."""
    episodes = sorted({r['episode'] for r in rows})
    parent = {e:e for e in episodes}

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    def union(a, b):
        parent[find(b)] = find(a)

    dated = sorted((datetime.strptime(e[:19], '%Y_%m_%d_%H_%M_%S'), e) for e in episodes)
    for (a, ea), (b, eb) in zip(dated, dated[1:]):
        if a.date() == b.date() and (b-a).total_seconds() <= 120:
            union(ea, eb)
    # Similarity is a conservative grouping signal, never automatic label truth.
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if np.count_nonzero(a['phash'] != b['phash']) <= 4:
                union(a['episode'], b['episode'])
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['episode'])].append(row)
    dates = defaultdict(list)
    for group, members in groups.items():
        dates[group[:10]].append((group, members))
    rng = np.random.default_rng(seed)
    for day_groups in sorted(dates.values(), key=lambda g:g[0][0]):
        rng.shuffle(day_groups)
        day_groups.sort(key=lambda g:-len(g[1]))
        n = sum(len(m) for _,m in day_groups)
        target = dict(zip(SPLITS, (n*.70, n*.15, n*.15)))
        counts = Counter()
        for group, members in day_groups:
            split = min(SPLITS, key=lambda s:(counts[s]+len(members))/target[s])
            counts[split] += len(members)
            for row in members:
                row['split'], row['group'] = split, 'countpeople:'+group
    return rows


def augment(image, rng, variant):
    if variant == 1:
        gamma = float(rng.uniform(.85, 1.18))
        gain = float(rng.uniform(.93, 1.07))
        table = np.clip((np.arange(256)/255.)**gamma*255*gain, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table), {'kind':'exposure', 'gamma':gamma, 'gain':gain}
    sigma = float(rng.uniform(1.5, 4.0))
    result = np.clip(image.astype(np.float32)+rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
    quality = int(rng.integers(65, 91))
    ok, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('Augmentation JPEG encoding failed')
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), {'kind':'noise_compression', 'sigma':sigma, 'jpeg_quality':quality}


def verify(root):
    root = Path(root).resolve()
    audit = json.loads((root/'annotations.json').read_text(encoding='utf-8'))
    groups, parents, pixels = {}, {}, {}
    expected = set()
    counts = {s:Counter() for s in SPLITS}
    for name, row in audit['images'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or row.get('reviewed') is not True:
            raise ValueError('Invalid or unreviewed prepared sample')
        split = row['split']
        if split not in SPLITS or Path(name).parts[:2] != ('images', split):
            raise ValueError('Image directory differs from recorded split')
        if digest(path) != row['sha256']:
            raise ValueError('Prepared image changed')
        for mapping, key in ((groups,row['group']), (parents,row['parent']), (pixels,row['pixel_sha256'])):
            if mapping.setdefault(key, split) != split:
                raise ValueError('Source group, parent or identical pixels leak between splits')
        if row['augmentation'] and split != 'train':
            raise ValueError('Augmentation in held-out data')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Unreadable prepared image')
        if hashlib.sha256(image.tobytes()).hexdigest() != row['pixel_sha256']:
            raise ValueError('Decoded pixels differ from manifest')
        label = root/'labels'/split/(path.stem+'.txt')
        if digest(label) != row['label_sha256']:
            raise ValueError('Prepared label changed')
        boxes = read_boxes(label, image.shape[1], image.shape[0])
        if np.asarray(boxes).shape != np.asarray(row['boxes_xyxy']).shape or (boxes and not np.allclose(boxes,row['boxes_xyxy'],atol=1e-5)):
            raise ValueError('Labels differ from reviewed boxes')
        expected.add(path)
        counts[split].update(images=1, boxes=len(boxes), negative=int(not boxes), augmented=int(bool(row['augmentation'])))
    actual = {p.resolve() for p in (root/'images').glob('*/*') if p.is_file()}
    labels = {p.resolve() for p in (root/'labels').glob('*/*') if p.is_file()}
    if actual != expected or labels != {root/'labels'/p.parent.name/(p.stem+'.txt') for p in expected}:
        raise ValueError('Unmanifested image or label files')
    if any(not counts[s]['images'] or not counts[s]['boxes'] for s in SPLITS):
        raise ValueError('Every split must contain annotated positive images')
    return {s:dict(counts[s]) for s in SPLITS}


def prepare(source, own, review_path, negative_review, output, seed=42):
    source, own, output = map(lambda p:Path(p).resolve(), (source, own, output))
    if output.exists():
        raise FileExistsError('Use a new dataset directory; existing data are never overwritten')
    review_path, negative_review = Path(review_path).resolve(), Path(negative_review).resolve()
    review = json.loads(review_path.read_text(encoding='utf-8'))
    source_images = sorted((source/'train/images').glob('*.jpg'))
    if {p.name for p in source_images} != set(review['records']):
        raise ValueError('Review must cover every source image')
    rows, excluded = [], []
    for p in source_images:
        rec = review['records'][p.name]
        label = source/'train/labels'/(p.stem+'.txt')
        if (rec.get('reviewed') is not True or digest(p) != rec['image_sha256']
                or digest(label) != rec['label_sha256'] or rec['decision'] not in {'keep','exclude'}):
            raise ValueError('Source changed or decision not reviewed')
        if rec['decision'] == 'exclude':
            excluded.append(dict(name=p.name,reason=rec['reason']))
            continue
        im = cv2.imread(str(p)); h,w = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:]
        rows.append(dict(source=p, source_label=label, episode=rec['episode'], phash=dct>np.median(dct),
                         boxes=read_boxes(label,w,h), origin='countpeople'))
    episode_splits(rows, seed)
    own_a = json.loads((own/'annotations.json').read_text(encoding='utf-8'))
    for name, rec in own_a['images'].items():
        if name in review.get('excluded_own_negatives',[]):
            continue
        p = own/name
        if rec.get('reviewed') is not True or digest(p) != rec['sha256']:
            raise ValueError('Existing local annotation changed or not reviewed')
        split = Path(name).parts[1]
        match = re.fullmatch(r'(test2|test)_(\d+)',p.stem)
        if match is None:
            raise ValueError('Local source/frame provenance missing')
        source_id, frame = match[1],int(match[2])
        rows.append(dict(source=p, split=split, group=f'own:{source_id}:{frame//1500}', boxes=rec['boxes_xyxy'], origin='own'))
    neg = json.loads(negative_review.read_text(encoding='utf-8'))
    holdout = []
    for name in own_a['images']:
        match = re.fullmatch(r'(test2|test)_(\d+)',Path(name).stem)
        if match and Path(name).parts[1] != 'train':
            holdout.append((match[1], int(match[2])))
    for rec in neg['samples']:
        p = (negative_review.parent/'negative_candidates'/rec['path']).resolve()
        if not p.is_relative_to(negative_review.parent) or rec.get('reviewed') is not True or digest(p) != rec['sha256']:
            raise ValueError('Unreviewed or changed negative image')
        source_id=Path(rec['source']).stem
        if any(s == source_id and abs(f-rec['frame']) < 1500 for s,f in holdout):
            raise ValueError('New negative is too close to an existing local holdout')
        rows.append(dict(source=p,split='train',group=f'own:{source_id}:{rec["frame"]//1500}',boxes=[],origin='new_negative',provenance=rec))
    output.mkdir(parents=True)
    for split in SPLITS:
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    images = {}
    for row in rows:
        image = cv2.imread(str(row['source']))
        if image is None:
            raise ValueError('Cannot read curated image')
        h,w = image.shape[:2]
        base = row['origin']+'_'+digest(row['source'])[:20]
        parent = str(row['source'])
        rng = np.random.default_rng(int(hashlib.sha256((str(seed)+base).encode()).hexdigest()[:16],16))
        variants = range(3) if row['split']=='train' else range(1)
        for v in variants:
            name = base + (f'_a{v}' if v else '')+'.jpg'
            path = output/'images'/row['split']/name
            if path.exists():
                raise ValueError('Duplicate source image; deduplicate reviewed inputs first')
            transform = None
            if v:
                augmented, transform = augment(image, rng, v)
                if not cv2.imwrite(str(path),augmented,[cv2.IMWRITE_JPEG_QUALITY,95]):
                    raise OSError('Cannot write augmented image')
            else:
                shutil.copyfile(row['source'],path)
            label = output/'labels'/row['split']/(path.stem+'.txt')
            write_labels(label,row['boxes'],w,h)
            images[path.relative_to(output).as_posix()] = dict(
                reviewed=True,split=row['split'],group=row['group'],origin=row['origin'],parent=parent,
                source_sha256=digest(row['source']),augmentation=transform,boxes_xyxy=row['boxes'],
                sha256=digest(path),pixel_sha256=hashlib.sha256(cv2.imread(str(path)).tobytes()).hexdigest(),
                label_sha256=digest(label),provenance=row.get('provenance'))
    (output/'annotations.json').write_text(json.dumps(dict(version=1,seed=seed,images=images,
        review_sha256=digest(review_path),negative_review_sha256=digest(negative_review),
        own_annotations_sha256=digest(own/'annotations.json'),excluded=excluded),indent=2),encoding='utf-8')
    shutil.copyfile(review_path,output/'source_review.json')
    shutil.copyfile(negative_review,output/'negative_review.json')
    (output/'data.yaml').write_text(f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n',encoding='utf-8')
    counts=verify(output)
    (output/'summary.json').write_text(json.dumps(dict(counts=counts,excluded_external=len(excluded),
        base_images=len(rows),added_negative_frames=len(neg['samples']),
        originals_by_origin_split={s:dict(Counter(r['origin'] for r in rows if r['split']==s)) for s in SPLITS}),indent=2),encoding='utf-8')
    return counts


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--own',type=Path)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--negative-review',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        result=verify(args.output)
    else:
        if any(x is None for x in (args.source,args.own,args.review,args.negative_review)):
            parser.error('Build requires source, own, review and negative-review')
        result=prepare(args.source,args.own,args.review,args.negative_review,args.output)
    print(json.dumps(result,indent=2))
