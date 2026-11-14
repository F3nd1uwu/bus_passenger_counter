from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np

from tools.prepare_countpeople import augment, digest, episode_splits, prepare, verify


class CountPeoplePreparationTests(unittest.TestCase):
    def test_photometric_variants_preserve_geometry_and_are_deterministic(self):
        image = np.random.default_rng(9).integers(20,230,(48,64,3),dtype=np.uint8)
        for variant in (1,2):
            first, meta = augment(image,np.random.default_rng(42),variant)
            second, _ = augment(image,np.random.default_rng(42),variant)
            self.assertEqual(first.shape,image.shape)
            self.assertEqual(first.dtype,np.uint8)
            np.testing.assert_array_equal(first,second)
            self.assertFalse(np.array_equal(first,image))
            self.assertIn(meta['kind'],{'exposure','noise_compression'})

    def test_adjacent_clips_and_visual_neighbours_stay_in_one_split(self):
        rng=np.random.default_rng(3)
        rows=[]
        for day in (1,2,3):
            for minute in (0,1,20,40):
                episode=f'2016_04_{day:02}_08_{minute:02}_00FrontColor'
                for _ in range(2):
                    rows.append(dict(episode=episode,phash=rng.integers(0,2,63,dtype=np.uint8)))
        rows[-1]['phash']=rows[0]['phash'].copy()
        episode_splits(rows)
        mapping={r['episode']:(r['group'],r['split']) for r in rows}
        for day in (1,2,3):
            self.assertEqual(mapping[f'2016_04_{day:02}_08_00_00FrontColor'],
                             mapping[f'2016_04_{day:02}_08_01_00FrontColor'])
        self.assertEqual(rows[0]['split'],rows[-1]['split'])

    def fixture(self, root):
        source,own=root/'source',root/'own'
        p=source/'train/images/sample.jpg';p.parent.mkdir(parents=True)
        cv2.imwrite(str(p),np.random.default_rng(1).integers(0,256,(48,64,3),dtype=np.uint8))
        label=source/'train/labels/sample.txt';label.parent.mkdir(parents=True)
        label.write_text('0 0.5 0.5 0.5 0.5\n')
        review=root/'review.json';review.write_text(json.dumps({'records':{'sample.jpg':dict(
            reviewed=True,decision='keep',reason='reviewed',image_sha256=digest(p),label_sha256=digest(label),
            episode='2016_04_01_08_00_00FrontColor')}}))
        records={}
        for i,split in enumerate(('train','val','test')):
            name=f'images/{split}/test_{i*3000:08d}.jpg';image=own/name;image.parent.mkdir(parents=True)
            cv2.imwrite(str(image),np.random.default_rng(10+i).integers(0,256,(48,64,3),dtype=np.uint8))
            records[name]=dict(reviewed=True,sha256=digest(image),boxes_xyxy=[[16,12,48,36]])
        (own/'annotations.json').write_text(json.dumps({'images':records}))
        negative=root/'negative.json';negative.write_text(json.dumps({'samples':[]}))
        return source,own,review,negative,root/'prepared'

    def test_build_preserves_sources_and_rejects_overwrite_and_holdout_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(Path(folder))
            before=digest(args[0]/'train/images/sample.jpg')
            result=prepare(*args)
            self.assertEqual(result['train']['images'],6)
            self.assertEqual(result['val']['augmented'],0)
            self.assertEqual(result['test']['augmented'],0)
            self.assertEqual(before,digest(args[0]/'train/images/sample.jpg'))
            with self.assertRaises(FileExistsError):
                prepare(*args)
            path=args[-1]/'annotations.json';data=json.loads(path.read_text())
            train=next(v for v in data['images'].values() if v['split']=='train')
            val=next(v for v in data['images'].values() if v['split']=='val')
            val['group']=train['group'];path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'leak'):
                verify(args[-1])

    def test_source_label_change_invalidates_review_before_build(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(Path(folder))
            (args[0]/'train/labels/sample.txt').write_text('')
            with self.assertRaisesRegex(ValueError,'Source changed'):
                prepare(*args)
            self.assertFalse(args[-1].exists())
