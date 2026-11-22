import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.prepare_archive_dataset import digest, extract, validate_plan, verify_reviewed_dataset
from tools.annotation_review import compile_annotations
from tools.evaluate_detection_dataset import match_boxes


class ArchiveTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'FFV1'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(20):
            writer.write(np.full((48, 64, 3), 20+i*5, np.uint8))
        writer.release()
        self.plan = dict(source=str(self.source), source_sha256=digest(self.source),
                         minimum_source_gap_frames=3, policy='Synthetic exact extraction test',
                         samples=[dict(frame=2, split='train'), dict(frame=10, split='val'),
                                  dict(frame=18, split='test')])
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def reviewed(self):
        dataset = self.root / 'dataset'
        audit = extract(self.plan_path, dataset)
        for row in audit['images'].values():
            row.update(reviewed=True, boxes_xyxy=[[10, 10, 40, 45]])
        (dataset / 'annotations.json').write_text(json.dumps(audit))
        compile_annotations(dataset)
        return dataset, audit

    def test_sequential_frame_numbers_and_compiled_labels(self):
        dataset, audit = self.reviewed()
        for name, row in audit['images'].items():
            self.assertAlmostEqual(float(cv2.imread(str(dataset/name)).mean()), 20+5*row['frame'], delta=1)
        self.assertEqual(verify_reviewed_dataset(dataset)['test'], dict(images=1, boxes=1))

    def test_neighbour_leakage_rejected_before_extraction(self):
        plan = copy.deepcopy(self.plan)
        plan['samples'][1]['frame'] = 4
        with self.assertRaisesRegex(ValueError, 'leak'):
            validate_plan(plan)

    def test_duplicate_source_index_is_rejected(self):
        self.plan['samples'].append(dict(frame=2, split='train'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_plan(self.plan)

    def test_changed_source_or_existing_output_is_rejected(self):
        dataset, _ = self.reviewed()
        with self.assertRaises(FileExistsError):
            extract(self.plan_path, dataset)
        self.plan['source_sha256'] = '0'*64
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, 'Source differs'):
            extract(self.plan_path, self.root / 'other')

    def test_silently_emptying_label_is_rejected(self):
        dataset, _ = self.reviewed()
        (dataset / 'labels/train/source_00000002.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            verify_reviewed_dataset(dataset)

    def test_unreviewed_or_changed_image_is_rejected(self):
        dataset, audit = self.reviewed()
        name, row = next(iter(audit['images'].items()))
        row['reviewed'] = False
        (dataset / 'annotations.json').write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            verify_reviewed_dataset(dataset)
        row['reviewed'] = True
        (dataset / 'annotations.json').write_text(json.dumps(audit))
        cv2.imwrite(str(dataset/name), np.full((48, 64, 3), 220, np.uint8))
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_reviewed_dataset(dataset)


class DetectionMatchingTests(unittest.TestCase):
    def test_duplicate_cannot_match_one_person_twice(self):
        result = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual((result['tp'], result['fp'], result['duplicate_fp']), (1, 1, 1))

    def test_distinct_people_are_matched_one_to_one(self):
        result = match_boxes([[0, 0, 10, 10], [8, 0, 18, 10]],
                             [[5, 0, 15, 10], [0, 0, 10, 10]], .3)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 0, 0))

    def test_empty_scene_false_positive_and_full_miss(self):
        self.assertEqual(match_boxes([], [[0, 0, 1, 1]])['fp'], 1)
        self.assertEqual(match_boxes([[0, 0, 1, 1]], [])['fn'], 1)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            match_boxes([[0, 0, 0, 1]], [])
