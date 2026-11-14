import unittest

from tools.check_identity_pairs import evaluate_pairs


class IdentityPairTests(unittest.TestCase):
    def labels(self):
        return {'samples': [{'id': 'a', 'frame': 1, 'bbox': [0, 0, 10, 10]},
                            {'id': 'b', 'frame': 2, 'bbox': [0, 0, 10, 10]}],
                'pairs': [{'left': 'a', 'right': 'b', 'same_person': False}]}

    def test_missing_person_is_unresolved_not_a_successful_separation(self):
        result = evaluate_pairs([], self.labels())
        self.assertEqual(result['summary']['identity'], {'correct': 0, 'incorrect': 0, 'unresolved': 1})

    def test_tracker_split_cannot_hide_a_reid_merge(self):
        trace = [{'frame': f, 'detections': [{'id': f, 'identity': 'same', 'conf': .9, 'bbox': [0, 0, 10, 10]}]}
                 for f in [1, 2]]
        result = evaluate_pairs(trace, self.labels())
        self.assertEqual(result['pairs'][0]['tracker'], 'correct')
        self.assertEqual(result['pairs'][0]['identity'], 'incorrect')

    def test_overlapping_duplicate_boxes_do_not_choose_an_arbitrary_identity(self):
        trace = [{'frame': f, 'detections': [
            {'id': i, 'identity': str(i), 'conf': .9, 'bbox': [0, 0, 10, 10]} for i in [1, 2]]}
                 for f in [1, 2]]
        result = evaluate_pairs(trace, self.labels())
        self.assertEqual(result['pairs'][0]['identity'], 'unresolved')
