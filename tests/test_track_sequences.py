import unittest

from tools.evaluate_track_sequences import evaluate_sequence, match_targets


class TemporalSequenceTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(source='data/example.mp4', start=0, end=2)
        self.sequence = dict(id='test', people={'A': {'role': 'passenger'}},
                             samples=[dict(frame=f, decoded_sha256=str(f), camera_time=str(f),
                                           objects=[dict(person_id='A', bbox=[0, 0, 10, 10])]) for f in (0, 2)],
                             events=[dict(person_id='A', direction='IN', start=0, end=2)])
        self.trace = dict(source='data/example.mp4', start_frame=0, end_frame=2,
                          trace=[dict(frame=f, decoded_sha256=str(f), camera_time=str(f), tracker_epoch=0,
                                      detections=[dict(id=1, identity='one', conf=.9, bbox=[0, 0, 10, 10])])
                                 for f in range(3)],
                          events=[dict(track_id=1, event_type='IN', crossing_frame=1, counted=True)])

    def test_merge_cannot_resolve_two_people(self):
        targets = [dict(bbox=[0, 0, 10, 10]), dict(bbox=[0, 0, 10, 10])]
        self.assertEqual(match_targets(targets, [dict(bbox=[0, 0, 10, 10], conf=.9)]), [None, None])

    def test_close_predictions_are_unresolved(self):
        self.assertEqual(match_targets([dict(bbox=[0, 0, 10, 10])],
                                      [dict(bbox=[0, 0, 10, 10], conf=.9)] * 2), [None])

    def test_frame_hash_mismatch_rejected(self):
        self.trace['trace'][0]['decoded_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'image/time mismatch'):
            evaluate_sequence(self.sequence, self.trace, self.context)

    def test_reset_is_not_switch(self):
        self.trace['trace'][2].update(tracker_epoch=1)
        self.trace['trace'][2]['detections'][0]['id'] = 2
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual(result['trajectories']['A']['changes'], [])
        self.assertEqual(result['passages'][0]['status'], 'correct')

    def test_missing_detection_is_not_switch(self):
        self.trace['trace'][2]['detections'] = []
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual(result['trajectories']['A']['unresolved'], 1)
        self.assertEqual(result['trajectories']['A']['changes'], [])

    def test_persistent_identity_change_survives_tracker_reset(self):
        self.trace['trace'][2]['tracker_epoch'] = 1
        self.trace['trace'][2]['detections'][0].update(id=2, identity='two')
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual([r['field'] for r in result['trajectories']['A']['changes']], ['identity'])

    def test_actual_switch_and_shared_id_are_reported(self):
        self.trace['trace'][2]['detections'][0]['id'] = 2
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual(len(result['trajectories']['A']['changes']), 1)
        self.sequence['people']['B'] = {'role': 'passenger'}
        self.sequence['samples'][1]['objects'][0]['person_id'] = 'B'
        self.trace['trace'][2]['detections'][0]['id'] = 1
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual({r['field'] for r in result['shared_predicted_ids']}, {'track_id', 'identity'})

    def test_single_event_does_not_verify_two_passengers(self):
        self.sequence['people']['B'] = {'role': 'passenger'}
        self.sequence['samples'][1]['objects'][0]['person_id'] = 'B'
        self.sequence['events'].append(dict(person_id='B', direction='IN', start=0, end=2))
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual([p['status'] for p in result['passages']], ['unresolved', 'unresolved'])

    def test_role_exclusion_and_wrong_direction(self):
        self.trace['events'][0]['counted'] = False
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual(result['passages'][0]['status'], 'wrong_exclusion')
        self.sequence['events'][0]['counted'] = False
        self.assertEqual(evaluate_sequence(self.sequence, self.trace, self.context)['passages'][0]['status'], 'correct')
        self.trace['events'][0]['event_type'] = 'OUT'
        self.assertEqual(evaluate_sequence(self.sequence, self.trace, self.context)['passages'][0]['status'], 'unresolved')

    def test_pending_passage_can_be_emitted_by_recovered_track(self):
        self.trace['events'][0].update(track_id=99, identity_id='one')
        result = evaluate_sequence(self.sequence, self.trace, self.context)
        self.assertEqual(result['passages'][0]['status'], 'correct')


if __name__ == '__main__':
    unittest.main()
