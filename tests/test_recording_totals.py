from dataclasses import replace
from types import SimpleNamespace
import unittest

from config.settings import Settings
from core.service import RecordingTotals


class RecordingTotalsTests(unittest.TestCase):
    def test_record_counts_survive_midnight_and_restart(self):
        first = RecordingTotals()
        first.add([SimpleNamespace(counted=True, event_type='IN'), SimpleNamespace(counted=False, event_type='IN')])
        payload = dict(record_totals=first.snapshot())
        restored = RecordingTotals()
        restored.restore(payload, Settings())
        restored.add([SimpleNamespace(counted=True, event_type='IN'), SimpleNamespace(counted=True, event_type='OUT')])
        self.assertEqual(restored.snapshot(), {'in':2, 'out':1, 'complete':True})

    def test_legacy_single_day_can_be_recovered_but_midnight_is_not_guessed(self):
        cfg = replace(Settings(), TIME_SOURCE='pts', VIDEO_START='2026-11-01T08:00:00+04:00')
        totals = RecordingTotals()
        payload = dict(engine=dict(day='2026-11-01', total_in=5, total_out=3))
        totals.restore(payload, cfg)
        self.assertTrue(totals.complete)
        payload['engine']['day'] = '2026-11-02'
        totals.restore(payload, cfg)
        self.assertFalse(totals.complete)

    def test_invalid_checkpoint_totals_are_rejected(self):
        for counts in [{'in':-1,'out':0,'complete':True}, {'in':True,'out':0,'complete':True}, None]:
            with self.assertRaises((ValueError, KeyError)):
                RecordingTotals().restore(dict(record_totals=counts), Settings())
