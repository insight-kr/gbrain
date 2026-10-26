"""Offline checks for the measurement runner's transmission and output boundaries."""
import contextlib
import importlib.util
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('measurement', Path(__file__).with_name('gemini-embedding-2.py'))
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class MeasurementBoundaryTests(unittest.TestCase):
    def test_default_invocation_never_reads_credentials_or_sends_requests(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['measurement']), contextlib.redirect_stdout(output), \
                patch.object(sys, 'stdin', None), \
                patch.object(measurement, 'request', side_effect=AssertionError('Unexpected transmission')):
            measurement.main()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['mode'], 'offline-plan')
        self.assertIn('overflow-false', plan['cases'])
        self.assertIn('png-384-caption', plan['cases'])
        self.assertIn('jpeg-384', plan['cases'])

    def test_provider_echoes_are_never_read_or_reported(self):
        echoed = io.BytesIO(b'private echo')
        error = urllib.error.HTTPError('https://example.invalid', 403, 'denied', {}, echoed)
        class Denied:
            def open(self, *args, **kwargs):
                raise error
        status, data = measurement.request(Denied(), 'https://example.invalid/v1beta', 'embedContent', {}, 'synthetic')
        self.assertEqual((status, data), (403, {}))
        self.assertEqual(echoed.tell(), 0)

    def test_output_keeps_measurements_and_discards_untrusted_fields(self):
        result = measurement.summary(200, {'embedding': {'values': [0.5, 1]},
            'usageMetadata': {'promptTokenCount': 17, 'promptTokenDetails': [{'modality': 'IMAGE', 'tokenCount': 16}],
                              'echo': 'private echo'}, 'echo': 'private echo'})
        self.assertEqual(result['dimensions'], 2)
        self.assertEqual(result['promptTokenCount'], 17)
        self.assertEqual(result['promptTokenDetails'], [{'modality': 'IMAGE', 'tokenCount': 16}])
        self.assertNotIn('private echo', json.dumps(result))
        self.assertFalse(measurement.summary(200, {'embedding': {'values': [float('inf')]}})['finiteVector'])


if __name__ == '__main__':
    unittest.main()
