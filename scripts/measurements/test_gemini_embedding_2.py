"""Offline checks for the measurement runner's transmission and output boundaries."""
import contextlib
import importlib.util
import io
import http.client
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('measurement', Path(__file__).with_name('gemini-embedding-2.py'))
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class MeasurementBoundaryTests(unittest.TestCase):
    def test_http_protocol_errors_are_reduced_to_an_empty_transport_result(self):
        for error in [http.client.BadStatusLine('untrusted protocol echo'),
                      http.client.IncompleteRead(b'untrusted partial echo')]:
            class Broken:
                def open(self, *args, **kwargs):
                    raise error
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                self.assertEqual(measurement.request(Broken(), 'https://example.invalid', 'embedContent', {}, 'synthetic'), (None, {}))
            self.assertEqual(output.getvalue() + errors.getvalue(), '')

    def test_redirects_and_quota_responses_stop_before_the_next_attempt(self):
        rows = [{'name': 'fixture', 'content': {'parts': [{'text': 'synthetic'}]}}]
        for status in [301, 302, 307, 308, 429]:
            attempts = []
            def respond(*args):
                attempts.append(args[2])
                return status, {}
            output = io.StringIO()
            with patch.object(sys, 'argv', ['measurement', '--execute', '--approval-ref', 'synthetic-approval',
                    '--endpoint', 'https://example.invalid/v1beta', '--case', 'fixture', '--max-requests', '2']), \
                    patch.object(sys, 'stdin', io.StringIO('synthetic')), patch.object(measurement, 'cases', return_value=rows), \
                    patch.object(measurement, 'request', side_effect=respond), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as error:
                    measurement.main()
            self.assertEqual(error.exception.code, 1)
            self.assertEqual(attempts, ['countTokens'])
            self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_system_proxy_settings_cannot_change_the_approved_route(self):
        destinations = []
        class RecordingHTTPS(urllib.request.HTTPSHandler):
            def https_open(self, req):
                destinations.append(req.host)
                raise OSError('offline fixture transport')
        build_opener = urllib.request.build_opener
        def offline_opener(*handlers):
            return build_opener(*handlers, RecordingHTTPS())
        rows = [{'name': 'fixture', 'content': {'parts': [{'text': 'synthetic'}]}}]
        with patch.object(sys, 'argv', ['measurement', '--execute', '--approval-ref', 'synthetic-approval',
                '--endpoint', 'https://example.invalid/v1beta', '--case', 'fixture', '--max-requests', '2']), \
                patch.object(sys, 'stdin', io.StringIO('synthetic')), patch.object(measurement, 'cases', return_value=rows), \
                patch.object(urllib.request, 'build_opener', side_effect=offline_opener), \
                patch.object(urllib.request, 'getproxies', return_value={'https': 'http://proxy.invalid:3128'}), \
                patch.object(urllib.request, 'proxy_bypass', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                measurement.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(destinations, ['example.invalid'])

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
        self.addCleanup(error.close)
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
