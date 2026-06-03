import json
import unittest

from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import (
    GeminiUpstreamError,
    ensure_response_text,
    extract_response_text,
    parse_upstream_error,
)
from gemini_web2api.server import GeminiHandler


class GeminiParsingTests(unittest.TestCase):
    def setUp(self):
        self._config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self._config)

    def test_extract_response_text_uses_longest_incremental_text(self):
        inner = [None] * 5
        inner[0] = "padding-" * 40
        inner[4] = [[None, ["Hello", "Hello world"]]]
        raw = json.dumps([["wrb.fr", None, json.dumps(inner)]])

        self.assertEqual(extract_response_text(raw), "Hello world")

    def test_bard_error_info_is_reported_as_upstream_error(self):
        raw = ')]}\'\nBardErrorInfo [1060]\n'
        err = parse_upstream_error(raw)

        self.assertIsInstance(err, GeminiUpstreamError)
        self.assertEqual(err.code, "bard_error_1060")
        self.assertIn("BardErrorInfo [1060]", str(err))

    def test_empty_response_policy_defaults_to_error(self):
        CONFIG["empty_response_policy"] = "error"

        with self.assertRaises(GeminiUpstreamError) as cm:
            ensure_response_text("", "no parseable Gemini text")

        self.assertEqual(cm.exception.code, "empty_response")

    def test_empty_response_policy_can_preserve_null_behavior(self):
        CONFIG["empty_response_policy"] = "null"

        self.assertEqual(ensure_response_text("", "no parseable Gemini text"), "")


class AuthTests(unittest.TestCase):
    def setUp(self):
        self._config = dict(CONFIG)
        CONFIG["api_keys"] = ["sk-test"]

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self._config)

    def _handler(self, path, headers=None):
        handler = GeminiHandler.__new__(GeminiHandler)
        handler.path = path
        handler.headers = headers or {}
        return handler

    def test_v1beta_requires_auth_when_api_keys_are_configured(self):
        handler = self._handler("/v1beta/models")

        self.assertTrue(handler._requires_auth())
        self.assertFalse(handler._authorized())

    def test_google_api_key_header_is_accepted(self):
        handler = self._handler(
            "/v1beta/models/gemini-3.5-flash:generateContent",
            {"x-goog-api-key": "sk-test"},
        )

        self.assertTrue(handler._authorized())

    def test_google_api_key_query_param_is_accepted(self):
        handler = self._handler(
            "/v1beta/models/gemini-3.5-flash:generateContent?key=sk-test"
        )

        self.assertTrue(handler._authorized())


if __name__ == "__main__":
    unittest.main()
