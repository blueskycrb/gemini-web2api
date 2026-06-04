import importlib.util
import json
import pathlib
import unittest

from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import (
    GeminiUpstreamError,
    ensure_response_text,
    extract_response_text,
    parse_upstream_error,
    raise_http_upstream_error,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gemini_web2api_monolith",
    ROOT / "gemini_web2api.py",
)
monolith = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monolith)


class PackageUpstreamErrorTests(unittest.TestCase):
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

    def test_bard_error_info_becomes_structured_error(self):
        err = parse_upstream_error(")]}'\nBardErrorInfo [1060]\n")

        self.assertIsInstance(err, GeminiUpstreamError)
        self.assertEqual(err.code, "bard_error_1060")
        self.assertIn("BardErrorInfo [1060]", str(err))

    def test_http_error_body_can_report_bard_error_info(self):
        with self.assertRaises(GeminiUpstreamError) as cm:
            raise_http_upstream_error(400, "BardErrorInfo [1060]")

        self.assertEqual(cm.exception.code, "bard_error_1060")

    def test_empty_response_defaults_to_error(self):
        CONFIG["empty_response_policy"] = "error"

        with self.assertRaises(GeminiUpstreamError) as cm:
            ensure_response_text("", "no parseable Gemini text")

        self.assertEqual(cm.exception.code, "empty_response")

    def test_empty_response_can_keep_old_null_behavior(self):
        CONFIG["empty_response_policy"] = "null"

        self.assertEqual(ensure_response_text("", "no parseable Gemini text"), "")


class MonolithUpstreamErrorTests(unittest.TestCase):
    def setUp(self):
        self._config = dict(monolith.CONFIG)
        self._inline_cookie = monolith.INLINE_COOKIE
        self._inline_sapisid = monolith.INLINE_SAPISID

    def tearDown(self):
        monolith.CONFIG.clear()
        monolith.CONFIG.update(self._config)
        monolith.INLINE_COOKIE = self._inline_cookie
        monolith.INLINE_SAPISID = self._inline_sapisid

    def test_monolith_detects_bard_error_info(self):
        err = monolith.parse_upstream_error("BardErrorInfo [1060]")

        self.assertIsInstance(err, monolith.GeminiUpstreamError)
        self.assertEqual(err.code, "bard_error_1060")

    def test_monolith_http_error_body_can_report_bard_error_info(self):
        with self.assertRaises(monolith.GeminiUpstreamError) as cm:
            monolith.raise_http_upstream_error(400, "BardErrorInfo [1060]")

        self.assertEqual(cm.exception.code, "bard_error_1060")

    def test_monolith_call_gemini_rejects_empty_upstream_response(self):
        handler = monolith.GeminiHandler.__new__(monolith.GeminiHandler)
        original = monolith.gemini_stream_generate
        monolith.gemini_stream_generate = lambda *_args, **_kwargs: "BardErrorInfo [1060]"
        try:
            with self.assertRaises(monolith.GeminiUpstreamError) as cm:
                handler._call_gemini("prompt", 1, 4, None)
        finally:
            monolith.gemini_stream_generate = original

        self.assertEqual(cm.exception.code, "bard_error_1060")

    def test_monolith_inline_cookie_takes_precedence(self):
        monolith.CONFIG["cookie_file"] = "does-not-exist.txt"
        monolith.INLINE_COOKIE = "SID=sid;SAPISID=sapisid; __Secure-1PSID=psid"

        self.assertEqual(
            monolith.load_cookie(),
            ("SID=sid;SAPISID=sapisid; __Secure-1PSID=psid", "sapisid"),
        )

    def test_monolith_inline_cookie_can_override_sapisid(self):
        monolith.INLINE_COOKIE = "SID=sid; SAPISID=from_cookie"
        monolith.INLINE_SAPISID = "from_override"

        self.assertEqual(
            monolith.load_cookie(),
            ("SID=sid; SAPISID=from_cookie", "from_override"),
        )

    def test_monolith_json_cookie_content_still_parses(self):
        raw = json.dumps({
            "cookie": "SID=sid; SAPISID=from_json",
            "sapisid": "from_json",
        })

        self.assertEqual(
            monolith.parse_cookie_content(raw),
            ("SID=sid; SAPISID=from_json", "from_json"),
        )


if __name__ == "__main__":
    unittest.main()
