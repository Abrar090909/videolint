"""Jev adapter tests use mocked HTTP and never consume API quota."""
import io
import json
import unittest
from unittest.mock import patch

from videolint.ai import AIProviderError, get_jev_provider, get_provider, provider_configured
from videolint.jev_provider import JevProvider


class JevProviderTests(unittest.TestCase):
    def provider(self, answer, requests):
        def opener(request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(json.dumps({"answers": {"finding": answer}}).encode())
        return JevProvider(api_key="test-only", opener=opener)

    def test_focused_request_and_probability_mapping(self):
        requests = []
        provider = self.provider({"type": "noul", "noul": .92}, requests)
        result = provider.judge("Does the cut interrupt a thought?", {"cutTimestampMs": 4000})
        self.assertEqual((result.verdict, result.confidence), ("yes", .92))
        self.assertIn("92%", result.reason)
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
        self.assertEqual(timeout, 45)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(body["state"], {"cutTimestampMs": 4000})
        self.assertEqual(body["questions"]["finding"]["type"], "noul")

    def test_no_and_uncertain_do_not_become_positive(self):
        for value, expected in ((.1, "no"), (.5, "uncertain")):
            with self.subTest(value=value):
                result = self.provider({"type": "noul", "noul": value}, []).judge("Question?", {})
                self.assertEqual(result.verdict, expected)

    def test_invalid_response_and_rate_limit_fail_closed(self):
        for answer in ({"type": "noul", "noul": True}, {"type": "noul", "noul": 1.2},
                       {"type": "other", "noul": .9}):
            with self.subTest(answer=answer), self.assertRaises(AIProviderError):
                self.provider(answer, []).judge("Question?", {})
        error = RuntimeError("secret response body")
        error.code = 429
        def rate_limited(request, timeout):
            raise error
        with self.assertRaisesRegex(AIProviderError, "HTTP 429") as raised:
            JevProvider("test-only", opener=rate_limited).judge("Question?", {})
        self.assertNotIn("secret response body", str(raised.exception))

    def test_provider_selection_and_missing_key(self):
        with patch.dict("os.environ", {"VIDEOLINT_AI_PROVIDER": "jev", "BEATAPI_API_KEY": "",
                                    "OPENROUTER_API_KEY": "", "TYPESAFE_API_KEY": ""}):
            self.assertFalse(provider_configured())
            with self.assertRaisesRegex(AIProviderError, "Jev API key not configured"):
                get_jev_provider()
        with patch.dict("os.environ", {"VIDEOLINT_AI_PROVIDER": "jev", "BEATAPI_API_KEY": "",
                                    "OPENROUTER_API_KEY": "", "TYPESAFE_API_KEY": "test-only"}):
            self.assertTrue(provider_configured())
            self.assertIsInstance(get_jev_provider(), JevProvider)

    def test_beatapi_free_endpoint_is_preferred_and_never_paid(self):
        with patch.dict("os.environ", {"VIDEOLINT_AI_PROVIDER": "jev", "BEATAPI_API_KEY": "beat-test",
                                    "OPENROUTER_API_KEY": "router-test", "TYPESAFE_API_KEY": "direct-test"}):
            provider = get_jev_provider()
        self.assertEqual(provider.model, "jev-1.13-free")
        self.assertEqual(provider.url, "https://api.beatapi.io/v1/systemone")
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"answers": {"finding": {"type": "noul", "noul": .85}},
                                         "usage": {"input_tokens": 42}}).encode())
        provider._opener = opener
        result = provider.judge("Is this cut interrupted?", {"cutTimestampMs": 1000})
        self.assertEqual(result.verdict, "yes")
        self.assertEqual(json.loads(requests[0].data)["model"], "jev-1.13-free")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer beat-test")
        self.assertEqual(provider.last_cost_usd, 0.0)

    def test_openrouter_tutorial_endpoint_and_reported_cost(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"answers": {"finding": {"type": "noul", "noul": .95}},
                                         "usage": {"cost": .00002}}).encode())
        provider = JevProvider("test-only", opener=opener, openrouter=True)
        provider.judge("Is it interrupted?", {"cutTimestampMs": 1000})
        self.assertEqual(provider.model, "typesafe/jev-1.13")
        self.assertEqual(requests[0].full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(provider.last_cost_usd, .00002)


if __name__ == "__main__":
    unittest.main()
