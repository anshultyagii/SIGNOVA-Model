# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit & API Tests for FastAPI Backend
===================================================
Tests cover:
  1. GET /health: application status and vocabulary class count.
  2. POST /translate: successful known sequence translation.
  3. POST /translate: unknown gloss and safe fallback handling.
  4. POST /translate: empty gloss sequence handling.
  5. POST /translate: malformed payload validation (HTTP 422).
  6. POST /translate: oversized sequence rejection (HTTP 400/422).
  7. CORS headers for authorized local development origin.
"""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app, MAX_GLOSS_SEQUENCE_LEN, _translator


class TestSignSpeakAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if hasattr(_translator, "set_mode"):
            _translator.set_mode("rules")

    # ── 1. GET /health ────────────────────────────────────────────────────────

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data.get("status"), ("healthy", "ok"))
        self.assertEqual(data.get("supported_classes_count"), 25)
        self.assertIn("version", data)

    # ── 2. POST /translate: Known Sequences ───────────────────────────────────

    def test_translate_known_sequence(self):
        payload = {
            "glosses": ["I_Me", "Want_Desire_Wish_For", "Water"]
        }
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "I want water.")
        self.assertFalse(data["is_fallback"])
        self.assertEqual(data["normalized_glosses"], ["I_Me", "Want_Desire_Wish_For", "Water"])
        self.assertIsNotNone(data.get("matched_rule"))

    def test_translate_with_digit_prefix_normalization(self):
        payload = {
            "glosses": ["1_One", "Water"]
        }
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "One water.")
        self.assertEqual(data["normalized_glosses"], ["One", "Water"])
        self.assertFalse(data["is_fallback"])

    # ── 3. POST /translate: Unknown Gloss Fallback ────────────────────────────

    def test_translate_unknown_gloss_fallback(self):
        payload = {
            "glosses": ["UnseenCustomSign", "Water"]
        }
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "UnseenCustomSign water.")
        self.assertTrue(data["is_fallback"])

    def test_translate_unmodeled_combination_safe_fallback(self):
        payload = {
            "glosses": ["What", "Water"]
        }
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "What water.")
        self.assertTrue(data["is_fallback"])

    # ── 4. POST /translate: Empty Sequence ────────────────────────────────────

    def test_translate_empty_sequence(self):
        payload = {"glosses": []}
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "")
        self.assertEqual(data["normalized_glosses"], [])
        self.assertFalse(data["is_fallback"])

    def test_translate_empty_strings_and_invalid_tokens(self):
        payload = {"glosses": ["", "-", "unknown"]}
        response = self.client.post("/translate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["translation"], "")
        self.assertEqual(data["normalized_glosses"], [])

    # ── 5. POST /translate: Malformed Payloads ─────────────────────────────────

    def test_malformed_payload_missing_glosses(self):
        response = self.client.post("/translate", json={})
        self.assertEqual(response.status_code, 422)

    def test_malformed_payload_wrong_type(self):
        # glosses is a string instead of a list
        response = self.client.post("/translate", json={"glosses": "Water"})
        self.assertEqual(response.status_code, 422)

    def test_malformed_payload_null(self):
        response = self.client.post("/translate", json={"glosses": None})
        self.assertEqual(response.status_code, 422)

    # ── 6. POST /translate: Oversized Sequence ─────────────────────────────────

    def test_oversized_sequence_rejected(self):
        oversized = ["Water"] * (MAX_GLOSS_SEQUENCE_LEN + 1)
        response = self.client.post("/translate", json={"glosses": oversized})
        self.assertIn(response.status_code, (400, 422))

    # ── 7. CORS Configuration ─────────────────────────────────────────────────

    def test_cors_headers_for_allowed_origin(self):
        response = self.client.options(
            "/translate",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            }
        )
        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173"
        )


    # ── 8. Translator Mode Switching ──────────────────────────────────────────

    def test_translator_mode_endpoints(self):
        # Check current mode
        res_get = self.client.get("/translator/mode")
        self.assertEqual(res_get.status_code, 200)
        self.assertIn("mode", res_get.json())

        # Switch to model mode
        res_post = self.client.post("/translator/mode", json={"mode": "model"})
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["mode"], "model")

        # Test translation in model mode
        payload = {"glosses": ["I_Me", "Want_Desire_Wish_For", "Water"]}
        res_trans = self.client.post("/translate", json=payload)
        self.assertEqual(res_trans.status_code, 200)
        self.assertIn("water", res_trans.json()["translation"].lower())
        self.assertEqual(res_trans.json()["model_mode"], "model")

        # Invalid mode rejected
        res_inv = self.client.post("/translator/mode", json={"mode": "invalid_engine"})
        self.assertEqual(res_inv.status_code, 400)


if __name__ == "__main__":
    unittest.main()
