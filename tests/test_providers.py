from __future__ import annotations

import unittest

from vgec.providers import (
    MockProvider,
    ProviderError,
    _provider_error,
    build_type_and_correction_payload,
    build_correction_only_payload,
    parse_correction_only,
    parse_type_and_correction,
)


class FakeProviderException(Exception):
    def __init__(self, *, status_code: int, body: dict):
        super().__init__("simulated provider error")
        self.status_code = status_code
        self.body = body


class MockProviderTests(unittest.TestCase):
    def test_payload_contains_only_prompt_and_input(self) -> None:
        provider = MockProvider("gpt-4.1", "fixed prompt")
        result = provider.generate("Câu đầu vào.")
        payload = result.raw_response["request_payload"]
        self.assertEqual(set(payload), {"instructions", "input"})
        self.assertEqual(payload["input"], "Câu đầu vào.")
        for forbidden in ("id", "error_type", "references", "fixed_sentence"):
            self.assertNotIn(forbidden, payload)

    def test_bartpho_mock_has_no_instruction(self) -> None:
        provider = MockProvider("bartpho-pretrained", None)
        payload = provider.generate("Câu đầu vào.").raw_response["request_payload"]
        self.assertEqual(payload, {"input": "Câu đầu vào."})


class ProviderErrorClassificationTests(unittest.TestCase):
    def test_credit_balance_429_is_terminal_quota(self) -> None:
        exc = FakeProviderException(
            status_code=429,
            body={
                "error": {
                    "message": "You have no credits remaining. Add credits to continue.",
                    "type": "insufficient_quota",
                    "code": "credit_balance_exhausted",
                }
            },
        )
        error = _provider_error(exc)
        self.assertEqual(error.category, "quota")
        self.assertFalse(error.retryable)
        self.assertEqual(error.status_code, 429)


class StructuredPredictionTests(unittest.TestCase):
    def test_parse_valid_exact_schema(self) -> None:
        self.assertEqual(
            parse_type_and_correction(
                '{"type_id": 1, "fixed_sentence": "Câu đã sửa."}'
            ),
            (1, "Câu đã sửa."),
        )

    def test_parse_rejects_extra_field_and_invalid_type(self) -> None:
        with self.assertRaises(ProviderError):
            parse_type_and_correction(
                '{"type_id": 1, "fixed_sentence": "Câu.", "reason": "x"}'
            )
        with self.assertRaises(ProviderError):
            parse_type_and_correction(
                '{"type_id": 5, "fixed_sentence": "Câu."}'
            )

    def test_payload_contains_only_prompts_and_input_sentence(self) -> None:
        payload = build_type_and_correction_payload(
            model="azure-4.1",
            system_prompt="System cố định.",
            user_prompt_template="Câu đầu vào:\n{sentence}",
            input_text="Câu thử nghiệm.",
            max_output_tokens=128,
        )
        self.assertEqual(payload["messages"][1]["content"], "Câu đầu vào:\nCâu thử nghiệm.")
        serialized = str(payload)
        for forbidden in ("record_id", "error_type", "references", "source_ids"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_correction_only_payload_has_no_json_response_format(self) -> None:
        payload = build_correction_only_payload(
            model="azure-4.1",
            system_prompt="Sửa câu.",
            user_prompt_template="Câu đầu vào:\n{sentence}",
            input_text="Một câu.",
            max_output_tokens=256,
        )
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["temperature"], 0)

    def test_correction_only_parser_preserves_sentence_content(self) -> None:
        raw = "  “Tên riêng” vẫn giữ...\nCâu thứ hai.  "
        self.assertEqual(
            parse_correction_only(raw), "“Tên riêng” vẫn giữ...\nCâu thứ hai."
        )

    def test_correction_only_parser_rejects_empty_wrappers_and_truncation(self) -> None:
        for raw in ("   ", '```text\nCâu.\n```', '{"fixed_sentence":"Câu."}', "Output: Câu."):
            with self.subTest(raw=raw), self.assertRaises(ProviderError):
                parse_correction_only(raw)
        with self.assertRaises(ProviderError):
            parse_correction_only("Câu chưa hết", finish_reason="length")

    def test_temporary_rate_limit_429_remains_retryable(self) -> None:
        exc = FakeProviderException(
            status_code=429,
            body={
                "error": {
                    "message": "Rate limit reached for requests per minute.",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            },
        )
        error = _provider_error(exc)
        self.assertEqual(error.category, "rate_limit")
        self.assertTrue(error.retryable)
        self.assertEqual(error.status_code, 429)


if __name__ == "__main__":
    unittest.main()
