from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vgec.io_utils import atomic_write_json, sha256_file
from vgec.providers import ProviderError, ProviderResult
from vgec.run import _load_or_create_config, _successful_ids, build_parser, run


class QuotaProvider:
    provider_name = "openai"
    method = "zero_shot_prompt"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, input_text: str) -> ProviderResult:
        self.calls += 1
        raise ProviderError(
            "credit balance exhausted",
            retryable=False,
            category="quota",
            status_code=429,
        )


class SuccessProvider:
    provider_name = "openai"
    method = "zero_shot_prompt"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, input_text: str) -> ProviderResult:
        self.calls += 1
        return ProviderResult(
            raw_text=input_text,
            model_id="gpt-4.1-test",
            request_id=f"request-{self.calls}",
            usage={},
            raw_response={"test": True},
        )


class InvalidOutputProvider:
    provider_name = "litellm_openai_compatible"
    method = "zero_shot_correction_only"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, input_text: str) -> ProviderResult:
        self.calls += 1
        raise ProviderError(
            "JSON wrapper is not allowed",
            retryable=False,
            category="invalid_output",
            raw_output='{"fixed_sentence":"Câu."}',
        )


class RunnerQuotaTests(unittest.TestCase):
    def test_resume_rejects_generation_protocol_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "run" / "config.json"
            config_path.parent.mkdir()
            base = {
                "run_id": "run",
                "selected_model": "azure-4.1",
                "provider": "litellm_openai_compatible",
                "method": "zero_shot_type_and_correction",
                "dry_run": False,
                "prompt_version": "v2_type_and_correction",
                "prompt_sha256": "prompt",
                "data_hash": "data",
                "generation": {"temperature": 0},
            }
            atomic_write_json(config_path, base)
            changed = {**base, "generation": {"temperature": 1}}
            with self.assertRaisesRegex(ValueError, "generation"):
                _load_or_create_config(config_path, changed)

    def test_structured_success_requires_predicted_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            rows = [
                {
                    "id": 1,
                    "status": "ok",
                    "prediction": "Câu.",
                    "selected_model": "azure-4.1",
                    "predicted_type_id": None,
                },
                {
                    "id": 2,
                    "status": "ok",
                    "prediction": "Câu.",
                    "selected_model": "azure-4.1",
                    "predicted_type_id": 0,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(_successful_ids(path), {2})

    def test_correction_only_success_allows_null_predicted_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "status": "ok",
                        "prediction": "Câu.",
                        "selected_model": "azure-4.1",
                        "method": "zero_shot_correction_only",
                        "output_mode": "correction_only_text",
                        "predicted_type_id": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_successful_ids(path), {1})

    def test_quota_stops_after_first_failure_and_failed_ids_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "eval.jsonl"
            records = [
                {
                    "id": record_id,
                    "input": f"Câu {record_id}.",
                    "error_type": 0,
                    "references": [f"Câu {record_id}."],
                    "source_version": "test",
                    "source_ids": [record_id],
                    "has_ellipsis": False,
                }
                for record_id in (1, 2, 3)
            ]
            data_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            data_hash = sha256_file(data_path)
            manifest_path = root / "manifest.json"
            atomic_write_json(
                manifest_path,
                {
                    "evaluation_locked": True,
                    "unresolved_reviews_excluded": True,
                    "evaluation": {
                        "sha256": data_hash,
                        "ids": [1, 2, 3],
                    },
                },
            )
            prompt_path = root / "prompt.txt"
            prompt_path.write_text("Test prompt.", encoding="utf-8")
            runs_dir = root / "runs"
            args = build_parser().parse_args(
                [
                    "--model",
                    "gpt-4.1",
                    "--limit",
                    "3",
                    "--run-id",
                    "quota-resume",
                    "--data",
                    str(data_path),
                    "--manifest",
                    str(manifest_path),
                    "--prompt",
                    str(prompt_path),
                    "--runs-dir",
                    str(runs_dir),
                    "--max-retries",
                    "4",
                    "--retry-base-seconds",
                    "0",
                ]
            )

            quota_provider = QuotaProvider()
            with patch("vgec.run._build_provider", return_value=quota_provider):
                first = run(args)

            predictions_path = runs_dir / "quota-resume" / "predictions.jsonl"
            first_rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(quota_provider.calls, 1)
            self.assertEqual(first["attempted"], 1)
            self.assertEqual(first["failed"], 1)
            self.assertEqual(first["stop_reason"], "quota")
            self.assertEqual(len(first_rows), 1)
            self.assertEqual(first_rows[0]["attempt"], 1)
            self.assertEqual(first_rows[0]["error"]["category"], "quota")
            self.assertFalse(first_rows[0]["error"]["retryable"])
            self.assertEqual(_successful_ids(predictions_path), set())

            success_provider = SuccessProvider()
            with patch("vgec.run._build_provider", return_value=success_provider):
                resumed = run(args)

            self.assertEqual(success_provider.calls, 3)
            self.assertEqual(resumed["attempted"], 3)
            self.assertEqual(resumed["succeeded"], 3)
            self.assertIsNone(resumed["stop_reason"])
            self.assertEqual(_successful_ids(predictions_path), {1, 2, 3})

    def test_invalid_output_is_not_retried_and_remains_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "eval.jsonl"
            records = [
                {
                    "id": record_id,
                    "input": f"Câu {record_id}.",
                    "error_type": 0,
                    "references": [f"Câu {record_id}."],
                    "source_version": "test",
                    "source_ids": [record_id],
                    "has_ellipsis": False,
                }
                for record_id in (1, 2)
            ]
            data_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            data_hash = sha256_file(data_path)
            manifest_path = root / "manifest.json"
            atomic_write_json(
                manifest_path,
                {
                    "evaluation_locked": True,
                    "unresolved_reviews_excluded": True,
                    "evaluation": {"sha256": data_hash, "ids": [1, 2]},
                },
            )
            prompt_path = root / "prompt.json"
            atomic_write_json(
                prompt_path,
                {
                    "prompt_version": "v3_correction_only",
                    "system_prompt": "Sửa câu.",
                    "user_prompt_template": "Câu đầu vào:\n{sentence}",
                },
            )
            args = build_parser().parse_args(
                [
                    "--model", "azure-4.1",
                    "--experiment", "correction-only",
                    "--limit", "2",
                    "--run-id", "invalid",
                    "--data", str(data_path),
                    "--manifest", str(manifest_path),
                    "--prompt", str(prompt_path),
                    "--runs-dir", str(root / "runs"),
                    "--max-retries", "4",
                    "--retry-base-seconds", "0",
                ]
            )
            invalid_provider = InvalidOutputProvider()
            with patch("vgec.run._build_provider", return_value=invalid_provider):
                summary = run(args)
            rows = [
                json.loads(line)
                for line in (root / "runs/invalid/predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(invalid_provider.calls, 2)
            self.assertEqual(summary["failed"], 2)
            self.assertTrue(all(row["attempt"] == 1 for row in rows))
            self.assertTrue(all(row["raw_prediction"] for row in rows))
            self.assertEqual(_successful_ids(root / "runs/invalid/predictions.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
