from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from vgec.report import _comparison_rows, _validate_protocol


def entry(*, data_hash: str = "data-a", scored_hash: str = "ids-a"):
    return {
        "metrics": {
            "data_hash": data_hash,
            "scorer_version": "local_edit_v1",
            "scored_ids_sha256": scored_hash,
            "is_partial": False,
            "method": "zero_shot_prompt",
            "prompt_sha256": "prompt-a",
            "model": "gpt-4.1",
            "provider": "openai",
            "prompt_version": "v1",
        }
    }


class ReportProtocolTests(unittest.TestCase):
    def test_accepts_matching_protocol(self) -> None:
        _validate_protocol([entry(), entry()], allow_partial=False)

    def test_rejects_different_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset hashes"):
            _validate_protocol(
                [entry(data_hash="data-a"), entry(data_hash="data-b")],
                allow_partial=False,
            )

    def test_rejects_different_scored_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "different ID sets"):
            _validate_protocol(
                [entry(scored_hash="ids-a"), entry(scored_hash="ids-b")],
                allow_partial=False,
            )

    def test_accepts_matching_structured_proxy_protocol(self) -> None:
        first = entry()
        second = entry()
        for item, model in ((first, "azure-4.1"), (second, "azure-4o")):
            item["metrics"].update(
                {
                    "method": "zero_shot_type_and_correction",
                    "prompt_version": "v2_type_and_correction",
                    "model": model,
                    "provider": "litellm_openai_compatible",
                }
            )
        _validate_protocol([first, second], allow_partial=False)

    def test_experimental_mode_allows_predeclared_prompt_and_method_difference(self) -> None:
        baseline = entry()
        baseline["metrics"].update(
            {
                "method": "zero_shot_type_and_correction",
                "prompt_version": "v2_type_and_correction",
                "prompt_sha256": "prompt-v2",
                "model": "azure-4.1",
            }
        )
        correction = entry()
        correction["metrics"].update(
            {
                "method": "zero_shot_correction_only",
                "prompt_version": "v3_correction_only",
                "prompt_sha256": "prompt-v3",
                "output_mode": "correction_only_text",
                "model": "azure-4.1",
            }
        )
        with self.assertRaisesRegex(ValueError, "different prompts"):
            _validate_protocol([baseline, correction], allow_partial=False)
        _validate_protocol(
            [baseline, correction],
            allow_partial=False,
            experimental_comparison=True,
        )

    def test_experimental_mode_still_rejects_dataset_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset hashes"):
            _validate_protocol(
                [entry(data_hash="a"), entry(data_hash="b")],
                allow_partial=False,
                experimental_comparison=True,
            )

    def test_comparison_supports_only_two_selected_azure_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "eval.jsonl"
            data_path.write_text(
                json.dumps(
                    {"id": 1, "input": "Câu.", "references": ["Câu."], "error_type": 0}
                )
                + "\n",
                encoding="utf-8",
            )
            entries = []
            for model in ("azure-4.1", "azure-4o"):
                run_dir = root / model
                run_dir.mkdir()
                (run_dir / "predictions.jsonl").write_text(
                    json.dumps({"id": 1, "status": "ok", "prediction": "Câu."})
                    + "\n",
                    encoding="utf-8",
                )
                entries.append(
                    {
                        "run_dir": run_dir,
                        "config": {"data_path": str(data_path), "started_at_utc": model},
                        "metrics": {"model": model},
                    }
                )
            rows = _comparison_rows(entries, 20)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Azure-4.1"], "Câu.")
            self.assertEqual(rows[0]["Azure-4o"], "Câu.")


if __name__ == "__main__":
    unittest.main()
