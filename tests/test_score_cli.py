from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

from vgec.io_utils import sha256_file
from vgec.score import main, score_run


class ScoreCliIsolationTests(unittest.TestCase):
    def test_correction_only_null_type_is_valid_and_classification_is_na(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            data_path = root / "eval.jsonl"
            record = {
                "id": 1,
                "input": "Tôi đang hoc bài.",
                "error_type": 2,
                "references": ["Tôi đang học bài."],
                "has_ellipsis": False,
            }
            data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            data_hash = sha256_file(data_path)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "selected_model": "azure-4.1",
                        "actual_model_ids": [],
                        "provider": "litellm_openai_compatible",
                        "method": "zero_shot_correction_only",
                        "output_mode": "correction_only_text",
                        "prompt_version": "v3_correction_only",
                        "prompt_sha256": "v3",
                        "data_path": str(data_path),
                        "data_hash": data_hash,
                    }
                ),
                encoding="utf-8",
            )
            prediction = {
                "id": 1,
                "prediction": "Tôi đang học bài.",
                "predicted_type_id": None,
                "status": "ok",
                "usage": {},
                "data_hash": data_hash,
            }
            (run_dir / "predictions.jsonl").write_text(
                json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            metrics = score_run(run_dir)
            self.assertEqual(metrics["invalid_output_count"], 0)
            self.assertFalse(metrics["classification_applicable"])
            self.assertFalse(metrics["classification"]["applicable"])
            self.assertIsNone(metrics["classification"]["accuracy"])

    def test_partial_missing_ids_are_not_invalid_and_other_runs_are_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_dir = root / "runs"
            target_dir = runs_dir / "target"
            old_dir = runs_dir / "old"
            target_dir.mkdir(parents=True)
            old_dir.mkdir(parents=True)

            data_path = root / "eval.jsonl"
            records = [
                {
                    "id": 1,
                    "input": "Câu đúng.",
                    "error_type": 0,
                    "references": ["Câu đúng."],
                    "source_version": "test",
                    "source_ids": [1],
                    "has_ellipsis": False,
                },
                {
                    "id": 2,
                    "input": "Tôi đang hoc bài.",
                    "error_type": 1,
                    "references": ["Tôi đang học bài."],
                    "source_version": "test",
                    "source_ids": [2],
                    "has_ellipsis": False,
                },
            ]
            data_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
            data_hash = sha256_file(data_path)
            config = {
                "run_id": "target",
                "selected_model": "gpt-4.1",
                "actual_model_ids": ["mock:gpt-4.1"],
                "provider": "mock",
                "method": "dry_run_mock",
                "prompt_version": "v1",
                "prompt_sha256": "prompt-hash",
                "data_path": str(data_path),
                "data_hash": data_hash,
                "dry_run": True,
            }
            (target_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            prediction = {
                "id": 1,
                "input": "Câu đúng.",
                "prediction": unicodedata.normalize("NFD", "Câu đúng."),
                "status": "mock",
                "latency_seconds": 0.0,
                "usage": {},
                "data_hash": data_hash,
            }
            (target_dir / "predictions.jsonl").write_text(
                json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            # This incompatible completed run would make vgec.report reject a merge.
            (old_dir / "config.json").write_text(
                json.dumps({**config, "run_id": "old", "data_hash": "old-hash"}),
                encoding="utf-8",
            )
            (old_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "data_hash": "old-hash",
                        "scorer_version": "local_edit_v1",
                        "scored_ids_sha256": "old-ids",
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "--run-id",
                        "target",
                        "--runs-dir",
                        str(runs_dir),
                        "--allow-partial",
                    ]
                )

            self.assertEqual(exit_code, 0, output.getvalue())
            self.assertTrue((target_dir / "metrics.json").exists())
            self.assertTrue((target_dir / "errors.csv").exists())
            metrics = json.loads((target_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["data_hash"], data_hash)
            self.assertEqual(metrics["n_scored"], 1)
            self.assertTrue(metrics["is_partial"])
            self.assertEqual(metrics["missing_count"], 1)
            self.assertEqual(metrics["invalid_output_count"], 0)
            self.assertEqual(metrics["invalid_output_rate"], 0.0)
            self.assertEqual(metrics["scorer_version"], "local_edit_v2")
            self.assertEqual(metrics["EM"], 1.0)
            self.assertEqual(metrics["unchanged_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
