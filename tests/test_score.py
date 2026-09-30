from __future__ import annotations

import unittest
import unicodedata

from vgec.score import (
    _classification_metrics,
    extract_edits,
    prf,
    score_against_references,
    tokenize,
)


class LocalEditScorerTests(unittest.TestCase):
    def test_tokenizer_preserves_ellipsis_as_one_token(self) -> None:
        self.assertEqual(tokenize("Tôi... đi."), ["Tôi", "...", "đi", "."])

    def test_unchanged_type0_has_no_edits_or_artificial_score(self) -> None:
        result = score_against_references("Câu đúng.", "Câu đúng.", ["Câu đúng."])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))
        self.assertIsNone(result["P"])
        self.assertIsNone(result["R"])
        self.assertIsNone(result["F0.5"])

    def test_exact_correction_has_one_true_positive(self) -> None:
        result = score_against_references(
            "Tôi đang hoc bài.", "Tôi đang học bài.", ["Tôi đang học bài."]
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["F0.5"], 1.0)

    def test_canonically_equivalent_unicode_correction_matches(self) -> None:
        reference = "học."
        prediction = unicodedata.normalize("NFD", reference)
        result = score_against_references("hoc.", prediction, [reference])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["F0.5"], 1.0)

    def test_wrong_correction_has_fp_and_fn(self) -> None:
        result = score_against_references(
            "Tôi đang hoc bài.", "Tôi đang đọc bài.", ["Tôi đang học bài."]
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["F0.5"], 0.0)

    def test_multiple_references_selects_best(self) -> None:
        result = score_against_references(
            "Nó đi hoc.",
            "Nó đi học.",
            ["Nó đến trường.", "Nó đi học."],
        )
        self.assertEqual(result["reference_index"], 1)
        self.assertEqual(result["F0.5"], 1.0)

    def test_multiple_references_prefers_exact_unchanged_reference(self) -> None:
        result = score_against_references(
            "Câu gốc.", "Câu gốc.", ["Câu đã sửa.", "Câu gốc."]
        )
        self.assertEqual(result["reference_index"], 1)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))

    def test_edit_boundaries_are_deterministic(self) -> None:
        self.assertEqual(
            extract_edits("A B C", "A X C"),
            {next(iter(extract_edits("A B C", "A X C")))},
        )
        edit = next(iter(extract_edits("A B C", "A X C")))
        self.assertEqual((edit.start, edit.end, edit.replacement), (1, 2, ("X",)))

    def test_zero_denominator_policy(self) -> None:
        self.assertEqual(prf(0, 1, 0), {"P": 0.0, "R": None, "F0.5": 0.0})
        self.assertEqual(prf(0, 0, 1), {"P": None, "R": 0.0, "F0.5": 0.0})

    def test_classification_metrics(self) -> None:
        metrics = _classification_metrics(
            [
                {"error_type": 0, "predicted_type_id": 0},
                {"error_type": 1, "predicted_type_id": 1},
                {"error_type": 1, "predicted_type_id": 0},
            ]
        )
        self.assertEqual(metrics["n_eligible"], 3)
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(
            metrics["confusion_matrix"]["rows_are_gold_columns_are_prediction"][1],
            [1, 1, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
