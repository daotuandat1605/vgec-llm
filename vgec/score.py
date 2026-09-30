from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_csv,
)


SCORER_VERSION = "local_edit_v2"
TOKEN_PATTERN = re.compile(r"\.\.\.|\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, order=True)
class Edit:
    start: int
    end: int
    replacement: tuple[str, ...]


def normalize_text(text: str) -> str:
    """Normalize canonically equivalent Unicode strings before comparison."""
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer over NFC Unicode words and separate punctuation."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def extract_edits(source: str, target: str) -> set[Edit]:
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    matcher = difflib.SequenceMatcher(
        a=source_tokens, b=target_tokens, autojunk=False
    )
    return {
        Edit(start=i1, end=i2, replacement=tuple(target_tokens[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if tp == 0:
        f05 = 0.0 if fp or fn else None
    elif precision is None or recall is None:
        f05 = None
    else:
        denominator = 0.25 * precision + recall
        f05 = 1.25 * precision * recall / denominator if denominator else None
    return {"P": precision, "R": recall, "F0.5": f05}


def score_against_references(
    source: str, prediction: str, references: list[str]
) -> dict[str, Any]:
    if not references:
        raise ValueError("At least one reference is required")
    predicted_edits = extract_edits(source, prediction)
    candidates: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        gold_edits = extract_edits(source, reference)
        tp = len(predicted_edits & gold_edits)
        fp = len(predicted_edits - gold_edits)
        fn = len(gold_edits - predicted_edits)
        metrics = prf(tp, fp, fn)
        candidates.append(
            {
                "reference_index": index,
                "reference": reference,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                **metrics,
            }
        )

    # Predeclared multi-reference rule: maximize sentence F0.5, then TP, then
    # minimize FP+FN, then use source order. None ranks below a numerical score.
    def rank(item: dict[str, Any]) -> tuple[float, int, int, int]:
        f05 = item["F0.5"]
        # Identical empty edit sets are a perfect match for reference selection,
        # although corpus P/R/F remain undefined when every denominator is zero.
        selection_score = (
            1.0
            if item["tp"] == item["fp"] == item["fn"] == 0
            else (-1.0 if f05 is None else float(f05))
        )
        return (
            selection_score,
            item["tp"],
            -(item["fp"] + item["fn"]),
            -item["reference_index"],
        )

    return max(candidates, key=rank)


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("id")
        if not isinstance(record_id, int):
            raise ValueError("Prediction row has a missing/non-integer id")
        result[record_id] = row
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    result = {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
    if rows:
        result["EM"] = sum(row["exact_match"] for row in rows) / len(rows)
        result["unchanged_rate"] = sum(row["unchanged"] for row in rows) / len(rows)
    else:
        result["EM"] = None
        result["unchanged_rate"] = None
    return result


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    totals = Counter()
    observed = False
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        for canonical, keys in aliases.items():
            for key in keys:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[canonical] += value
                    observed = True
                    break
    return {
        key: totals[key] if observed else None
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for row in rows:
        matrix[row["error_type"]][row["predicted_type_id"]] += 1
    per_type: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for type_id in range(5):
        true_positive = matrix[type_id][type_id]
        support = sum(matrix[type_id])
        predicted = sum(matrix[gold][type_id] for gold in range(5))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_type[str(type_id)] = {
            "support": support,
            "predicted": predicted,
            "tp": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(matrix[type_id][type_id] for type_id in range(5))
    return {
        "n_eligible": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(f1_values) / 5 if rows else None,
        "per_type": per_type,
        "confusion_matrix": {
            "labels": [0, 1, 2, 3, 4],
            "rows_are_gold_columns_are_prediction": matrix,
        },
        "zero_division": 0.0,
    }


def score_run(
    run_dir: Path,
    data_path: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    if data_path is None:
        data_path = Path(config["data_path"])
    data = read_jsonl(data_path)
    data_hash = sha256_file(data_path)
    if data_hash != config.get("data_hash"):
        raise ValueError("Dataset hash differs from run config")
    predictions_path = run_dir / "predictions.jsonl"
    all_prediction_rows = read_jsonl(predictions_path)
    for row in all_prediction_rows:
        if row.get("data_hash") != data_hash:
            raise ValueError(f"Prediction ID {row.get('id')} has a mismatched data hash")
    latest = _latest_by_id(all_prediction_rows)
    requires_type_prediction = config.get("method") == "zero_shot_type_and_correction"
    expected_ids = {row["id"] for row in data}
    unexpected = sorted(set(latest) - expected_ids)
    if unexpected:
        raise ValueError(f"Predictions contain IDs outside this dataset: {unexpected}")

    missing_ids = sorted(expected_ids - set(latest))
    invalid_ids = sorted(
        record_id
        for record_id, row in latest.items()
        if row.get("status") not in {"ok", "mock"}
        or not isinstance(row.get("prediction"), str)
        or not row["prediction"]
        or (
            requires_type_prediction
            and (
                isinstance(row.get("predicted_type_id"), bool)
                or not isinstance(row.get("predicted_type_id"), int)
                or row["predicted_type_id"] not in range(5)
            )
        )
    )
    if (missing_ids or invalid_ids) and not allow_partial:
        raise ValueError(
            f"Run is incomplete: {len(missing_ids)} missing, {len(invalid_ids)} invalid/error. "
            "Use --allow-partial only for an explicitly partial result."
        )

    detail_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for record in data:
        prediction_row = latest.get(record["id"])
        if prediction_row is None:
            error_rows.append(
                {
                    "id": record["id"],
                    "status": "missing",
                    "error_type": record["error_type"],
                    "has_ellipsis": record["has_ellipsis"],
                    "input": record["input"],
                    "prediction": "",
                    "predicted_type_id": "",
                    "references": json.dumps(record["references"], ensure_ascii=False),
                    "chosen_reference": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "exact_match": "",
                    "unchanged": "",
                }
            )
            continue
        if record["id"] in invalid_ids:
            error_rows.append(
                {
                    "id": record["id"],
                    "status": prediction_row.get("status", "invalid"),
                    "error_type": record["error_type"],
                    "has_ellipsis": record["has_ellipsis"],
                    "input": record["input"],
                    "prediction": prediction_row.get("prediction") or "",
                    "predicted_type_id": prediction_row.get("predicted_type_id", ""),
                    "references": json.dumps(record["references"], ensure_ascii=False),
                    "chosen_reference": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "exact_match": "",
                    "unchanged": "",
                }
            )
            continue
        prediction = prediction_row["prediction"]
        normalized_prediction = normalize_text(prediction)
        chosen = score_against_references(
            record["input"], prediction, record["references"]
        )
        detail = {
            "id": record["id"],
            "status": prediction_row["status"],
            "error_type": record["error_type"],
            "has_ellipsis": bool(record["has_ellipsis"]),
            "input": record["input"],
            "prediction": prediction,
            "predicted_type_id": prediction_row.get("predicted_type_id"),
            "references": json.dumps(record["references"], ensure_ascii=False),
            "chosen_reference": chosen["reference"],
            "tp": chosen["tp"],
            "fp": chosen["fp"],
            "fn": chosen["fn"],
            "exact_match": normalized_prediction
            in {normalize_text(reference) for reference in record["references"]},
            "unchanged": normalized_prediction == normalize_text(record["input"]),
        }
        detail_rows.append(detail)
        error_rows.append(detail)

    overall = _aggregate(detail_rows)
    by_type = {
        str(error_type): {
            "n_total": sum(row["error_type"] == error_type for row in data),
            **_aggregate(
                [row for row in detail_rows if row["error_type"] == error_type]
            ),
        }
        for error_type in range(5)
    }
    ellipsis_breakdown = {
        label: {
            "n_total": sum(bool(row["has_ellipsis"]) == flag for row in data),
            **_aggregate(
                [row for row in detail_rows if bool(row["has_ellipsis"]) == flag]
            ),
        }
        for label, flag in (("with_ellipsis", True), ("without_ellipsis", False))
    }
    type0_rows = [row for row in detail_rows if row["error_type"] == 0]
    type0_overcorrection = (
        sum(not row["unchanged"] for row in type0_rows) / len(type0_rows)
        if type0_rows
        else None
    )
    scored_ids = [row["id"] for row in detail_rows]
    usage = _sum_usage(all_prediction_rows)
    latency = sum(
        float(row.get("latency_seconds", 0.0) or 0.0) for row in all_prediction_rows
    )
    classification_conflict_ids = sorted(
        record["id"]
        for record in data
        if record["error_type"] != 0
        and normalize_text(record["input"])
        in {normalize_text(reference) for reference in record["references"]}
    )
    typed_rows = [
        row
        for row in detail_rows
        if isinstance(row.get("predicted_type_id"), int)
        and not isinstance(row.get("predicted_type_id"), bool)
        and row["predicted_type_id"] in range(5)
    ]
    classification_rows = [
        row for row in typed_rows if row["id"] not in classification_conflict_ids
    ]
    classification = {
        "applicable": requires_type_prediction,
        "available": requires_type_prediction and bool(typed_rows),
        "n_with_prediction": len(typed_rows),
        "n_excluded_conflicts_in_dataset": len(classification_conflict_ids),
        "excluded_conflict_ids": classification_conflict_ids,
        "n_excluded_conflicts_scored": sum(
            row["id"] in classification_conflict_ids for row in typed_rows
        ),
        **_classification_metrics(classification_rows),
        "definition": (
            "Gold is error_type from eval_clean; predicted_type_id never replaces gold. "
            "Rows whose nonzero source label conflicts with an input-identical reference "
            "are excluded from the primary classification denominator."
        ),
    }
    metrics = {
        "run_id": config["run_id"],
        "model": config["selected_model"],
        "actual_model_ids": config.get("actual_model_ids", []),
        "provider": config["provider"],
        "method": config["method"],
        "output_mode": config.get(
            "output_mode",
            "correction_only_text"
            if config.get("method") == "zero_shot_correction_only"
            else "type_and_correction_json"
            if config.get("method") == "zero_shot_type_and_correction"
            else "correction_text",
        ),
        "prompt_version": config["prompt_version"],
        "prompt_sha256": config.get("prompt_sha256"),
        "data_hash": data_hash,
        "scorer_version": SCORER_VERSION,
        "score_label": "local_edit_P/R/F0.5",
        "is_partial": len(detail_rows) != len(data),
        "n_total": len(data),
        "n_scored": len(detail_rows),
        "coverage": len(detail_rows) / len(data) if data else None,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "invalid_output_count": len(invalid_ids),
        "invalid_ids": invalid_ids,
        # Missing IDs measure run coverage and are not invalid model outputs.
        # For an explicitly partial run, compute this rate only over predictions
        # that were successfully evaluated (n_scored).
        "invalid_output_rate": len(invalid_ids) / len(detail_rows)
        if detail_rows
        else None,
        "scored_ids_sha256": sha256_text(
            json.dumps(scored_ids, separators=(",", ":"))
        ),
        "scored_ids": scored_ids,
        "tp": overall["tp"],
        "fp": overall["fp"],
        "fn": overall["fn"],
        "P": overall["P"],
        "R": overall["R"],
        "F0.5": overall["F0.5"],
        "EM": overall["EM"],
        "unchanged_rate": overall["unchanged_rate"],
        "type0_overcorrection": type0_overcorrection,
        "classification_applicable": requires_type_prediction,
        "by_error_type": by_type,
        "ellipsis_breakdown": ellipsis_breakdown,
        "classification": classification,
        "usage": usage,
        "latency_seconds": latency,
        "scorer_definition": {
            "unicode_normalization": "NFC before tokenization, EM, and unchanged checks",
            "tokenizer": "NFC Unicode word runs; punctuation separate; literal ... is one token",
            "alignment": "difflib.SequenceMatcher(autojunk=False) token opcodes",
            "edit_match": "exact (source start, source end, replacement token tuple)",
            "multi_reference": (
                "maximize sentence F0.5, then TP, then minimize FP+FN, then reference order"
            ),
            "zero_denominator": (
                "P or R is null when its denominator is zero; F0.5 is null when no gold or "
                "predicted edits exist, otherwise 0 when TP=0"
            ),
            "limitations": (
                "Internal deterministic scorer, not M2 and not an official VLSP scorer. "
                "SequenceMatcher edit boundaries can differ from linguistically equivalent alignments."
            ),
        },
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    write_csv(
        run_dir / "errors.csv",
        error_rows,
        [
            "id",
            "status",
            "error_type",
            "has_ellipsis",
            "input",
            "prediction",
            "predicted_type_id",
            "references",
            "chosen_reference",
            "tp",
            "fp",
            "fn",
            "exact_match",
            "unchanged",
        ],
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score one VGEC run")
    parser.add_argument("--run-id", help="Directory name below --runs-dir")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.run_id) == bool(args.run_dir):
        print("ERROR: provide exactly one of --run-id or --run-dir")
        return 2
    run_dir = args.run_dir or args.runs_dir / args.run_id
    try:
        metrics = score_run(run_dir, args.data, args.allow_partial)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
