from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, read_json, read_jsonl, write_csv
from .score import normalize_text, prf, score_against_references


LEADERBOARD_FIELDS = [
    "run_id",
    "model",
    "provider",
    "method",
    "output_mode",
    "prompt_version",
    "prompt_sha256",
    "n_total",
    "n_scored",
    "coverage",
    "missing_count",
    "invalid_output_count",
    "tp",
    "fp",
    "fn",
    "P",
    "R",
    "F0.5",
    "EM",
    "unchanged_rate",
    "type0_overcorrection",
    "classification_applicable",
    "classification_n",
    "classification_accuracy",
    "classification_macro_f1",
    "cost_or_tokens",
    "latency",
    "scorer_version",
    "data_hash",
    "setting",
]
COMPARISON_FIELDS = [
    "id",
    "input",
    "references",
    "GPT-4.1",
    "GPT-4o",
    "Qwen",
    "BARTpho-pretrained",
    "Azure-4.1",
    "Azure-4o",
    "error_type",
]
EVALUATION_EXAMPLE_FIELDS = [
    "run_id",
    "model",
    "id",
    "error_type",
    "predicted_type_id",
    "classification_eligible",
    "classification_correct",
    "correction_exact_match",
    "outcome",
    "input",
    "prediction",
    "references",
]


def _latest_successes(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") in {"ok", "mock"} and isinstance(row.get("prediction"), str):
            result[row["id"]] = row
    return result


def _format_usage(usage: dict[str, Any]) -> str:
    return json.dumps(usage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_protocol(
    entries: list[dict[str, Any]],
    allow_partial: bool,
    experimental_comparison: bool = False,
) -> None:
    if not entries:
        return
    data_hashes = {entry["metrics"]["data_hash"] for entry in entries}
    scorers = {entry["metrics"]["scorer_version"] for entry in entries}
    scored_sets = {entry["metrics"]["scored_ids_sha256"] for entry in entries}
    if len(data_hashes) != 1:
        raise ValueError("Refusing to merge runs with different dataset hashes")
    if len(scorers) != 1:
        raise ValueError("Refusing to merge runs with different scorer versions")
    if len(scored_sets) != 1:
        raise ValueError("Refusing to merge runs scored on different ID sets")
    partial = [entry for entry in entries if entry["metrics"].get("is_partial")]
    if partial and (not allow_partial or experimental_comparison):
        raise ValueError("Refusing to report partial runs without --allow-partial")
    llm_prompt_hashes = {
        entry["metrics"].get("prompt_sha256")
        for entry in entries
        if str(entry["metrics"].get("method", "")).startswith("zero_shot_")
    }
    if len(llm_prompt_hashes) > 1 and not experimental_comparison:
        raise ValueError("Refusing to merge zero-shot LLM runs with different prompts")
    for entry in entries:
        metrics = entry["metrics"]
        if (
            metrics.get("model") == "bartpho-pretrained"
            and metrics.get("method") != "dry_run_mock"
        ):
            if metrics.get("provider") != "local" or metrics.get("prompt_version") != "none":
                raise ValueError("BARTpho run has an invalid provider/prompt protocol")
        elif metrics.get("method") == "zero_shot_prompt":
            if metrics.get("prompt_version") != "v1":
                raise ValueError("LLM run does not use prompt_version=v1")
        elif metrics.get("method") == "zero_shot_type_and_correction":
            if metrics.get("prompt_version") != "v2_type_and_correction":
                raise ValueError(
                    "Structured proxy run does not use prompt_version=v2_type_and_correction"
                )
        elif metrics.get("method") == "zero_shot_correction_only":
            if (
                metrics.get("prompt_version") != "v3_correction_only"
                or metrics.get("output_mode") != "correction_only_text"
            ):
                raise ValueError("Correction-only run has an invalid prompt/output protocol")


def _comparison_rows(entries: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    if not entries:
        return []
    labels = {
        "gpt-4.1": "GPT-4.1",
        "gpt-4o": "GPT-4o",
        "qwen": "Qwen",
        "bartpho-pretrained": "BARTpho-pretrained",
        "azure-4.1": "Azure-4.1",
        "azure-4o": "Azure-4o",
    }
    by_model: dict[str, dict[int, dict[str, Any]]] = {}
    selected_entry: dict[str, dict[str, Any]] = {}
    for entry in sorted(
        entries, key=lambda item: item["config"].get("started_at_utc", "")
    ):
        model = entry["metrics"]["model"]
        if model in labels:
            selected_entry[model] = entry
    for model, entry in selected_entry.items():
        by_model[model] = _latest_successes(entry["run_dir"] / "predictions.jsonl")
    if len(by_model) < 2:
        return []

    config = next(iter(selected_entry.values()))["config"]
    data = read_jsonl(Path(config["data_path"]))
    rows: list[dict[str, Any]] = []
    for record in data:
        record_id = record["id"]
        if not all(record_id in predictions for predictions in by_model.values()):
            continue
        predictions = {
            labels[model]: by_model[model][record_id]["prediction"] for model in by_model
        }
        # Prefer informative disagreements; fall back to the first complete rows.
        informative = len(set(predictions.values())) > 1
        rows.append(
            {
                "id": record_id,
                "input": record["input"],
                "references": json.dumps(record["references"], ensure_ascii=False),
                **predictions,
                "error_type": record["error_type"],
                "_informative": informative,
            }
        )
    rows.sort(key=lambda row: (not row["_informative"], row["id"]))
    for row in rows:
        row.pop("_informative", None)
    return rows[:max_examples]


def _evaluation_example_rows(
    entries: list[dict[str, Any]], max_examples_per_run: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    outcome_order = (
        "both_correct",
        "classification_only",
        "correction_only",
        "both_wrong",
        "classification_conflict_excluded",
    )
    for entry in entries:
        config = entry["config"]
        metrics = entry["metrics"]
        data = {row["id"]: row for row in read_jsonl(Path(config["data_path"]))}
        predictions = _latest_successes(entry["run_dir"] / "predictions.jsonl")
        conflict_ids = set(
            metrics.get("classification", {}).get("excluded_conflict_ids", [])
        )
        candidates: list[dict[str, Any]] = []
        for record_id, prediction_row in predictions.items():
            record = data[record_id]
            predicted_type = prediction_row.get("predicted_type_id")
            classification_eligible = (
                isinstance(predicted_type, int)
                and not isinstance(predicted_type, bool)
                and predicted_type in range(5)
                and record_id not in conflict_ids
            )
            classification_correct = (
                classification_eligible and predicted_type == record["error_type"]
            )
            correction_exact = prediction_row["prediction"] in record["references"]
            if record_id in conflict_ids:
                outcome = "classification_conflict_excluded"
            elif classification_correct and correction_exact:
                outcome = "both_correct"
            elif classification_correct:
                outcome = "classification_only"
            elif correction_exact:
                outcome = "correction_only"
            else:
                outcome = "both_wrong"
            candidates.append(
                {
                    "run_id": metrics["run_id"],
                    "model": metrics["model"],
                    "id": record_id,
                    "error_type": record["error_type"],
                    "predicted_type_id": predicted_type,
                    "classification_eligible": classification_eligible,
                    "classification_correct": classification_correct,
                    "correction_exact_match": correction_exact,
                    "outcome": outcome,
                    "input": record["input"],
                    "prediction": prediction_row["prediction"],
                    "references": json.dumps(record["references"], ensure_ascii=False),
                }
            )
        candidates.sort(key=lambda row: row["id"])
        selected: list[dict[str, Any]] = []
        quota = max(1, max_examples_per_run // len(outcome_order))
        for outcome in outcome_order:
            selected.extend(
                row for row in candidates if row["outcome"] == outcome
            )
            selected = selected[: len(selected) - max(0, len(selected) - quota * (outcome_order.index(outcome) + 1))]
        selected_ids = {row["id"] for row in selected}
        for row in candidates:
            if len(selected) >= max_examples_per_run:
                break
            if row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])
        output.extend(selected[:max_examples_per_run])
    return output


ABLATION_COMPARISON_FIELDS = [
    "id",
    "error_type",
    "input",
    "references",
    "azure41_baseline_predicted_type",
    "azure41_baseline_prediction",
    "azure41_baseline_chosen_reference",
    "azure41_baseline_tp",
    "azure41_baseline_fp",
    "azure41_baseline_fn",
    "azure41_baseline_F0.5",
    "azure41_baseline_exact_match",
    "azure41_correction_prediction",
    "azure41_correction_chosen_reference",
    "azure41_correction_tp",
    "azure41_correction_fp",
    "azure41_correction_fn",
    "azure41_correction_F0.5",
    "azure41_correction_exact_match",
    "azure41_exact_transition",
    "azure4o_baseline_predicted_type",
    "azure4o_baseline_prediction",
    "azure4o_baseline_chosen_reference",
    "azure4o_baseline_tp",
    "azure4o_baseline_fp",
    "azure4o_baseline_fn",
    "azure4o_baseline_F0.5",
    "azure4o_baseline_exact_match",
    "azure4o_correction_prediction",
    "azure4o_correction_chosen_reference",
    "azure4o_correction_tp",
    "azure4o_correction_fp",
    "azure4o_correction_fn",
    "azure4o_correction_F0.5",
    "azure4o_correction_exact_match",
    "azure4o_exact_transition",
]


def _output_mode(metrics: dict[str, Any]) -> str:
    if metrics.get("output_mode"):
        return str(metrics["output_mode"])
    return (
        "type_and_correction_json"
        if metrics.get("method") == "zero_shot_type_and_correction"
        else "correction_only_text"
        if metrics.get("method") == "zero_shot_correction_only"
        else "correction_text"
    )


def _score_prediction(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    chosen = score_against_references(
        record["input"], row["prediction"], record["references"]
    )
    exact = normalize_text(row["prediction"]) in {
        normalize_text(reference) for reference in record["references"]
    }
    return {**chosen, "exact_match": exact}


def _transition(before: bool, after: bool) -> str:
    if before == after:
        return "exact_both" if before else "nonexact_both"
    return "improved_to_exact" if after else "worsened_from_exact"


def _ablation_comparison_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (entry["metrics"]["model"], entry["metrics"]["method"]): entry
        for entry in entries
    }
    required = {
        (model, method)
        for model in ("azure-4.1", "azure-4o")
        for method in ("zero_shot_type_and_correction", "zero_shot_correction_only")
    }
    if set(by_key) != required:
        raise ValueError(
            "Correction-only ablation requires exactly baseline and correction-only "
            "runs for azure-4.1 and azure-4o"
        )
    data = read_jsonl(Path(entries[0]["config"]["data_path"]))
    predictions = {
        key: _latest_successes(entry["run_dir"] / "predictions.jsonl")
        for key, entry in by_key.items()
    }
    rows: list[dict[str, Any]] = []
    aliases = {"azure-4.1": "azure41", "azure-4o": "azure4o"}
    for record in data:
        output: dict[str, Any] = {
            "id": record["id"],
            "error_type": record["error_type"],
            "input": record["input"],
            "references": json.dumps(record["references"], ensure_ascii=False),
        }
        for model, alias in aliases.items():
            baseline_row = predictions[(model, "zero_shot_type_and_correction")][record["id"]]
            correction_row = predictions[(model, "zero_shot_correction_only")][record["id"]]
            baseline = _score_prediction(record, baseline_row)
            correction = _score_prediction(record, correction_row)
            output.update(
                {
                    f"{alias}_baseline_predicted_type": baseline_row.get("predicted_type_id"),
                    f"{alias}_baseline_prediction": baseline_row["prediction"],
                    f"{alias}_baseline_chosen_reference": baseline["reference"],
                    f"{alias}_baseline_tp": baseline["tp"],
                    f"{alias}_baseline_fp": baseline["fp"],
                    f"{alias}_baseline_fn": baseline["fn"],
                    f"{alias}_baseline_F0.5": baseline["F0.5"],
                    f"{alias}_baseline_exact_match": baseline["exact_match"],
                    f"{alias}_correction_prediction": correction_row["prediction"],
                    f"{alias}_correction_chosen_reference": correction["reference"],
                    f"{alias}_correction_tp": correction["tp"],
                    f"{alias}_correction_fp": correction["fp"],
                    f"{alias}_correction_fn": correction["fn"],
                    f"{alias}_correction_F0.5": correction["F0.5"],
                    f"{alias}_correction_exact_match": correction["exact_match"],
                    f"{alias}_exact_transition": _transition(
                        baseline["exact_match"], correction["exact_match"]
                    ),
                }
            )
        rows.append(output)
    return rows


def _ablation_summary(
    entries: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key = {
        (entry["metrics"]["model"], entry["metrics"]["method"]): entry["metrics"]
        for entry in entries
    }
    pairs: dict[str, Any] = {}
    for model, alias in (("azure-4.1", "azure41"), ("azure-4o", "azure4o")):
        baseline = by_key[(model, "zero_shot_type_and_correction")]
        correction = by_key[(model, "zero_shot_correction_only")]
        transitions: dict[str, int] = {}
        for row in comparison_rows:
            key = row[f"{alias}_exact_transition"]
            transitions[key] = transitions.get(key, 0) + 1
        pairs[model] = {
            "baseline_run_id": baseline["run_id"],
            "correction_only_run_id": correction["run_id"],
            "delta": {
                key: correction.get(key) - baseline.get(key)
                for key in ("P", "R", "F0.5", "EM", "unchanged_rate", "type0_overcorrection")
                if correction.get(key) is not None and baseline.get(key) is not None
            },
            "exact_match_transitions": transitions,
        }
    examples: dict[str, list[int]] = {
        "correction_only_improves": [],
        "correction_only_worsens": [],
        "different_nonexact_outputs": [],
        "type0_unchanged_behavior_differs": [],
    }
    for row in comparison_rows:
        transitions = [row["azure41_exact_transition"], row["azure4o_exact_transition"]]
        if "improved_to_exact" in transitions:
            examples["correction_only_improves"].append(row["id"])
        if "worsened_from_exact" in transitions:
            examples["correction_only_worsens"].append(row["id"])
        if (
            row["azure41_baseline_prediction"] != row["azure41_correction_prediction"]
            or row["azure4o_baseline_prediction"] != row["azure4o_correction_prediction"]
        ) and all(value == "nonexact_both" for value in transitions):
            examples["different_nonexact_outputs"].append(row["id"])
        if row["error_type"] == 0 and (
            (row["azure41_baseline_prediction"] == row["input"])
            != (row["azure41_correction_prediction"] == row["input"])
            or (row["azure4o_baseline_prediction"] == row["input"])
            != (row["azure4o_correction_prediction"] == row["input"])
        ):
            examples["type0_unchanged_behavior_differs"].append(row["id"])
    return {"pairs": pairs, "illustrative_example_ids": {k: v[:5] for k, v in examples.items()}}


def _write_ablation_markdown(
    path: Path, entries: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    lines = [
        "# Correction-only ablation v1",
        "",
        "This internal evaluation compares the existing structured type-and-correction "
        "prompt with a correction-only plain-text prompt on the same 448 records.",
        "",
        "Important: this is a compound ablation. It removes the classification task and "
        "changes the output format from JSON to plain text, so it does not isolate either "
        "factor by itself.",
        "",
        "## Paired deltas (correction-only minus baseline)",
        "",
    ]
    for model, pair in summary["pairs"].items():
        delta = ", ".join(f"{key}={value:+.6f}" for key, value in pair["delta"].items())
        lines.extend([f"- {model}: {delta}", f"  EM transitions: {json.dumps(pair['exact_match_transitions'], sort_keys=True)}"])
    lines.extend(
        [
            "",
            "## Protocol and limitations",
            "",
            "All four runs use the same dataset hash, scorer version, and scored ID set. "
            "Dry and partial runs are rejected in this comparison mode. Scores use the "
            "project's deterministic local edit scorer, not an official VLSP scorer. "
            "Model outputs may vary across provider deployments despite temperature 0.",
            "",
            "See `comparison_examples.csv` for all 448 paired per-ID results and "
            "`report_summary.json` for exact deltas and selected example IDs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    runs_dir: Path,
    results_dir: Path,
    include_dry_run: bool = False,
    allow_partial: bool = False,
    max_examples: int = 20,
    run_ids: set[str] | None = None,
    comparison_mode: str | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            if run_ids is not None and run_dir.name not in run_ids:
                continue
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.json"
            if not config_path.exists() or not metrics_path.exists():
                continue
            config = read_json(config_path)
            if config.get("dry_run") and not include_dry_run:
                continue
            entries.append(
                {
                    "run_dir": run_dir,
                    "config": config,
                    "metrics": read_json(metrics_path),
                }
            )
    found_ids = {entry["metrics"].get("run_id") for entry in entries}
    if run_ids is not None and found_ids != run_ids:
        missing = sorted(run_ids - found_ids)
        raise ValueError(f"Requested runs are missing metrics or were filtered out: {missing}")
    experimental = comparison_mode == "correction-only-ablation"
    if experimental and any(entry["config"].get("dry_run") for entry in entries):
        raise ValueError("Correction-only ablation refuses dry runs")
    _validate_protocol(entries, allow_partial, experimental_comparison=experimental)
    results_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: list[dict[str, Any]] = []
    for entry in entries:
        metrics = entry["metrics"]
        config = entry["config"]
        setting: dict[str, Any] = {
            "actual_model_ids": metrics.get("actual_model_ids", []),
            "run_dir": str(entry["run_dir"]),
        }
        if metrics["model"] == "bartpho-pretrained":
            setting["bartpho"] = config.get("bartpho", {})
        classification = metrics.get("classification", {})
        classification_applicable = metrics.get(
            "classification_applicable",
            metrics.get("method") == "zero_shot_type_and_correction",
        )
        leaderboard_rows.append(
            {
                "run_id": metrics["run_id"],
                "model": metrics["model"],
                "provider": metrics["provider"],
                "method": metrics["method"],
                "output_mode": _output_mode(metrics),
                "prompt_version": metrics["prompt_version"],
                "prompt_sha256": metrics.get("prompt_sha256"),
                "n_total": metrics["n_total"],
                "n_scored": metrics["n_scored"],
                "coverage": metrics.get("coverage", metrics["n_scored"] / metrics["n_total"]),
                "missing_count": metrics.get("missing_count", 0),
                "invalid_output_count": metrics.get("invalid_output_count", 0),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "P": metrics["P"],
                "R": metrics["R"],
                "F0.5": metrics["F0.5"],
                "EM": metrics["EM"],
                "unchanged_rate": metrics["unchanged_rate"],
                "type0_overcorrection": metrics["type0_overcorrection"],
                "classification_applicable": classification_applicable,
                "classification_n": classification.get("n_eligible")
                if classification_applicable
                else None,
                "classification_accuracy": classification.get("accuracy")
                if classification_applicable
                else None,
                "classification_macro_f1": classification.get("macro_f1")
                if classification_applicable
                else None,
                "cost_or_tokens": _format_usage(metrics.get("usage", {})),
                "latency": metrics.get("latency_seconds"),
                "scorer_version": metrics["scorer_version"],
                "data_hash": metrics["data_hash"],
                "setting": json.dumps(
                    setting, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    write_csv(results_dir / "leaderboard.csv", leaderboard_rows, LEADERBOARD_FIELDS)
    if experimental:
        comparisons = _ablation_comparison_rows(entries)
        write_csv(
            results_dir / "comparison_examples.csv",
            comparisons,
            ABLATION_COMPARISON_FIELDS,
        )
        evaluation_examples: list[dict[str, Any]] = []
        ablation = _ablation_summary(entries, comparisons)
    else:
        comparisons = _comparison_rows(entries, max_examples)
        write_csv(results_dir / "comparison_examples.csv", comparisons, COMPARISON_FIELDS)
        evaluation_examples = _evaluation_example_rows(entries, max_examples)
        write_csv(
            results_dir / "evaluation_examples.csv",
            evaluation_examples,
            EVALUATION_EXAMPLE_FIELDS,
        )
        ablation = None
    summary = {
        "n_runs": len(entries),
        "include_dry_run": include_dry_run,
        "allow_partial": allow_partial,
        "comparison_mode": comparison_mode,
        "protocol": None
        if not entries
        else {
            "data_hash": entries[0]["metrics"]["data_hash"],
            "scorer_version": entries[0]["metrics"]["scorer_version"],
            "scored_ids_sha256": entries[0]["metrics"]["scored_ids_sha256"],
            "n_scored": entries[0]["metrics"]["n_scored"],
        },
        "comparison_examples": len(comparisons),
        "evaluation_examples": len(evaluation_examples),
        "note": (
            "Internal evaluation on cleaned public training data; not a leaderboard score. "
            "BARTpho is pretrained-only denoising inference, not instruction following or GEC fine-tuning."
        ),
    }
    if ablation is not None:
        summary["ablation"] = ablation
    atomic_write_json(results_dir / "report_summary.json", summary)
    if ablation is not None:
        _write_ablation_markdown(results_dir / "report.md", entries, ablation)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a protocol-consistent results table")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--include-dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--comparison-mode",
        choices=("correction-only-ablation",),
        help="Allow a predeclared cross-prompt/method ablation while retaining data/scorer guards",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Include only this run ID; repeat to compare selected runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_report(
            runs_dir=args.runs_dir,
            results_dir=args.results_dir,
            include_dry_run=args.include_dry_run,
            allow_partial=args.allow_partial,
            max_examples=args.max_examples,
            run_ids=set(args.run_ids) if args.run_ids else None,
            comparison_mode=args.comparison_mode,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
