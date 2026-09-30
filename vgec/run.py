from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .io_utils import (
    append_jsonl_durable,
    atomic_write_json,
    git_commit_or_none,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .providers import MockProvider, ProviderError, build_api_provider


PROXY_MODELS = {"azure-4.1", "azure-4o"}
MODEL_CHOICES = (
    "gpt-4.1",
    "gpt-4o",
    "azure-4.1",
    "azure-4o",
    "qwen",
    "bartpho-pretrained",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(
            "run_id must be 1-128 characters containing only letters, digits, _, ., or -"
        )
    return value


def _default_run_id(model: str, dry_run: bool) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()
    prefix = "dryrun" if dry_run else "run"
    return f"{prefix}-{timestamp}-{model_slug}"


def _select_records(
    records: list[dict[str, Any]], limit: int | None, sample_by_type: bool
) -> list[dict[str, Any]]:
    if limit is None:
        return records
    if not sample_by_type:
        return records[:limit]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for error_type in range(5):
        match = next((row for row in records if row["error_type"] == error_type), None)
        if match is not None and len(selected) < limit:
            selected.append(match)
            selected_ids.add(match["id"])
    for row in records:
        if len(selected) >= limit:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    return selected


def _successful_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    successful: set[int] = set()
    for row in read_jsonl(predictions_path):
        output_mode = row.get("output_mode")
        structured = output_mode == "type_and_correction_json" or (
            output_mode is None
            and row.get("selected_model") in PROXY_MODELS
            and row.get("method") != "zero_shot_correction_only"
        )
        valid_type = (
            isinstance(row.get("predicted_type_id"), int)
            and not isinstance(row.get("predicted_type_id"), bool)
            and row["predicted_type_id"] in range(5)
        )
        if (
            row.get("status") in {"ok", "mock"}
            and isinstance(row.get("prediction"), str)
            and (not structured or valid_type)
        ):
            successful.add(row["id"])
    return successful


def _base_config(
    *,
    run_id: str,
    model: str,
    dry_run: bool,
    data_path: Path,
    data_hash: str,
    manifest_path: Path,
    prompt_path: Path,
    prompt_hash: str | None,
    prompt_version: str,
    prompt_definition: dict[str, str] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_mode = (
        "correction_only_text"
        if args.experiment == "correction-only"
        else "type_and_correction_json"
        if model in PROXY_MODELS
        else "correction_text"
    )
    if dry_run:
        provider = "mock"
        method = "dry_run_mock"
    elif model == "bartpho-pretrained":
        provider = "local"
        method = "pretrained_only_no_finetune"
    elif model == "qwen":
        provider = "qwen_openai_compatible"
        method = "zero_shot_prompt"
    elif model in PROXY_MODELS and args.experiment == "correction-only":
        provider = "litellm_openai_compatible"
        method = "zero_shot_correction_only"
    elif model in PROXY_MODELS:
        provider = "litellm_openai_compatible"
        method = "zero_shot_type_and_correction"
    else:
        provider = "openai"
        method = "zero_shot_prompt"
    generation: dict[str, Any] = {
        "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
    }
    if model in PROXY_MODELS:
        generation.update(
            {
                "temperature": 0,
                "endpoint_protocol": "openai_compatible_chat_completions",
            }
        )
        if output_mode == "type_and_correction_json":
            generation["response_format"] = {"type": "json_object"}
    config = {
        "schema_version": 1,
        "pipeline_version": __version__,
        "run_id": run_id,
        "selected_model": model,
        "actual_model_ids": [],
        "provider": provider,
        "method": method,
        "experiment": args.experiment,
        "output_mode": output_mode,
        "dry_run": dry_run,
        "prompt_version": prompt_version,
        "prompt_path": None if model == "bartpho-pretrained" else str(prompt_path),
        "prompt_sha256": prompt_hash,
        "data_path": str(data_path),
        "data_hash": data_hash,
        "manifest_path": str(manifest_path),
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "source_commit": git_commit_or_none(),
        "generation": generation,
        "selection": {
            "limit": args.limit,
            "all": args.all,
            "sample_by_type": args.sample_by_type,
            "max_requests": args.max_requests,
        },
    }
    if prompt_definition is not None:
        config["prompt_definition"] = prompt_definition
    if args.experiment == "correction-only":
        source_prompt = Path("prompts/v2_type_and_correction.json")
        config["prompt_provenance"] = {
            "derived_from": str(source_prompt),
            "derived_from_sha256": sha256_file(source_prompt),
            "diff_path": "prompts/v2_to_v3_correction_only.diff",
        }
    return config


def _load_or_create_config(path: Path, proposed: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        atomic_write_json(path, proposed)
        return proposed
    existing = read_json(path)
    # Backward-compatible inference for runs created before these explicit
    # protocol locks were introduced. This does not rewrite historical configs.
    existing_for_compare = dict(existing)
    existing_method = existing.get("method")
    existing_for_compare.setdefault(
        "experiment",
        "correction-only"
        if existing_method == "zero_shot_correction_only"
        else "type-and-correction",
    )
    existing_for_compare.setdefault(
        "output_mode",
        "correction_only_text"
        if existing_method == "zero_shot_correction_only"
        else "type_and_correction_json"
        if existing.get("selected_model") in PROXY_MODELS
        else "correction_text",
    )
    required_matches = (
        "run_id",
        "selected_model",
        "provider",
        "method",
        "experiment",
        "output_mode",
        "dry_run",
        "prompt_version",
        "prompt_sha256",
        "data_hash",
        "generation",
    )
    mismatches = [
        key
        for key in required_matches
        if existing_for_compare.get(key) != proposed.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume {path.parent.name}; config differs for: {', '.join(mismatches)}"
        )
    return existing


def _build_provider(
    args: argparse.Namespace,
    prompt: str | None,
    user_prompt_template: str | None,
):
    if args.dry_run:
        output_mode = (
            "correction_only_text"
            if args.experiment == "correction-only"
            else "type_and_correction_json"
            if args.model in PROXY_MODELS
            else "correction_text"
        )
        return MockProvider(args.model, prompt, output_mode)
    if args.model == "bartpho-pretrained":
        from .bartpho_infer import BartphoPretrainedProvider

        return BartphoPretrainedProvider(
            device=args.device,
            max_new_tokens=args.max_output_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            local_files_only=args.local_files_only,
        )
    assert prompt is not None
    return build_api_provider(
        args.model,
        prompt,
        args.max_output_tokens,
        args.timeout,
        user_prompt_template,
        (
            "correction_only_text"
            if args.experiment == "correction-only"
            else "type_and_correction_json"
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.all and not args.confirm_full_run:
        raise ValueError("--all requires --confirm-full-run")
    if args.experiment == "correction-only" and args.model not in PROXY_MODELS:
        raise ValueError("--experiment correction-only is supported only for azure-* models")

    manifest = read_json(args.manifest)
    if not manifest.get("evaluation_locked"):
        raise ValueError("Preprocessing manifest is not locked")
    if not manifest.get("unresolved_reviews_excluded"):
        raise ValueError("Manifest does not confirm exclusion of unresolved reviews")
    records = read_jsonl(args.data)
    data_hash = sha256_file(args.data)
    expected = manifest.get("evaluation", {})
    if expected.get("sha256") != data_hash:
        raise ValueError("eval_clean.jsonl hash does not match the preprocessing manifest")
    if expected.get("ids") != [row.get("id") for row in records]:
        raise ValueError("Evaluation IDs/order do not match the preprocessing manifest")

    prompt: str | None = None
    user_prompt_template: str | None = None
    prompt_hash: str | None = None
    prompt_definition: dict[str, str] | None = None
    prompt_version = "none"
    prompt_path = args.prompt
    if args.model != "bartpho-pretrained":
        if prompt_path is None:
            prompt_path = Path(
                "prompts/v3_correction_only.json"
                if args.model in PROXY_MODELS and args.experiment == "correction-only"
                else "prompts/v2_type_and_correction.json"
                if args.model in PROXY_MODELS
                else "prompts/v1.txt"
            )
        prompt_raw = prompt_path.read_text(encoding="utf-8")
        prompt_hash = sha256_text(prompt_raw)
        if args.model in PROXY_MODELS:
            definition = json.loads(prompt_raw)
            required_prompt_keys = {
                "prompt_version",
                "system_prompt",
                "user_prompt_template",
            }
            if not isinstance(definition, dict) or set(definition) != required_prompt_keys:
                raise ValueError(
                    "Proxy prompt JSON must contain exactly prompt_version, "
                    "system_prompt, and user_prompt_template"
                )
            if not all(isinstance(definition[key], str) for key in required_prompt_keys):
                raise ValueError("Proxy prompt fields must all be strings")
            if definition["user_prompt_template"].count("{sentence}") != 1:
                raise ValueError("Proxy user prompt must contain {sentence} exactly once")
            prompt = definition["system_prompt"]
            user_prompt_template = definition["user_prompt_template"]
            prompt_version = definition["prompt_version"]
            prompt_definition = {
                "system_prompt": prompt,
                "user_prompt_template": user_prompt_template,
            }
        else:
            prompt = prompt_raw
            prompt_version = "v1"
    if prompt_path is None:
        prompt_path = Path("prompts/v1.txt")

    selected = _select_records(records, args.limit, args.sample_by_type)
    run_id = _safe_run_id(args.run_id or _default_run_id(args.model, args.dry_run))
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    proposed = _base_config(
        run_id=run_id,
        model=args.model,
        dry_run=args.dry_run,
        data_path=args.data,
        data_hash=data_hash,
        manifest_path=args.manifest,
        prompt_path=prompt_path,
        prompt_hash=prompt_hash,
        prompt_version=prompt_version,
        prompt_definition=prompt_definition,
        args=args,
    )
    config = _load_or_create_config(config_path, proposed)
    completed = _successful_ids(predictions_path)
    pending_all = [row for row in selected if row["id"] not in completed]
    pending = pending_all
    if args.max_requests is not None:
        pending = pending[: args.max_requests]

    provider = _build_provider(args, prompt, user_prompt_template)
    configuration = getattr(provider, "configuration", None)
    if callable(configuration):
        configuration_value = configuration()
        if args.model == "bartpho-pretrained":
            config["bartpho"] = configuration_value
        else:
            config["provider_configuration"] = configuration_value
        atomic_write_json(config_path, config)

    attempted = 0
    succeeded = 0
    failed = 0
    stop_reason: str | None = None
    actual_model_ids = set(config.get("actual_model_ids", []))
    for record in pending:
        attempted += 1
        for attempt in range(args.max_retries + 1):
            started = time.perf_counter()
            try:
                result = provider.generate(record["input"])
                latency = time.perf_counter() - started
                prediction = result.raw_text.strip()
                if not prediction:
                    raise ProviderError(
                        "Model returned an empty output",
                        retryable=False,
                        category="invalid_output",
                    )
                row = {
                    "id": record["id"],
                    "input": record["input"],
                    "prediction": prediction,
                    "raw_prediction": result.raw_output or result.raw_text,
                    "predicted_type_id": result.predicted_type_id,
                    "status": "mock" if args.dry_run else "ok",
                    "latency_seconds": latency,
                    "request_id": result.request_id,
                    "usage": result.usage,
                    "selected_model": args.model,
                    "actual_model_id": result.model_id,
                    "provider": provider.provider_name,
                    "method": provider.method,
                    "output_mode": config["output_mode"],
                    "prompt_version": prompt_version,
                    "prompt_sha256": prompt_hash,
                    "data_hash": data_hash,
                    "http_status": result.http_status,
                    "attempt": attempt + 1,
                    "created_at_utc": utc_now(),
                    "raw_response": result.raw_response,
                }
                append_jsonl_durable(predictions_path, row)
                if result.model_id:
                    actual_model_ids.add(result.model_id)
                succeeded += 1
                break
            except ProviderError as exc:
                latency = time.perf_counter() - started
                append_jsonl_durable(
                    predictions_path,
                    {
                        "id": record["id"],
                        "input": record["input"],
                        "prediction": None,
                        "raw_prediction": exc.raw_output,
                        "predicted_type_id": None,
                        "status": "error",
                        "latency_seconds": latency,
                        "request_id": None,
                        "usage": {},
                        "selected_model": args.model,
                        "actual_model_id": None,
                        "provider": provider.provider_name,
                        "method": provider.method,
                        "output_mode": config["output_mode"],
                        "prompt_version": prompt_version,
                        "prompt_sha256": prompt_hash,
                        "data_hash": data_hash,
                        "http_status": exc.status_code,
                        "attempt": attempt + 1,
                        "created_at_utc": utc_now(),
                        "error": {
                            "category": exc.category,
                            "message": str(exc),
                            "status_code": exc.status_code,
                            "retryable": exc.retryable,
                        },
                        "raw_response": exc.raw_response,
                    },
                )
                if exc.retryable and attempt < args.max_retries:
                    delay = args.retry_base_seconds * (2**attempt)
                    delay *= random.uniform(0.8, 1.2)
                    time.sleep(delay)
                    continue
                failed += 1
                if exc.category in {"authentication", "quota"}:
                    stop_reason = exc.category
                break
        config["actual_model_ids"] = sorted(actual_model_ids)
        atomic_write_json(config_path, config)
        if stop_reason:
            break

    config["actual_model_ids"] = sorted(actual_model_ids)
    config["last_updated_at_utc"] = utc_now()
    config["completed_at_utc"] = utc_now() if not stop_reason else None
    config["last_invocation"] = {
        "selected": len(selected),
        "already_complete": len(selected) - len(pending_all),
        "deferred_by_max_requests": len(pending_all) - len(pending),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "stop_reason": stop_reason,
    }
    atomic_write_json(config_path, config)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "data_hash": data_hash,
        "selected": len(selected),
        "already_complete": len(selected) - len(pending_all),
        "deferred_by_max_requests": len(pending_all) - len(pending),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "stop_reason": stop_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VGEC inference with safe resume")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument(
        "--experiment",
        choices=("type-and-correction", "correction-only"),
        default="type-and-correction",
        help="Azure output protocol; existing behavior remains the default",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Use a local mock only")
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required with --all to avoid accidental cost/long local inference",
    )
    parser.add_argument("--sample-by-type", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--data", type=Path, default=Path("data/processed/eval_clean.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/reports/manifest.json"))
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Prompt file; defaults to v2 JSON for azure-* and v1 text otherwise",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["stop_reason"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
