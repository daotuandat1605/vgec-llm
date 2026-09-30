from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .providers import MockProvider, ProviderError, build_api_provider
from .run import MODEL_CHOICES


def _load_prompt(path: Path) -> tuple[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"prompt_version", "system_prompt", "user_prompt_template"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{path} must contain exactly {sorted(required)}")
    if not all(isinstance(value[key], str) for key in required):
        raise ValueError("Prompt fields must all be strings")
    if value["user_prompt_template"].count("{sentence}") != 1:
        raise ValueError("user_prompt_template must contain {sentence} exactly once")
    return value["system_prompt"], value["user_prompt_template"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify and correct one Vietnamese sentence"
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sentence")
    source.add_argument("--input-file", type=Path)
    parser.add_argument(
        "--prompt", type=Path, default=Path("prompts/v2_type_and_correction.json")
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    sentence = (
        args.sentence
        if args.sentence is not None
        else args.input_file.read_text(encoding="utf-8-sig").strip()
    )
    if not sentence:
        print("ERROR: input sentence is empty")
        return 2
    try:
        system_prompt, user_template = _load_prompt(args.prompt)
        provider = (
            MockProvider(args.model, system_prompt, "type_and_correction_json")
            if args.dry_run
            else build_api_provider(
                args.model,
                system_prompt,
                args.max_output_tokens,
                args.timeout,
                user_template,
                "type_and_correction_json",
            )
        )
        result = provider.generate(sentence)
    except (OSError, ValueError, ProviderError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        json.dumps(
            {
                "type_id": result.predicted_type_id,
                "fixed_sentence": result.raw_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
