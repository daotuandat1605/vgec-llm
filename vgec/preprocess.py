from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
    write_csv,
)


EXPECTED_DISTRIBUTION = {0: 200, 1: 154, 2: 34, 3: 33, 4: 36}
ANNOTATION_ONLY = re.compile(
    r"^\s*\((?:câu(?: này)?(?: được dùng làm mẫu)?(?: đúng)?|sửa|dùng)\b",
    flags=re.IGNORECASE,
)
INLINE_ALTERNATIVE = re.compile(r"\(hoặc\s*:|\blà/xem\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    action: str
    references: list[str]
    reason: str


def normalize_text(value: str) -> str:
    """NFC-normalize and collapse whitespace only; punctuation is untouched."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _validate_raw(data: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    seen: set[int] = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not an object")
        if set(("ID", "Input", "Output")) - row.keys():
            raise ValueError(f"Record {index} is missing ID/Input/Output")
        record_id = row["ID"]
        if not isinstance(record_id, int) or isinstance(record_id, bool):
            raise ValueError(f"Record {index} has a non-integer ID")
        if record_id in seen:
            raise ValueError(f"Duplicate raw ID: {record_id}")
        seen.add(record_id)
        if not isinstance(row["Input"], str) or not row["Input"].strip():
            raise ValueError(f"ID {record_id} has an empty/non-string Input")
        output = row["Output"]
        if not isinstance(output, dict):
            raise ValueError(f"ID {record_id} has an invalid Output")
        error_type = output.get("Error_Type_Id")
        if error_type not in range(5):
            raise ValueError(f"ID {record_id} has invalid Error_Type_Id={error_type!r}")
        fixed = output.get("Fixed Sentence")
        if fixed is not None and not isinstance(fixed, (str, list)):
            raise ValueError(f"ID {record_id} has invalid Fixed Sentence")
        if isinstance(fixed, list) and not all(isinstance(item, str) for item in fixed):
            raise ValueError(f"ID {record_id} has a non-string reference")
    return data


def _references(row: dict[str, Any], normalized_input: str) -> list[str]:
    if row["Output"]["Error_Type_Id"] == 0:
        return [normalized_input]
    raw = row["Output"].get("Fixed Sentence")
    values = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    result: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _review_reasons(references: list[str]) -> list[str]:
    reasons: list[str] = []
    if not references:
        reasons.append("missing_reference")
    if any(ANNOTATION_ONLY.search(reference) for reference in references):
        reasons.append("annotation_instead_of_complete_sentence")
    if any(INLINE_ALTERNATIVE.search(reference) for reference in references):
        reasons.append("alternatives_embedded_in_reference")
    return reasons


def _parse_decision_references(raw: str, row_number: int) -> list[str]:
    if not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"review_decisions.csv row {row_number}: references must be a JSON string/list"
        ) from exc
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"review_decisions.csv row {row_number}: references must be a JSON string/list"
        )
    result: list[str] = []
    for item in value:
        normalized = normalize_text(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def load_decisions(path: Path) -> dict[int, Decision]:
    if not path.exists():
        return {}
    decisions: dict[int, Decision] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"id", "action", "references", "reason"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"{path} must have columns: id,action,references,reason")
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                record_id = int(row["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: invalid id") from exc
            action = (row["action"] or "").strip().lower()
            if action not in {"replace", "exclude", "keep"}:
                raise ValueError(f"{path}:{row_number}: invalid action={action!r}")
            references = _parse_decision_references(row["references"] or "", row_number)
            if action == "replace" and not references:
                raise ValueError(f"{path}:{row_number}: replace requires references")
            if record_id in decisions:
                raise ValueError(f"{path}:{row_number}: duplicate decision for ID {record_id}")
            decisions[record_id] = Decision(action, references, (row["reason"] or "").strip())
    return decisions


def _audit_source(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return result
    result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if path.suffix.lower() == ".json":
        data = _validate_raw(read_json(path), path)
        result["n_records"] = len(data)
        result["error_type_distribution"] = dict(
            sorted(Counter(row["Output"]["Error_Type_Id"] for row in data).items())
        )
    return result


def _cross_version_audit(v1_path: Path, v2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not v1_path.exists():
        return {"available": False}
    v1_rows = _validate_raw(read_json(v1_path), v1_path)
    v1_by_id = {row["ID"]: row for row in v1_rows}
    v2_by_id = {row["ID"]: row for row in v2_rows}
    common_ids = sorted(set(v1_by_id) & set(v2_by_id))
    input_differences: list[int] = []
    type_differences: list[int] = []
    reference_differences: list[int] = []
    for record_id in common_ids:
        v1 = v1_by_id[record_id]
        v2 = v2_by_id[record_id]
        v1_input = normalize_text(v1["Input"])
        v2_input = normalize_text(v2["Input"])
        if v1_input != v2_input:
            input_differences.append(record_id)
        if v1["Output"]["Error_Type_Id"] != v2["Output"]["Error_Type_Id"]:
            type_differences.append(record_id)
        if set(_references(v1, v1_input)) != set(_references(v2, v2_input)):
            reference_differences.append(record_id)
    return {
        "available": True,
        "common_id_count": len(common_ids),
        "v1_only_ids": sorted(set(v1_by_id) - set(v2_by_id)),
        "v2_only_ids": sorted(set(v2_by_id) - set(v1_by_id)),
        "normalized_input_difference_ids": input_differences,
        "error_type_difference_ids": type_differences,
        "normalized_reference_difference_ids": reference_differences,
        "note": "V1 is audit-only and was not appended to V2.",
    }


def preprocess(
    input_path: Path,
    output_path: Path,
    reports_dir: Path,
    decisions_path: Path,
    audit_v1_json: Path,
    audit_v1_xlsx: Path,
) -> dict[str, Any]:
    raw_data = _validate_raw(read_json(input_path), input_path)
    distribution = dict(
        sorted(Counter(row["Output"]["Error_Type_Id"] for row in raw_data).items())
    )
    decisions = load_decisions(decisions_path)
    raw_ids = {row["ID"] for row in raw_data}
    unknown_decisions = sorted(set(decisions) - raw_ids)
    if unknown_decisions:
        raise ValueError(f"Decisions refer to unknown IDs: {unknown_decisions}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    if not decisions_path.exists():
        write_csv(decisions_path, [], ["id", "action", "references", "reason"])

    audit_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for raw in raw_data:
        record_id = raw["ID"]
        normalized_input = normalize_text(raw["Input"])
        references = _references(raw, normalized_input)
        reasons = _review_reasons(references)
        decision = decisions.get(record_id)
        status = "not_required"
        if reasons:
            if decision is None:
                status = "pending"
            else:
                status = f"resolved_{decision.action}"
            review_rows.append(
                {
                    "id": record_id,
                    "error_type": raw["Output"]["Error_Type_Id"],
                    "input": raw["Input"],
                    "raw_references": json.dumps(
                        raw["Output"].get("Fixed Sentence"), ensure_ascii=False
                    ),
                    "reasons": "|".join(reasons),
                    "status": status,
                }
            )

        audit_rows.append(
            {
                "id": record_id,
                "raw_record": raw,
                "normalized_input": normalized_input,
                "normalized_references": references,
                "review_reasons": reasons,
                "decision": None
                if decision is None
                else {
                    "action": decision.action,
                    "references": decision.references,
                    "reason": decision.reason,
                },
            }
        )

        if reasons and decision is None:
            excluded_rows.append(
                {
                    "id": record_id,
                    "reason": "pending_manual_review",
                    "details": "|".join(reasons),
                    "canonical_id": "",
                }
            )
            continue
        if decision and decision.action == "exclude":
            excluded_rows.append(
                {
                    "id": record_id,
                    "reason": "manual_exclude",
                    "details": decision.reason,
                    "canonical_id": "",
                }
            )
            continue
        if decision and decision.action == "replace":
            references = decision.references
        if not references:
            raise ValueError(f"ID {record_id} has no usable reference")

        candidates.append(
            {
                "id": record_id,
                "input": normalized_input,
                "error_type": raw["Output"]["Error_Type_Id"],
                "references": references,
                "source_version": "VGEC_2026_v2",
                "source_ids": [record_id],
                "has_ellipsis": "..." in normalized_input
                or any("..." in reference for reference in references),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        groups[record["input"]].append(record)

    duplicate_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for records in groups.values():
        records.sort(key=lambda item: item["id"])
        if len(records) == 1:
            clean_rows.append(records[0])
            continue
        canonical = records[0]
        compatible = all(
            item["error_type"] == canonical["error_type"]
            and set(item["references"]) == set(canonical["references"])
            for item in records[1:]
        )
        group_id = "|".join(str(item["id"]) for item in records)
        decision_label = (
            f"merge_compatible_keep_{canonical['id']}" if compatible else "conflict_review"
        )
        for item in records:
            duplicate_rows.append(
                {
                    "group_id": group_id,
                    "id": item["id"],
                    "input": item["input"],
                    "error_type": item["error_type"],
                    "references": json.dumps(item["references"], ensure_ascii=False),
                    "decision": decision_label,
                    "canonical_id": canonical["id"] if compatible else "",
                }
            )
        if not compatible:
            for item in records:
                excluded_rows.append(
                    {
                        "id": item["id"],
                        "reason": "duplicate_conflict_pending_review",
                        "details": group_id,
                        "canonical_id": "",
                    }
                )
            continue
        canonical["source_ids"] = [item["id"] for item in records]
        clean_rows.append(canonical)
        for duplicate in records[1:]:
            excluded_rows.append(
                {
                    "id": duplicate["id"],
                    "reason": "merged_compatible_duplicate",
                    "details": group_id,
                    "canonical_id": canonical["id"],
                }
            )

    clean_rows.sort(key=lambda item: item["id"])
    clean_ids = [row["id"] for row in clean_rows]
    if len(clean_ids) != len(set(clean_ids)):
        raise AssertionError("Clean IDs are not unique")
    if any(not row["input"] or not row["references"] for row in clean_rows):
        raise AssertionError("Clean data contains empty input/references")

    atomic_write_jsonl(output_path, clean_rows)
    eval_hash = sha256_file(output_path)
    atomic_write_jsonl(reports_dir / "raw_audit.jsonl", audit_rows)
    write_csv(
        reports_dir / "review_queue.csv",
        review_rows,
        ["id", "error_type", "input", "raw_references", "reasons", "status"],
    )
    write_csv(
        reports_dir / "duplicate_groups.csv",
        duplicate_rows,
        [
            "group_id",
            "id",
            "input",
            "error_type",
            "references",
            "decision",
            "canonical_id",
        ],
    )
    write_csv(
        reports_dir / "excluded.csv",
        sorted(excluded_rows, key=lambda item: item["id"]),
        ["id", "reason", "details", "canonical_id"],
    )

    pending_ids = sorted(
        row["id"] for row in review_rows if row["status"] == "pending"
    )
    version_audit = _cross_version_audit(audit_v1_json, raw_data)
    report = {
        "pipeline_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(raw_data),
        "source_distribution": distribution,
        "expected_distribution": EXPECTED_DISTRIBUTION,
        "distribution_matches_plan": distribution == EXPECTED_DISTRIBUTION,
        "id_402_457_all_type_1": all(
            row["Output"]["Error_Type_Id"] == 1
            for row in raw_data
            if 402 <= row["ID"] <= 457
        ),
        "multi_reference_source_records": sum(
            isinstance(row["Output"].get("Fixed Sentence"), list)
            and len(row["Output"].get("Fixed Sentence") or []) > 1
            for row in raw_data
        ),
        "duplicate_groups": len({row["group_id"] for row in duplicate_rows}),
        "duplicate_records": len(duplicate_rows),
        "merged_duplicate_rows": sum(
            row["reason"] == "merged_compatible_duplicate" for row in excluded_rows
        ),
        "review_queue_count": len(review_rows),
        "pending_review_count": len(pending_ids),
        "pending_review_ids": pending_ids,
        "ellipsis_source_count": sum(
            "..." in row["Input"]
            or "..." in json.dumps(row["Output"].get("Fixed Sentence"), ensure_ascii=False)
            for row in raw_data
        ),
        "clean_count": len(clean_rows),
        "clean_distribution": dict(
            sorted(Counter(row["error_type"] for row in clean_rows).items())
        ),
        "clean_ellipsis_count": sum(row["has_ellipsis"] for row in clean_rows),
        "excluded_count": len(excluded_rows),
        "unresolved_reviews_excluded": True,
        "evaluation_locked": True,
        "evaluation_scope": (
            "Internal evaluation on the cleaned public training data; not a leaderboard score."
        ),
        "eval_data_sha256": eval_hash,
        "v1_v2_audit": version_audit,
    }
    atomic_write_json(reports_dir / "cleaning_report.json", report)

    manifest = {
        "schema_version": 1,
        "pipeline_version": __version__,
        "created_at_utc": report["created_at_utc"],
        "evaluation_locked": True,
        "unresolved_reviews_excluded": True,
        "pending_review_ids": pending_ids,
        "sources": {
            "v2": _audit_source(input_path),
            "v1_json": _audit_source(audit_v1_json),
            "v1_xlsx": _audit_source(audit_v1_xlsx),
        },
        "v1_v2_audit": version_audit,
        "review_decisions": {
            "path": str(decisions_path),
            "sha256": sha256_file(decisions_path),
            "n_decisions": len(decisions),
        },
        "evaluation": {
            "path": str(output_path),
            "sha256": eval_hash,
            "n_records": len(clean_rows),
            "ids": clean_ids,
        },
    }
    atomic_write_json(reports_dir / "manifest.json", manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the fixed VGEC evaluation set")
    parser.add_argument("--input", type=Path, default=Path("VGEC 2026_v2.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/eval_clean.jsonl")
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/reports/review_decisions.csv"),
    )
    parser.add_argument(
        "--audit-v1-json", type=Path, default=Path("VGEC 2026_v1.json")
    )
    parser.add_argument(
        "--audit-v1-xlsx", type=Path, default=Path("VGEC 2026_v1.xlsx")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = preprocess(
        input_path=args.input,
        output_path=args.output,
        reports_dir=args.reports_dir,
        decisions_path=args.decisions,
        audit_v1_json=args.audit_v1_json,
        audit_v1_xlsx=args.audit_v1_xlsx,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
