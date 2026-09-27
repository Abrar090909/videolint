"""Compare focused decisions on user-labeled transcript evidence, without video uploads."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .ai import AIProviderError, get_gemini_provider, get_jev_provider

KINDS = {"interrupted_thought", "repetition", "ending", "goal_alignment"}


def heuristic(kind: str, context: dict) -> str:
    """Cheap, deliberately simple comparison baseline; never used for production findings."""
    if kind == "interrupted_thought":
        before = str(context.get("transcriptBeforeCut", "")).strip()
        after = str(context.get("transcriptAfterCut", "")).strip()
        return "yes" if before and after and before[-1] not in ".!?" and after[0].islower() else "no"
    if kind == "repetition":
        left = set(re.findall(r"\w{4,}", str(context.get("earlierPassage", "")).lower()))
        right = set(re.findall(r"\w{4,}", str(context.get("currentPassage", "")).lower()))
        return "yes" if len(left & right) / max(1, len(left | right)) >= .4 else "no"
    if kind == "ending":
        tail = str(context.get("finalTranscript", "")).strip()
        return "yes" if tail and tail[-1] not in ".!?" and int(context.get("gapAfterSpeechMs", 0)) < 1000 else "no"
    goal = set(re.findall(r"\w{4,}", str(context.get("editingGoal", "")).lower()))
    sections = context.get("selectedTranscriptSections", [])
    text = " ".join(str(section.get("text", "")) for section in sections if isinstance(section, dict))
    relevant = set(re.findall(r"\w{4,}", text.lower()))
    return "yes" if goal and len(goal & relevant) / len(goal) < .15 else "no"


def validate_examples(data: object) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation input must be a nonempty JSON array")
    seen = set()
    for example in data:
        if not isinstance(example, dict) or example.get("kind") not in KINDS or not isinstance(example.get("question"), str) or not example["question"].strip() or not isinstance(example.get("context"), dict) or type(example.get("label")) is not bool:
            raise ValueError("Each example needs kind, nonempty question, context object, and boolean label")
        if not isinstance(example.get("id"), str) or not example["id"] or example["id"] in seen:
            raise ValueError("Each example needs a unique nonempty string id")
        seen.add(example["id"])
    return data


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "success"]
    decided = [row for row in completed if row["verdict"] in ("yes", "no")]
    true_positive = sum(row["verdict"] == "yes" and row["label"] for row in decided)
    false_positive = sum(row["verdict"] == "yes" and not row["label"] for row in decided)
    correct = sum((row["verdict"] == "yes") == row["label"] for row in decided)
    costs = [row["costUsd"] for row in completed if row["costUsd"] is not None]
    return {"total": len(rows), "decided": len(decided), "uncertain": len(completed) - len(decided),
            "failed": len(rows) - len(completed),
            "accuracy": correct / len(completed) if completed else None,
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
            "falsePositives": false_positive,
            "meanLatencyMs": round(sum(row["latencyMs"] for row in completed) / len(completed), 2) if completed else None,
            "reportedCostUsd": round(sum(costs), 8) if costs else None,
            "costCoverage": len(costs)}


def evaluate(examples: list[dict], providers: dict) -> dict:
    validate_examples(examples)
    results = {}
    for name, provider in providers.items():
        if provider is None:
            results[name] = {"status": "skipped", "reason": "API key not configured"}
            continue
        rows = []
        for example in examples:
            start = time.perf_counter()
            try:
                verdict = (heuristic(example["kind"], example["context"]) if name == "heuristic"
                           else provider.judge(example["question"], example["context"]).verdict)
                status, error = "success", None
            except Exception as exc:
                verdict, status, error = None, "failed", type(exc).__name__
            cost = getattr(provider, "last_cost_usd", None) if status == "success" else None
            rows.append({"id": example["id"], "label": example["label"], "verdict": verdict,
                         "status": status, "error": error, "latencyMs": round((time.perf_counter() - start) * 1000, 2),
                         "costUsd": cost})
        results[name] = {"status": "complete", "metrics": summarize(rows), "items": rows}
    return {"schemaVersion": 1, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark labeled editorial judgments")
    parser.add_argument("input", type=Path, help="JSON array of labeled focused judgments")
    parser.add_argument("--output", type=Path, help="Save full JSON results")
    args = parser.parse_args()
    try:
        examples = validate_examples(json.loads(args.input.read_text(encoding="utf-8")))
        try:
            gemini = get_gemini_provider()
        except AIProviderError:
            gemini = None
        try:
            jev = get_jev_provider()
        except AIProviderError:
            jev = None
        result = evaluate(examples, {"jev": jev, "gemini": gemini, "heuristic": object()})
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
            print(f"Evaluation saved to {args.output}")
        else:
            print(output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
