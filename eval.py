"""Evaluation CLI: scores the classifier on a labelled set and writes a report.

M2 reports type accuracy for the AI-only path. Rules and the remaining metrics arrive in M3.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from wattle import data
from wattle.classifier import classify
from wattle.config import LLM_MODEL, RESULTS_DIR, TIMEZONE
from wattle.llm import CachedLLM, GeminiClient, LLMClient, LLMError
from wattle.metrics import Prediction, expected_final_type, per_type_accuracy, type_accuracy
from wattle.models import LabelledRequest
from wattle.rules import ai_only_decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define the CLI in one place so the README can quote it exactly."""
    parser = argparse.ArgumentParser(description="Evaluate the Wattle request classifier.")
    parser.add_argument("--set", dest="dataset", choices=["dev", "test"], required=True)
    parser.add_argument("--no-rules", action="store_true", help="AI alone (ablation baseline)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N requests")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    return args


def load_rows(dataset: str, limit: int | None) -> list[LabelledRequest]:
    """Pick the labelled file; the test set is only ever read here, for scoring."""
    rows = data.load_dev() if dataset == "dev" else data.load_test()
    return rows[:limit] if limit is not None else rows


def predict(row: LabelledRequest, llm: LLMClient) -> Prediction:
    """Classify one request without its labels, then pair the result with the label.

    A failed LLM call is raised, not scored: a run with failures is not a measurement.
    """
    classification = classify(row.to_request(), llm)
    if classification.llm_error:
        raise LLMError(f"{row.id}: {classification.rationale}")
    decision = ai_only_decision(classification)
    return Prediction(
        id=row.id,
        expected=expected_final_type(row.expected_type),
        predicted=decision.final_type,
        ai_type=classification.request_type,
        confidence=classification.confidence,
    )


def run(rows: list[LabelledRequest], llm: LLMClient) -> list[Prediction]:
    """Score every row, printing progress because uncached runs are throttled and slow."""
    predictions: list[Prediction] = []
    for index, row in enumerate(rows, start=1):
        prediction = predict(row, llm)
        predictions.append(prediction)
        mark = "ok  " if prediction.correct else "MISS"
        print(f"[{index:>2}/{len(rows)}] {mark} {row.id}", flush=True)
    return predictions


def summarise(predictions: list[Prediction], dataset: str, mode: str) -> dict:
    """Collect the numbers once so the console, Markdown and JSON outputs all agree."""
    return {
        "dataset": dataset,
        "mode": mode,
        "model": LLM_MODEL,
        "n": len(predictions),
        "type_accuracy": round(type_accuracy(predictions), 4),
        "correct": sum(p.correct for p in predictions),
        "per_type": {
            final_type.value: {"correct": correct, "total": total}
            for final_type, (correct, total) in per_type_accuracy(predictions).items()
        },
        "misses": [
            {"id": p.id, "expected": p.expected.value, "predicted": p.predicted.value}
            for p in predictions
            if not p.correct
        ],
    }


def to_markdown(summary: dict) -> str:
    """Render the report as Markdown so it can be pasted into the README and slides."""
    lines = [
        f"# Eval: {summary['dataset']} set, {summary['mode']}",
        "",
        f"- Model: `{summary['model']}`",
        f"- Requests: {summary['n']}",
        f"- Type accuracy: {summary['correct']}/{summary['n']} ({summary['type_accuracy']:.1%})",
        "",
        "| Expected type | Correct | Total |",
        "|---|---|---|",
    ]
    for name, counts in summary["per_type"].items():
        lines.append(f"| {name} | {counts['correct']} | {counts['total']} |")
    lines += ["", "| Missed ID | Expected | Predicted |", "|---|---|---|"]
    for miss in summary["misses"]:
        lines.append(f"| {miss['id']} | {miss['expected']} | {miss['predicted']} |")
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict, predictions: list[Prediction]) -> Path:
    """Save the report (committed) and the per-item predictions (gitignored)."""
    stamp = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y%m%d-%H%M%S")
    name = f"{summary['dataset']}_{summary['mode']}_{stamp}"
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{name}.md").write_text(to_markdown(summary), encoding="utf-8")
    raw_lines = [p.model_dump_json() for p in predictions]
    (raw_dir / f"{name}.jsonl").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return RESULTS_DIR / f"{name}.md"


def main(argv: list[str] | None = None) -> int:
    """Run one evaluation and return a process exit code."""
    args = parse_args(argv)
    if not args.no_rules:
        print("Rules are added in milestone M3. For now run with --no-rules.")
        return 2
    try:
        llm = CachedLLM(GeminiClient(), model=LLM_MODEL)
        predictions = run(load_rows(args.dataset, args.limit), llm)
    except LLMError as exc:
        print(f"Stopped, no report written: {exc}")
        print("Answers received so far are cached, so re-running continues from here.")
        return 1
    summary = summarise(predictions, args.dataset, "no-rules")
    report_path = write_outputs(summary, predictions)
    print()
    print(to_markdown(summary))
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
