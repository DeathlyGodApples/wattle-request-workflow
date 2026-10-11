"""Evaluation CLI: scores "AI alone" or "AI + rules" on a labelled set and writes a report.

python eval.py --set dev|test [--no-rules] [--limit N] [--include-fewshot]
python eval.py --compare results/A.json results/B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from wattle import data
from wattle.config import LLM_MODEL, RESULTS_DIR, TIMEZONE
from wattle.llm import CachedLLM, GeminiClient, LLMClient, LLMError
from wattle.metrics import (
    Prediction,
    build_metrics,
    expected_final_type,
    hardship_days_started,
    has_auto_customer_draft,
    llm_failure,
)
from wattle.models import LabelledRequest
from wattle.pipeline import process
from wattle.prompts import FEW_SHOT_IDS

MODE_LABELS = {"no-rules": "AI alone", "rules": "AI + rules"}
NOT_AVAILABLE = "n/a"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define the CLI in one place so the README can quote it exactly."""
    parser = argparse.ArgumentParser(description="Evaluate the Wattle request workflow.")
    parser.add_argument("--set", dest="dataset", choices=["dev", "test"])
    parser.add_argument("--no-rules", action="store_true", help="AI alone (ablation baseline)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N requests")
    parser.add_argument(
        "--include-fewshot",
        action="store_true",
        help="dev only: also score the six requests used as few-shot examples",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("A_JSON", "B_JSON"), help="print two reports side by side"
    )
    args = parser.parse_args(argv)
    if (args.dataset is None) == (args.compare is None):
        parser.error("give exactly one of --set or --compare")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    return args


def load_rows(
    dataset: str, limit: int | None = None, *, include_fewshot: bool = False
) -> tuple[list[LabelledRequest], int]:
    """Pick the rows to score and say how many few-shot examples were left out.

    The model has already seen the few-shot examples with their answers, so scoring them
    would flatter the dev result. The test set has none, so nothing is dropped there.
    """
    rows = data.load_dev() if dataset == "dev" else data.load_test()
    excluded = 0
    if dataset == "dev" and not include_fewshot:
        kept = [row for row in rows if row.id not in FEW_SHOT_IDS]
        excluded = len(rows) - len(kept)
        rows = kept
    if limit is not None:
        rows = rows[:limit]
    return rows, excluded


def predict(row: LabelledRequest, llm: LLMClient, *, use_rules: bool = True) -> Prediction:
    """Run one request through the whole pipeline without its labels, then pair it with them.

    A failed LLM call is raised, not scored: a run with failures is not a measurement.
    """
    result = process(row.to_request(), llm, use_rules=use_rules)
    failure = llm_failure(result)
    if failure:
        raise LLMError(f"{row.id}: {failure}")
    return Prediction(
        id=row.id,
        expected=expected_final_type(row.expected_type),
        predicted=result.decision.final_type,
        ai_type=result.classification.request_type,
        confidence=result.classification.confidence,
        rules_fired=result.decision.rules_fired,
        keyword_hits=result.keyword_hits,
        requires_human=result.decision.requires_human,
        auto_customer_draft=has_auto_customer_draft(result),
        hardship_days=hardship_days_started(result),
        human_before_reply=row.human_before_reply,
        statutory_21d=row.statutory_21d,
    )


def run(rows: list[LabelledRequest], llm: LLMClient, *, use_rules: bool = True) -> list[Prediction]:
    """Score every row, printing progress because uncached runs are throttled and slow."""
    predictions: list[Prediction] = []
    for index, row in enumerate(rows, start=1):
        prediction = predict(row, llm, use_rules=use_rules)
        predictions.append(prediction)
        mark = "ok  " if prediction.correct else "MISS"
        print(f"[{index:>2}/{len(rows)}] {mark} {row.id}", flush=True)
    return predictions


def summarise(
    predictions: list[Prediction], dataset: str, mode: str, excluded_few_shot: int = 0
) -> dict:
    """Add the run's settings to the metrics, so a report says exactly what it measured."""
    return {
        "dataset": dataset,
        "mode": mode,
        "model": LLM_MODEL,
        "excluded_few_shot": excluded_few_shot,
        **build_metrics(predictions),
    }


def _ratio(counts: dict | None, correct_key: str = "correct", total_key: str = "total") -> str:
    """Show small-sample results as counts, e.g. 12/13, rather than a bare percentage."""
    if not counts or correct_key not in counts or not counts.get(total_key):
        return NOT_AVAILABLE
    return f"{counts[correct_key]}/{counts[total_key]}"


def headline_rows(summary: dict) -> list[tuple[str, str]]:
    """The slide-4 numbers as (label, value); older reports lacking a metric show n/a."""
    critical = summary.get("critical")
    hardship = summary.get("hardship")
    unsafe = summary.get("unsafe_auto_sends")
    review_rate = summary.get("human_review_rate")
    return [
        ("Type accuracy", f"{summary['correct']}/{summary['n']} ({summary['type_accuracy']:.1%})"),
        ("Critical caught", _ratio(critical, "caught")),
        (
            "Critical false alarms",
            str(len(critical["false_alarms"])) if critical else NOT_AVAILABLE,
        ),
        ("Hardship caught", _ratio(hardship, "caught")),
        ("Correct clock, of hardship caught", _ratio(hardship, "clock_correct", "clock_total")),
        ("Unsafe auto-sends (target 0)", str(len(unsafe)) if unsafe is not None else NOT_AVAILABLE),
        (
            "Needs a human (any type)",
            f"{review_rate:.1%}" if review_rate is not None else NOT_AVAILABLE,
        ),
        ("Clean accuracy", _ratio(summary.get("clean"))),
        ("Messy accuracy (REQ-051..058)", _ratio(summary.get("messy"))),
    ]


def _confusion_table(confusion: dict) -> list[str]:
    """Confusion matrix as a Markdown table: expected types down, predicted types across."""
    columns = list(confusion)
    lines = [
        "| expected \\ predicted | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for expected, counts in confusion.items():
        cells = " | ".join(str(counts[column]) for column in columns)
        lines.append(f"| {expected} | {cells} |")
    return lines


def to_markdown(summary: dict) -> str:
    """Render the report as Markdown so it can be pasted into the README and slides."""
    lines = [
        f"# Eval: {summary['dataset']} set, {MODE_LABELS[summary['mode']]}",
        "",
        f"- Model: `{summary['model']}`",
        f"- Requests scored: {summary['n']}",
        f"- Few-shot examples excluded: {summary['excluded_few_shot']}",
        "",
        "| Metric | Result |",
        "|---|---|",
    ]
    lines += [f"| {label} | {value} |" for label, value in headline_rows(summary)]
    lines += ["", "## Accuracy by expected type", "", "| Expected type | Correct |", "|---|---|"]
    lines += [f"| {name} | {_ratio(counts)} |" for name, counts in summary["per_type"].items()]
    lines += ["", "## Confusion matrix", "", *_confusion_table(summary["confusion"])]
    lines += ["", "## Misses", "", "| ID | Expected | Predicted |", "|---|---|---|"]
    lines += [f"| {m['id']} | {m['expected']} | {m['predicted']} |" for m in summary["misses"]]
    lines += [
        "",
        f"Critical false alarms: {', '.join(summary['critical']['false_alarms']) or 'none'}",
    ]
    lines += [f"Unsafe auto-sends: {', '.join(summary['unsafe_auto_sends']) or 'none'}"]
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


def compare(path_a: Path, path_b: Path) -> str:
    """Two saved reports side by side as a Markdown table (the slide-4 comparison)."""
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))
    lines = [
        f"| Metric | {_column_title(a)} | {_column_title(b)} |",
        "|---|---|---|",
    ]
    for (label, value_a), (_, value_b) in zip(headline_rows(a), headline_rows(b), strict=True):
        lines.append(f"| {label} | {value_a} | {value_b} |")
    for field in ("dataset", "model", "n", "excluded_few_shot"):
        if a.get(field) != b.get(field):
            lines.append("")
            lines.append(f"WARNING: reports differ in {field}: {a.get(field)} vs {b.get(field)}")
    return "\n".join(lines) + "\n"


def _column_title(summary: dict) -> str:
    """Name a comparison column by what was measured, e.g. 'AI + rules (test)'."""
    mode = MODE_LABELS.get(summary.get("mode", ""), summary.get("mode", "?"))
    return f"{mode} ({summary.get('dataset', '?')})"


def evaluate(args: argparse.Namespace) -> int:
    """Run one evaluation, write its report, and return a process exit code."""
    mode = "no-rules" if args.no_rules else "rules"
    rows, excluded = load_rows(args.dataset, args.limit, include_fewshot=args.include_fewshot)
    if excluded:
        print(f"Excluded {excluded} few-shot example(s) from scoring (--include-fewshot to keep).")
    try:
        llm = CachedLLM(GeminiClient(), model=LLM_MODEL)
        predictions = run(rows, llm, use_rules=not args.no_rules)
    except LLMError as exc:
        print(f"Stopped, no report written: {exc}")
        print("Answers received so far are cached, so re-running continues from here.")
        return 1
    summary = summarise(predictions, args.dataset, mode, excluded)
    report_path = write_outputs(summary, predictions)
    print()
    print(to_markdown(summary))
    print(f"Report written to {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a comparison or an evaluation run."""
    args = parse_args(argv)
    if args.compare:
        print(compare(Path(args.compare[0]), Path(args.compare[1])))
        return 0
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
