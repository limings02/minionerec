import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_value(metrics: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_best_metric(
    best_metrics: Dict[str, Any],
    candidate_keys: List[str],
) -> Optional[float]:
    for key in candidate_keys:
        if key not in best_metrics:
            continue
        node = best_metrics[key]
        if isinstance(node, dict) and "value" in node:
            try:
                return float(node["value"])
            except (TypeError, ValueError):
                continue
        try:
            return float(node)
        except (TypeError, ValueError):
            continue
    return None


def _or_empty(value: Optional[float]) -> Any:
    return "" if value is None else value


def _fallback_from_metrics_jsonl(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    result = {
        "final_loss": None,
        "final_reward": None,
        "final_kl": None,
        "final_invalid_sid_rate": None,
        "final_duplicate_rate": None,
        "final_hr10": None,
        "final_ndcg10": None,
        "best_hr10": None,
        "best_ndcg10": None,
    }
    if not metrics_rows:
        return result

    last_row = metrics_rows[-1]
    result["final_loss"] = _extract_value(last_row, ["loss", "eval_loss"])
    result["final_reward"] = _extract_value(last_row, ["reward", "eval_reward"])
    result["final_kl"] = _extract_value(last_row, ["kl", "eval_kl"])
    result["final_invalid_sid_rate"] = _extract_value(
        last_row, ["invalid_sid_rate", "eval_invalid_sid_rate"]
    )
    result["final_duplicate_rate"] = _extract_value(
        last_row, ["duplicate_rate", "eval_duplicate_rate"]
    )
    result["final_hr10"] = _extract_value(last_row, ["HR@10", "eval_HR@10"])
    result["final_ndcg10"] = _extract_value(last_row, ["NDCG@10", "eval_NDCG@10"])

    hr_values = []
    ndcg_values = []
    for row in metrics_rows:
        hr = _extract_value(row, ["HR@10", "eval_HR@10"])
        ndcg = _extract_value(row, ["NDCG@10", "eval_NDCG@10"])
        if hr is not None:
            hr_values.append(hr)
        if ndcg is not None:
            ndcg_values.append(ndcg)
    result["best_hr10"] = max(hr_values) if hr_values else None
    result["best_ndcg10"] = max(ndcg_values) if ndcg_values else None
    return result


def _build_summary_row(exp_dir: Path) -> Dict[str, Any]:
    summary_path = exp_dir / "summary.json"
    metrics_path = exp_dir / "metrics.jsonl"
    summary = _safe_load_json(summary_path)
    metrics_rows = _safe_load_jsonl(metrics_path)

    row = {
        "name": exp_dir.name,
        "output_dir": str(exp_dir),
        "status": "ok" if summary is not None else "missing_summary",
        "entry_script": "",
        "budget_hours": "",
        "budget_steps": "",
        "runtime_sec": "",
        "global_steps": "",
        "throughput_steps_per_sec": "",
        "peak_memory_mb": "",
        "final_loss": "",
        "final_reward": "",
        "final_kl": "",
        "final_invalid_sid_rate": "",
        "final_duplicate_rate": "",
        "final_hr10": "",
        "final_ndcg10": "",
        "best_hr10": "",
        "best_ndcg10": "",
    }

    if summary is None:
        fallback = _fallback_from_metrics_jsonl(metrics_rows)
        for key, value in fallback.items():
            row[key] = "" if value is None else value
        return row

    final_metrics = summary.get("final_metrics", {}) or {}
    best_metrics = summary.get("best_metrics", {}) or {}

    row.update(
        {
            "entry_script": summary.get("entry_script", ""),
            "budget_hours": summary.get("budget_hours", ""),
            "budget_steps": summary.get("budget_steps", ""),
            "runtime_sec": summary.get("runtime_sec", ""),
            "global_steps": summary.get("global_steps", ""),
            "throughput_steps_per_sec": summary.get("throughput_steps_per_sec", ""),
            "peak_memory_mb": summary.get("peak_memory_mb", ""),
            "final_loss": _or_empty(_extract_value(final_metrics, ["loss", "eval_loss"])),
            "final_reward": _or_empty(_extract_value(final_metrics, ["reward", "eval_reward"])),
            "final_kl": _or_empty(_extract_value(final_metrics, ["kl", "eval_kl"])),
            "final_invalid_sid_rate": _or_empty(_extract_value(
                final_metrics, ["invalid_sid_rate", "eval_invalid_sid_rate"]
            )),
            "final_duplicate_rate": _or_empty(_extract_value(
                final_metrics, ["duplicate_rate", "eval_duplicate_rate"]
            )),
            "final_hr10": _or_empty(_extract_value(final_metrics, ["HR@10", "eval_HR@10"])),
            "final_ndcg10": _or_empty(_extract_value(final_metrics, ["NDCG@10", "eval_NDCG@10"])),
            "best_hr10": _or_empty(_extract_best_metric(best_metrics, ["HR@10", "eval_HR@10"])),
            "best_ndcg10": _or_empty(_extract_best_metric(best_metrics, ["NDCG@10", "eval_NDCG@10"])),
        }
    )
    return row


def _collect_experiment_dirs(args: argparse.Namespace) -> List[Path]:
    exp_dirs: List[Path] = []
    if args.output_dirs:
        exp_dirs.extend(Path(p) for p in args.output_dirs)
    if args.output_root:
        root = Path(args.output_root)
        if root.exists():
            exp_dirs.extend(sorted([p for p in root.iterdir() if p.is_dir()]))

    # 去重并保持顺序
    seen = set()
    unique_dirs = []
    for d in exp_dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(d)
    return unique_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize budget experiment outputs.")
    parser.add_argument("--output_dirs", nargs="*", default=[], help="Explicit experiment output dirs.")
    parser.add_argument("--output_root", type=str, default="", help="Root directory containing experiment dirs.")
    parser.add_argument("--result_csv", type=str, default="results/summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dirs = _collect_experiment_dirs(args)
    if not exp_dirs:
        raise RuntimeError("No experiment output dirs found. Please provide --output_dirs or --output_root.")

    rows = [_build_summary_row(exp_dir) for exp_dir in exp_dirs]
    csv_path = Path(args.result_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "output_dir",
        "status",
        "entry_script",
        "budget_hours",
        "budget_steps",
        "runtime_sec",
        "global_steps",
        "throughput_steps_per_sec",
        "peak_memory_mb",
        "final_loss",
        "final_reward",
        "final_kl",
        "final_invalid_sid_rate",
        "final_duplicate_rate",
        "final_hr10",
        "final_ndcg10",
        "best_hr10",
        "best_ndcg10",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary CSV to {csv_path}")
    for row in rows:
        print(
            f"- {row['name']}: status={row['status']}, "
            f"steps/s={row['throughput_steps_per_sec']}, "
            f"HR@10={row['final_hr10']}, NDCG@10={row['final_ndcg10']}"
        )


if __name__ == "__main__":
    main()
