import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import TrainerCallback


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.floating, np.integer))


def resolve_subset_target(
    total_count: int,
    max_samples: int = -1,
    subset_ratio: float = -1.0,
) -> int:
    if total_count <= 0:
        return 0

    target = total_count
    if max_samples is not None and int(max_samples) > 0:
        target = min(target, int(max_samples))
    if subset_ratio is not None and float(subset_ratio) > 0:
        ratio_target = max(1, int(total_count * float(subset_ratio)))
        target = min(target, ratio_target)

    return max(1, target)


def _generate_deterministic_indices(total_count: int, target_count: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    chosen = rng.choice(total_count, size=target_count, replace=False)
    indices = sorted(int(x) for x in chosen.tolist())
    return indices


def _load_cached_indices(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cached_indices(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_or_create_subset_indices(
    *,
    total_count: int,
    target_count: int,
    seed: int,
    subset_index_path: str,
    subset_name: str,
) -> List[int]:
    # 为了确保每次实验可复现，优先复用缓存的索引文件，而不是每次重新抽样。
    cached = _load_cached_indices(subset_index_path)
    if cached is not None:
        cached_total = int(cached.get("total_count", -1))
        cached_target = int(cached.get("target_count", -1))
        cached_seed = int(cached.get("seed", -1))
        cached_indices = cached.get("indices", [])
        if (
            cached_total == total_count
            and cached_target == target_count
            and cached_seed == int(seed)
            and isinstance(cached_indices, list)
            and len(cached_indices) == target_count
        ):
            return [int(x) for x in cached_indices]

    indices = _generate_deterministic_indices(total_count, target_count, seed)
    payload = {
        "subset_name": subset_name,
        "seed": int(seed),
        "total_count": int(total_count),
        "target_count": int(target_count),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "indices": indices,
    }
    _save_cached_indices(subset_index_path, payload)
    return indices


def apply_reproducible_subset(
    dataset: Any,
    *,
    max_samples: int = -1,
    subset_ratio: float = -1.0,
    seed: int = 42,
    subset_index_path: Optional[str] = None,
    subset_name: str = "train",
) -> Tuple[Any, Dict[str, Any]]:
    total_count = len(dataset)
    target_count = resolve_subset_target(
        total_count=total_count,
        max_samples=max_samples,
        subset_ratio=subset_ratio,
    )

    meta = {
        "subset_name": subset_name,
        "seed": int(seed),
        "original_count": int(total_count),
        "selected_count": int(target_count),
        "max_samples": int(max_samples) if max_samples is not None else -1,
        "subset_ratio": float(subset_ratio) if subset_ratio is not None else -1.0,
        "subset_index_path": subset_index_path or "",
    }

    if target_count <= 0 or target_count >= total_count:
        return dataset, meta

    if subset_index_path:
        indices = get_or_create_subset_indices(
            total_count=total_count,
            target_count=target_count,
            seed=seed,
            subset_index_path=subset_index_path,
            subset_name=subset_name,
        )
    else:
        indices = _generate_deterministic_indices(total_count, target_count, seed)

    subset_dataset = dataset.select(indices)
    return subset_dataset, meta


class BudgetStopCallback(TrainerCallback):
    def __init__(self, budget_hours: float = 0.0):
        self.budget_hours = float(budget_hours or 0.0)
        self.budget_seconds = self.budget_hours * 3600.0
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.budget_seconds <= 0 or self.start_time is None:
            return control
        elapsed = time.time() - self.start_time
        # 固定 wall-clock 预算：达到上限后触发 Trainer 的优雅停止，而不是强制中断进程。
        if elapsed >= self.budget_seconds:
            control.should_training_stop = True
        return control


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: str, run_start_time: Optional[float] = None):
        self.metrics_path = metrics_path
        self.run_start_time = run_start_time or time.time()
        ensure_dir(os.path.dirname(metrics_path))

    def _append_row(self, row: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _base_row(self, state, event: str) -> Dict[str, Any]:
        elapsed = max(0.0, time.time() - self.run_start_time)
        return {
            "event": event,
            "step": int(getattr(state, "global_step", 0)),
            "time_sec": round(elapsed, 4),
            "loss": None,
            "reward": None,
            "kl": None,
            "invalid_sid_rate": None,
            "duplicate_rate": None,
            "lr": None,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        logs = logs or {}
        row = self._base_row(state, "log")
        row["loss"] = _to_float_or_none(logs.get("loss"))
        row["reward"] = _to_float_or_none(logs.get("reward"))
        row["kl"] = _to_float_or_none(logs.get("kl"))
        row["invalid_sid_rate"] = _to_float_or_none(logs.get("invalid_sid_rate"))
        row["duplicate_rate"] = _to_float_or_none(logs.get("duplicate_rate"))
        row["lr"] = _to_float_or_none(logs.get("learning_rate", logs.get("lr")))

        # 额外保留数值型字段，便于后续离线汇总（不会影响固定 schema 的核心字段）。
        for key, value in logs.items():
            if key in row:
                continue
            if _is_numeric(value):
                row[key] = float(value)

        self._append_row(row)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        metrics = metrics or {}
        row = self._base_row(state, "eval")
        row["loss"] = _to_float_or_none(metrics.get("eval_loss", metrics.get("loss")))
        row["reward"] = _to_float_or_none(metrics.get("eval_reward", metrics.get("reward")))
        row["kl"] = _to_float_or_none(metrics.get("eval_kl", metrics.get("kl")))
        row["invalid_sid_rate"] = _to_float_or_none(
            metrics.get("eval_invalid_sid_rate", metrics.get("invalid_sid_rate"))
        )
        row["duplicate_rate"] = _to_float_or_none(
            metrics.get("eval_duplicate_rate", metrics.get("duplicate_rate"))
        )
        row["lr"] = _to_float_or_none(metrics.get("learning_rate", metrics.get("lr")))

        row["HR@10"] = _to_float_or_none(metrics.get("eval_HR@10", metrics.get("HR@10")))
        row["NDCG@10"] = _to_float_or_none(metrics.get("eval_NDCG@10", metrics.get("NDCG@10")))

        for key, value in metrics.items():
            if key in row:
                continue
            if _is_numeric(value):
                row[key] = float(value)

        self._append_row(row)
        return control


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_memory_bytes() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _guess_metric_direction(metric_name: str) -> str:
    lower_name = metric_name.lower()
    if "hr@" in lower_name or "ndcg@" in lower_name:
        return "max"
    if "reward" in lower_name and "std" not in lower_name:
        return "max"
    if "loss" in lower_name or "kl" in lower_name or "invalid" in lower_name or "duplicate" in lower_name:
        return "min"
    if "rate" in lower_name:
        return "min"
    return "max"


def build_summary_from_log_history(
    log_history: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    final_metrics: Dict[str, float] = {}
    best_metrics: Dict[str, Dict[str, float]] = {}

    for row in log_history:
        step = int(row.get("step", 0))
        for key, value in row.items():
            if not _is_numeric(value):
                continue
            value_float = float(value)
            final_metrics[key] = value_float

            direction = _guess_metric_direction(key)
            if key not in best_metrics:
                best_metrics[key] = {"value": value_float, "step": step}
            else:
                prev_value = best_metrics[key]["value"]
                better = value_float > prev_value if direction == "max" else value_float < prev_value
                if better:
                    best_metrics[key] = {"value": value_float, "step": step}

    return final_metrics, best_metrics


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
