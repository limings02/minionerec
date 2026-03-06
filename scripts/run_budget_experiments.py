import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _serialize_fire_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _load_experiment_file(yaml_path: Path) -> List[Dict[str, Any]]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, dict) and "experiments" in data:
        experiments = data.get("experiments", [])
    else:
        experiments = [data]

    normalized = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        if "name" not in exp:
            exp["name"] = yaml_path.stem
        normalized.append(_expand_env(exp))
    return normalized


def _discover_experiments(config_dir: Path) -> List[Dict[str, Any]]:
    yaml_files = sorted(config_dir.glob("*.yaml"))
    experiments: List[Dict[str, Any]] = []
    for yaml_file in yaml_files:
        experiments.extend(_load_experiment_file(yaml_file))
    return experiments


def _build_command(
    exp: Dict[str, Any],
    cli_args: argparse.Namespace,
) -> Tuple[List[str], Path]:
    name = exp["name"]
    entry_script = exp["entry_script"]
    arg_map = dict(exp.get("args", {}))

    output_root = exp.get("output_dir", "output_dir/budget_experiments")
    final_output_dir = Path(output_root) / name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 强制每个实验进入独立目录，防止日志和 ckpt 相互覆盖。
    arg_map["output_dir"] = str(final_output_dir)
    if cli_args.budget_hours is not None:
        arg_map["budget_hours"] = cli_args.budget_hours
    if cli_args.budget_steps is not None:
        arg_map["budget_steps"] = cli_args.budget_steps
    if cli_args.max_train_samples is not None:
        arg_map["max_train_samples"] = cli_args.max_train_samples
    if cli_args.train_subset_ratio is not None:
        arg_map["train_subset_ratio"] = cli_args.train_subset_ratio
    if cli_args.max_eval_samples is not None:
        arg_map["max_eval_samples"] = cli_args.max_eval_samples
    if cli_args.seed is not None:
        arg_map["seed"] = cli_args.seed

    command = [cli_args.python_executable, entry_script]
    for key, value in arg_map.items():
        if value is None:
            continue
        command.append(f"--{key}={_serialize_fire_value(value)}")
    return command, final_output_dir


def _run_single_experiment(
    exp: Dict[str, Any],
    cli_args: argparse.Namespace,
) -> Dict[str, Any]:
    name = exp["name"]
    command, exp_output_dir = _build_command(exp, cli_args)
    train_log_path = exp_output_dir / "train.log"
    started_at = time.time()

    command_display = " ".join(command)
    print(f"\n[RUN] {name}\n{command_display}\n")

    if cli_args.dry_run:
        return {
            "name": name,
            "status": "dry_run",
            "returncode": 0,
            "output_dir": str(exp_output_dir),
            "train_log": str(train_log_path),
            "command": command,
            "duration_sec": 0.0,
        }

    with train_log_path.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(f"[{name}] {line}")
            log_f.write(line)
        process.wait()

    duration_sec = time.time() - started_at
    status = "ok" if process.returncode == 0 else "failed"
    result = {
        "name": name,
        "status": status,
        "returncode": int(process.returncode),
        "output_dir": str(exp_output_dir),
        "train_log": str(train_log_path),
        "command": command,
        "duration_sec": float(duration_sec),
    }
    return result


def _run_all_experiments(experiments: List[Dict[str, Any]], cli_args: argparse.Namespace) -> List[Dict[str, Any]]:
    if cli_args.parallel <= 1:
        return [_run_single_experiment(exp, cli_args) for exp in experiments]

    # 默认单卡串行；只有显式 --parallel>1 才并行。
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cli_args.parallel) as executor:
        future_map = {
            executor.submit(_run_single_experiment, exp, cli_args): exp["name"]
            for exp in experiments
        }
        for future in as_completed(future_map):
            results.append(future.result())
    # 结果按配置顺序排序，便于对照查看。
    order = {exp["name"]: idx for idx, exp in enumerate(experiments)}
    results.sort(key=lambda x: order.get(x["name"], 10**9))
    return results


def _run_summary(output_dirs: List[str], cli_args: argparse.Namespace) -> int:
    summary_cmd = [
        cli_args.python_executable,
        "scripts/summarize_results.py",
        "--result_csv",
        cli_args.summary_csv,
    ]
    if output_dirs:
        summary_cmd.append("--output_dirs")
        summary_cmd.extend(output_dirs)

    print("\n[SUMMARY CMD]")
    print(" ".join(summary_cmd))

    if cli_args.dry_run:
        return 0
    completed = subprocess.run(summary_cmd, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-budget experiments from YAML configs.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory that stores experiment YAML files.")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers. Default is 1 (serial).")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue remaining experiments if one fails.")

    parser.add_argument("--budget_hours", type=float, default=None)
    parser.add_argument("--budget_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--train_subset_ratio", type=float, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary_csv", type=str, default="results/summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir not found: {config_dir}")

    experiments = _discover_experiments(config_dir)
    if not experiments:
        raise RuntimeError(f"No YAML experiments found under: {config_dir}")

    results = _run_all_experiments(experiments, args)
    output_dirs = [r["output_dir"] for r in results]

    failed = [r for r in results if r["status"] == "failed"]
    if failed and not args.continue_on_error:
        print("\n[ERROR] At least one experiment failed:")
        for item in failed:
            print(f"  - {item['name']} (returncode={item['returncode']})")
        # 先尝试汇总已完成结果，再抛错，方便快速定位。
        _run_summary(output_dirs, args)
        raise SystemExit(1)

    summary_rc = _run_summary(output_dirs, args)
    if summary_rc != 0:
        raise SystemExit(summary_rc)

    print("\n[RESULTS]")
    for item in results:
        print(
            f"- {item['name']}: status={item['status']}, "
            f"output_dir={item['output_dir']}, log={item['train_log']}"
        )


if __name__ == "__main__":
    main()
