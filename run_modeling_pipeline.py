from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from distinguish_tc.modeling.pipeline import run_modeling_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行多缓冲整理与基线训练流程。")
    parser.add_argument("--config", default="data_sources/buffer_sources.json", help="缓冲配置文件")
    parser.add_argument("--processed-dir", default="data_processed", help="整理输出目录")
    parser.add_argument("--runs-dir", default="model_runs", help="训练结果目录")
    parser.add_argument("--overwrite-run-dir", default="", help="若提供则直接覆盖该训练结果目录")
    parser.add_argument("--grouped-eval-seed-count", type=int, default=30, help="重复分组评估次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_modeling_pipeline(
        config_path=(ROOT_DIR / args.config).resolve(),
        processed_dir=(ROOT_DIR / args.processed_dir).resolve(),
        runs_dir=(ROOT_DIR / args.runs_dir).resolve(),
        overwrite_run_dir=(ROOT_DIR / args.overwrite_run_dir).resolve() if args.overwrite_run_dir else None,
        grouped_eval_seed_count=args.grouped_eval_seed_count,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
