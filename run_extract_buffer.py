from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from distinguish_tc.parser import scan_images
from distinguish_tc.processor import build_project_state, detect_crops_for_samples, export_results, process_rgb_results
from distinguish_tc.project_store import load_project_state, save_project_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按缓冲批量提取 RGB 结果。")
    parser.add_argument("--source-dir", default="Photo", help="原始图片目录，相对路径默认是 Photo")
    parser.add_argument("--project-dir", required=True, help="输出项目目录，例如 project_data_t8")
    parser.add_argument("--roi-project", default="project_data/project.json", help="用于复用 ROI 的项目文件")
    parser.add_argument("--buffer", action="append", required=True, help="要处理的缓冲名，可重复传入")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = (ROOT_DIR / args.source_dir).resolve()
    project_dir = (ROOT_DIR / args.project_dir).resolve()
    roi_project_path = (ROOT_DIR / args.roi_project).resolve()

    roi_state = load_project_state(roi_project_path)
    if roi_state.roi_profile is None:
        raise ValueError(f"{roi_project_path} 中未找到 ROI 配置")

    buffer_names = set(args.buffer)
    samples = [sample for sample in scan_images(source_dir) if sample.buffer_name in buffer_names]
    if not samples:
        raise ValueError(f"在 {source_dir} 中未找到目标缓冲: {', '.join(sorted(buffer_names))}")

    state = build_project_state(source_dir=source_dir, project_dir=project_dir, samples=samples)
    state.roi_profile = roi_state.roi_profile
    state.crops = detect_crops_for_samples(samples=samples, project_dir=project_dir)
    save_project_state(state, project_dir / "project.json")

    results = process_rgb_results(state)
    csv_path = export_results(results, project_dir)
    save_project_state(state, project_dir / "project.json")

    flagged_count = sum(1 for crop in state.crops if crop.auto_flagged or crop.manual_flagged or crop.review_status == "pending")
    ok_count = sum(1 for result in results if result.status == "ok")
    print(f"buffer={','.join(sorted(buffer_names))}")
    print(f"samples={len(samples)}")
    print(f"ok_results={ok_count}")
    print(f"flagged_crops={flagged_count}")
    print(f"csv_path={csv_path}")


if __name__ == "__main__":
    main()
