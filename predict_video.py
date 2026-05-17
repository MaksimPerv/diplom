#!/usr/bin/env python3
"""
Оценка новых видео переворота вперед обученной моделью.

Новые ролики удобно складывать в data/test_videos, чтобы не смешивать их
с обучающей выборкой data/raw_videos.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.score_classes import QUALITY_CLASS_NAMES_RU
from src.score_dataset import build_sample_from_video


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _collect_videos(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        raise FileNotFoundError(f"Нет файла или папки: {path}")

    videos = sorted(
        str(item)
        for item in p.iterdir()
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(
            f"В папке {path} нет видео. Поддерживаются расширения: "
            f"{', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
    return videos


def _load_meta(model_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(model_dir, "dataset_meta.json")
    if not os.path.isfile(meta_path):
        return {"layout": "coco17", "seq_len": 96}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _probabilities_dict(model: Any, X_tab: np.ndarray) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}
    probs = model.predict_proba(X_tab)[0]
    classes = getattr(model, "classes_", np.arange(len(probs)))
    return {str(int(cls)): float(prob) for cls, prob in zip(classes, probs)}


def predict_one_video(
    video_path: str,
    *,
    model: Any,
    model_dir: str,
    backend: str,
    yolo_model: str,
    openpose_bin: str | None,
    smooth_window: int,
) -> dict[str, Any]:
    meta = _load_meta(model_dir)
    sample = build_sample_from_video(
        video_path,
        layout=str(meta.get("layout", "coco17")),
        backend=backend,
        yolo_model=yolo_model,
        openpose_bin=openpose_bin,
        smooth_window=smooth_window,
        seq_len=int(meta.get("seq_len", 96)),
        use_smoothing=True,
    )

    pred = int(model.predict(sample["X_tab"])[0])
    probs = _probabilities_dict(model, sample["X_tab"])
    confidence = float(max(probs.values())) if probs else None
    return {
        "filename": os.path.basename(video_path),
        "path": video_path,
        "pred_class": pred,
        "pred_name": QUALITY_CLASS_NAMES_RU.get(pred, str(pred)),
        "confidence": confidence,
        "probabilities": probs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Оценить новые видео переворота вперед")
    p.add_argument(
        "--video",
        default="data/test_videos",
        help="Путь к одному видео или папке с новыми видео",
    )
    p.add_argument(
        "--model-dir",
        default="data/processed/front_walkover_prepare_test",
        help="Папка с baseline_mlp_tabular.joblib и dataset_meta.json",
    )
    p.add_argument("--backend", default="yolo", choices=["yolo", "openpose"])
    p.add_argument("--yolo-model", default=os.environ.get("YOLO_POSE_MODEL", "yolov8n-pose.pt"))
    p.add_argument("--openpose-bin", default=os.environ.get("OPENPOSE_BIN"))
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument(
        "--out",
        default="data/processed/front_walkover_prepare_test/new_video_predictions.csv",
        help="CSV-файл для сохранения предсказаний",
    )
    args = p.parse_args()

    model_path = os.path.join(args.model_dir, "baseline_mlp_tabular.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Нет обученной модели: {model_path}")

    model = joblib.load(model_path)
    videos = _collect_videos(args.video)

    rows = []
    for video_path in videos:
        print(f"\nВидео: {video_path}")
        row = predict_one_video(
            video_path,
            model=model,
            model_dir=args.model_dir,
            backend=args.backend,
            yolo_model=args.yolo_model,
            openpose_bin=args.openpose_bin,
            smooth_window=args.smooth_window,
        )
        rows.append(row)
        conf_text = "" if row["confidence"] is None else f", уверенность {row['confidence']:.3f}"
        print(f"Предсказание: {row['pred_class']} — {row['pred_name']}{conf_text}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "probabilities"}
        for cls, prob in row["probabilities"].items():
            flat[f"prob_class_{cls}"] = prob
        out_rows.append(flat)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
