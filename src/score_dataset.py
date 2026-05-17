"""
Подготовка данных для обучения оценки техники: видео → поза → ресемплинг + табличные признаки.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src.feature_extraction import extract_forward_walkover_features_from_sequences
from src.pose_detection import extract_pose_from_video
from src.pose_schema import LAYOUTS
from src.score_classes import NUM_QUALITY_CLASSES, validate_class_ids
from src.video_processor import normalize_poses, smooth_pose_sequence


def load_labels_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"filename", "class_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок {missing}. Нужны как минимум: filename, class_id")
    df = df.copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["class_id"] = df["class_id"].astype(int)
    validate_class_ids(df["class_id"].tolist())
    return df


def _resample_xy_sequence(
    seq_flat: np.ndarray,
    layout: str,
    target_len: int,
) -> np.ndarray:
    """
    seq_flat: (T_orig, num_kp * feat_per_kp), значения после normalize_poses.
    Возвращает (target_len, num_kp * 2) только x,y.
    """
    lay = LAYOUTS[layout.lower()]
    t_orig = len(seq_flat)
    m = seq_flat.reshape(t_orig, lay.num_keypoints, lay.feat_per_kp).astype(np.float32)
    xy = m[:, :, :2].copy()
    xy[~np.isfinite(xy)] = 0.0

    if t_orig <= 0:
        raise ValueError("Пустая последовательность поз")

    if t_orig == 1:
        out = np.repeat(xy, target_len, axis=0)
        return out.reshape(target_len, -1).astype(np.float32)

    t_src = np.arange(t_orig, dtype=np.float32)
    t_dst = np.linspace(0.0, float(t_orig - 1), target_len, dtype=np.float32)
    out = np.zeros((target_len, lay.num_keypoints, 2), dtype=np.float32)
    for k in range(lay.num_keypoints):
        for d in range(2):
            out[:, k, d] = np.interp(t_dst, t_src, xy[:, k, d])
    return out.reshape(target_len, -1).astype(np.float32)


FORWARD_WALKOVER_FEATURE_KEYS: tuple[str, ...] = (
    "num_frames",
    "inverted_frame_idx",
    "inverted_frame_ratio",
    "bridge_frame_idx",
    "bridge_frame_ratio",
    "landing_start_idx",
    "handstand_alignment_min",
    "handstand_alignment_p10",
    "elbow_angle_min_deg",
    "elbow_angle_p10_deg",
    "knee_angle_min_deg",
    "knee_angle_p10_deg",
    "shoulder_opening_min_deg",
    "shoulder_opening_p10_deg",
    "bridge_hip_angle_min_deg",
    "bridge_span_min",
    "bridge_drop_from_inverted",
    "landing_torso_tilt_mean",
    "landing_torso_tilt_max",
    "landing_step_mean",
    "landing_step_max",
    "landing_foot_separation_mean",
    "landing_foot_separation_max",
    "landing_hip_sway_mean",
    "landing_hip_sway_max",
    "landing_shoulder_height_over_hip_mean",
    "landing_head_height_over_hip_mean",
    "confidence_mean",
)


def _features_dict_to_vector(feat: dict[str, Any]) -> np.ndarray:
    vec = []
    for k in FORWARD_WALKOVER_FEATURE_KEYS:
        v = float(feat[k])
        if not np.isfinite(v):
            v = 0.0
        vec.append(v)
    return np.asarray(vec, dtype=np.float32)


def build_sample_from_video(
    video_path: str,
    *,
    layout: str = "coco17",
    backend: str = "yolo",
    yolo_model: str | None = None,
    openpose_bin: str | None = None,
    smooth_window: int = 5,
    seq_len: int = 96,
    use_smoothing: bool = True,
) -> dict[str, Any]:
    """
    Подготовить один новый ролик для инференса: видео → поза → признаки.

    Возвращает X_seq формы (1, seq_len, D) и X_tab формы (1, F).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Нет файла: {video_path}")

    landmarks = extract_pose_from_video(
        video_path,
        show_video=False,
        fill_missing=True,
        backend=backend,  # type: ignore[arg-type]
        openpose_bin=openpose_bin,
        yolo_model=yolo_model,
    )
    if landmarks is None or len(landmarks) == 0:
        raise RuntimeError(f"Не удалось извлечь позу: {video_path}")

    processed = landmarks.astype(np.float32, copy=False)
    if use_smoothing:
        sm = smooth_pose_sequence(
            processed,
            layout=layout,
            window_size=smooth_window,
        )
        if sm is None:
            raise RuntimeError(f"Сглаживание не удалось: {video_path}")
        processed = sm

    normalized = normalize_poses(processed, layout=layout)
    if normalized is None:
        raise RuntimeError(f"Нормализация не удалась: {video_path}")

    seq = _resample_xy_sequence(normalized, layout, seq_len)
    walkover_feat = extract_forward_walkover_features_from_sequences(
        processed.astype(np.float32),
        normalized.astype(np.float32),
        layout=layout,
    )
    X_tab = _features_dict_to_vector(walkover_feat)

    return {
        "X_seq": seq[None, :, :].astype(np.float32),
        "X_tab": X_tab[None, :].astype(np.float32),
        "features": walkover_feat,
        "path": video_path,
    }


def build_samples_from_labels(
    labels_csv: str,
    video_dir: str,
    *,
    layout: str = "coco17",
    backend: str = "yolo",
    yolo_model: str | None = None,
    openpose_bin: str | None = None,
    smooth_window: int = 5,
    seq_len: int = 96,
    use_smoothing: bool = True,
) -> dict[str, Any]:
    """
    Для каждой строки labels: извлечь позу, сгладить, нормализовать,
    сформировать X_seq (T,D) и вектор признаков переворота вперед.

    Возвращает словарь с массивами numpy и списком путей.
    """
    df = load_labels_csv(labels_csv)
    X_list: list[np.ndarray] = []
    X_tab_list: list[np.ndarray] = []
    y_list: list[int] = []
    paths: list[str] = []

    for _, row in df.iterrows():
        name = row["filename"]
        cls = int(row["class_id"])
        video_path = os.path.join(video_dir, name)
        sample = build_sample_from_video(
            video_path,
            layout=layout,
            backend=backend,  # type: ignore[arg-type]
            yolo_model=yolo_model,
            openpose_bin=openpose_bin,
            smooth_window=smooth_window,
            seq_len=seq_len,
            use_smoothing=use_smoothing,
        )
        X_list.append(sample["X_seq"][0])
        X_tab_list.append(sample["X_tab"][0])

        y_list.append(cls)
        paths.append(video_path)

    X_seq = np.stack(X_list, axis=0).astype(np.float32)
    X_tab = np.stack(X_tab_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    meta = {
        "layout": layout,
        "seq_len": seq_len,
        "feature_dim_seq": int(X_seq.shape[-1]),
        "feature_dim_tab": int(X_tab.shape[-1]),
        "num_classes": NUM_QUALITY_CLASSES,
        "paths": paths,
        "filenames": df["filename"].tolist(),
    }
    return {"X_seq": X_seq, "X_tab": X_tab, "y": y, "meta": meta}
