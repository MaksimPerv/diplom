from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from src.pose_detection import extract_pose_from_video
from src.pose_schema import LAYOUTS, PoseLayout
from src.video_processor import normalize_poses, smooth_pose_sequence


COCO = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def _as_layout(layout: str | PoseLayout) -> PoseLayout:
    if isinstance(layout, str):
        return LAYOUTS[layout.lower()]
    return layout


def _valid_point(p: np.ndarray) -> bool:
    return bool(np.isfinite(p).all())


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    if not (_valid_point(a) and _valid_point(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (_valid_point(a) and _valid_point(b) and _valid_point(c)):
        return float("nan")
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return float("nan")
    cos_theta = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _safe_values(values: list[float]) -> list[float]:
    return [v for v in values if np.isfinite(v)]


def _safe_mean(values: list[float]) -> float:
    vals = _safe_values(values)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _safe_min(values: list[float]) -> float:
    vals = _safe_values(values)
    if not vals:
        return float("nan")
    return float(np.min(vals))


def _safe_max(values: list[float]) -> float:
    vals = _safe_values(values)
    if not vals:
        return float("nan")
    return float(np.max(vals))


def _safe_percentile(values: list[float], q: float) -> float:
    vals = _safe_values(values)
    if not vals:
        return float("nan")
    return float(np.percentile(vals, q))


def _nanargmin(values: list[float], fallback: int = 0) -> int:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return fallback
    return int(np.nanargmin(arr))


def _normalized_distance(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    d = _distance(a, b)
    if not np.isfinite(d):
        return float("nan")
    if not np.isfinite(scale) or scale < 1e-6:
        return d
    return float(d / scale)


def _vertical_alignment_score(xs: list[float], scale: float) -> float:
    vals = [x for x in xs if np.isfinite(x)]
    if not vals:
        return float("nan")
    center = float(np.mean(vals))
    score = float(np.mean([abs(x - center) for x in vals]))
    if np.isfinite(scale) and scale > 1e-6:
        score /= float(scale)
    return score


def extract_forward_walkover_features_from_sequences(
    smoothed_sequence: np.ndarray,
    normalized_sequence: np.ndarray,
    layout: str | PoseLayout = "coco17",
) -> dict[str, Any]:
    """
    Признаки для переворота вперед (стойка на руках -> мост -> подъем в стойку).

    Признаки ориентированы на:
    - вертикальность стойки на руках;
    - прямые руки и ноги;
    - раскрытие плеч и амплитуду мостика;
    - устойчивость и контроль выхода;
    - общее качество линий тела и натянутости.
    """
    lay = _as_layout(layout)
    if lay.name != "coco17":
        raise NotImplementedError(
            "Извлечение признаков переворота вперед пока реализовано только для COCO17"
        )

    raw = smoothed_sequence.reshape(len(smoothed_sequence), lay.num_keypoints, lay.feat_per_kp)
    norm = normalized_sequence.reshape(len(normalized_sequence), lay.num_keypoints, lay.feat_per_kp)
    t_total = len(raw)

    ls = COCO["left_shoulder"]
    rs = COCO["right_shoulder"]
    le = COCO["left_elbow"]
    re = COCO["right_elbow"]
    lw = COCO["left_wrist"]
    rw = COCO["right_wrist"]
    lh = COCO["left_hip"]
    rh = COCO["right_hip"]
    lk = COCO["left_knee"]
    rk = COCO["right_knee"]
    la = COCO["left_ankle"]
    ra = COCO["right_ankle"]
    nose = COCO["nose"]

    raw_hip_centers: list[np.ndarray] = []
    raw_shoulder_centers: list[np.ndarray] = []
    raw_ankle_centers: list[np.ndarray] = []
    raw_wrist_centers: list[np.ndarray] = []
    torso_scales: list[float] = []
    ankle_heights: list[float] = []
    hip_heights: list[float] = []
    mean_confidences: list[float] = []

    handstand_alignment_scores: list[float] = []
    elbow_angles: list[float] = []
    knee_angles: list[float] = []
    shoulder_opening_angles: list[float] = []
    hip_arch_angles: list[float] = []
    bridge_spans: list[float] = []
    torso_tilt_values: list[float] = []
    foot_separation_values: list[float] = []
    shoulder_height_over_hip: list[float] = []
    head_height_over_hip: list[float] = []

    for t in range(t_total):
        pts_r = raw[t]
        pts_n = norm[t]

        shoulder_center_r = _midpoint(pts_r[ls, :2], pts_r[rs, :2])
        hip_center_r = _midpoint(pts_r[lh, :2], pts_r[rh, :2])
        ankle_center_r = _midpoint(pts_r[la, :2], pts_r[ra, :2])
        wrist_center_r = _midpoint(pts_r[lw, :2], pts_r[rw, :2])

        shoulder_center_n = _midpoint(pts_n[ls, :2], pts_n[rs, :2])
        hip_center_n = _midpoint(pts_n[lh, :2], pts_n[rh, :2])
        ankle_center_n = _midpoint(pts_n[la, :2], pts_n[ra, :2])
        wrist_center_n = _midpoint(pts_n[lw, :2], pts_n[rw, :2])

        raw_hip_centers.append(hip_center_r)
        raw_shoulder_centers.append(shoulder_center_r)
        raw_ankle_centers.append(ankle_center_r)
        raw_wrist_centers.append(wrist_center_r)

        torso_scale = _distance(shoulder_center_r, hip_center_r)
        torso_scales.append(torso_scale)
        ankle_heights.append(float(ankle_center_r[1]) if _valid_point(ankle_center_r) else float("nan"))
        hip_heights.append(float(hip_center_r[1]) if _valid_point(hip_center_r) else float("nan"))

        conf_vals = pts_r[:, 2] if pts_r.shape[1] > 2 else np.ones(lay.num_keypoints, dtype=np.float32)
        conf_vals = conf_vals[np.isfinite(conf_vals)]
        mean_confidences.append(float(np.mean(conf_vals)) if conf_vals.size else float("nan"))

        handstand_alignment_scores.append(
            _vertical_alignment_score(
                [
                    float(wrist_center_r[0]) if _valid_point(wrist_center_r) else float("nan"),
                    float(shoulder_center_r[0]) if _valid_point(shoulder_center_r) else float("nan"),
                    float(hip_center_r[0]) if _valid_point(hip_center_r) else float("nan"),
                    float(ankle_center_r[0]) if _valid_point(ankle_center_r) else float("nan"),
                ],
                torso_scale,
            )
        )

        left_elbow = _angle_deg(pts_n[ls, :2], pts_n[le, :2], pts_n[lw, :2])
        right_elbow = _angle_deg(pts_n[rs, :2], pts_n[re, :2], pts_n[rw, :2])
        elbow_angles.append(_safe_mean([left_elbow, right_elbow]))

        left_knee = _angle_deg(pts_n[lh, :2], pts_n[lk, :2], pts_n[la, :2])
        right_knee = _angle_deg(pts_n[rh, :2], pts_n[rk, :2], pts_n[ra, :2])
        knee_angles.append(_safe_mean([left_knee, right_knee]))

        left_shoulder_open = _angle_deg(pts_n[lw, :2], pts_n[ls, :2], pts_n[lh, :2])
        right_shoulder_open = _angle_deg(pts_n[rw, :2], pts_n[rs, :2], pts_n[rh, :2])
        shoulder_opening_angles.append(_safe_mean([left_shoulder_open, right_shoulder_open]))

        left_hip_arch = _angle_deg(pts_n[ls, :2], pts_n[lh, :2], pts_n[lk, :2])
        right_hip_arch = _angle_deg(pts_n[rs, :2], pts_n[rh, :2], pts_n[rk, :2])
        hip_arch_angles.append(_safe_mean([left_hip_arch, right_hip_arch]))

        bridge_spans.append(_distance(wrist_center_n, ankle_center_n))

        torso_tilt_values.append(
            _normalized_distance(shoulder_center_r, np.array([hip_center_r[0], shoulder_center_r[1]]), torso_scale)
            if _valid_point(shoulder_center_r) and _valid_point(hip_center_r)
            else float("nan")
        )
        foot_separation_values.append(_normalized_distance(pts_r[la, :2], pts_r[ra, :2], torso_scale))

        if _valid_point(shoulder_center_r) and _valid_point(hip_center_r):
            shoulder_height_over_hip.append(float((hip_center_r[1] - shoulder_center_r[1]) / max(torso_scale, 1e-6)))
        else:
            shoulder_height_over_hip.append(float("nan"))

        if _valid_point(pts_r[nose, :2]) and _valid_point(hip_center_r):
            head_height_over_hip.append(float((hip_center_r[1] - pts_r[nose, 1]) / max(torso_scale, 1e-6)))
        else:
            head_height_over_hip.append(float("nan"))

    torso_scale = _safe_percentile(torso_scales, 50)
    if not np.isfinite(torso_scale) or torso_scale < 1e-6:
        torso_scale = 1.0

    inv_idx = _nanargmin(ankle_heights, fallback=0)
    top_count = max(1, int(0.15 * t_total))
    ankle_arr = np.asarray(ankle_heights, dtype=np.float32)
    finite_ankle_idx = np.where(np.isfinite(ankle_arr))[0]
    if finite_ankle_idx.size:
        ranked = finite_ankle_idx[np.argsort(ankle_arr[finite_ankle_idx])[:top_count]]
        handstand_top_scores = [handstand_alignment_scores[i] for i in ranked]
    else:
        handstand_top_scores = handstand_alignment_scores

    bridge_idx = _nanargmin(hip_arch_angles, fallback=inv_idx)
    bridge_window_left = max(0, bridge_idx - 2)
    bridge_window_right = min(t_total, bridge_idx + 3)

    bridge_spans_window = bridge_spans[bridge_window_left:bridge_window_right]
    shoulder_open_window = shoulder_opening_angles[bridge_window_left:bridge_window_right]
    hip_arch_window = hip_arch_angles[bridge_window_left:bridge_window_right]

    bridge_drop_from_inverted = float((hip_heights[bridge_idx] - hip_heights[inv_idx]) / torso_scale)
    if not np.isfinite(bridge_drop_from_inverted):
        bridge_drop_from_inverted = float("nan")

    landing_start_idx = max(bridge_idx + 1, int(0.80 * t_total))
    landing_hips = raw_hip_centers[landing_start_idx:]
    landing_shoulders = raw_shoulder_centers[landing_start_idx:]
    landing_ankles = raw_ankle_centers[landing_start_idx:]

    landing_torso_tilt = torso_tilt_values[landing_start_idx:]
    landing_foot_sep = foot_separation_values[landing_start_idx:]
    landing_shoulder_height = shoulder_height_over_hip[landing_start_idx:]
    landing_head_height = head_height_over_hip[landing_start_idx:]

    landing_center_mean = np.nanmean(np.asarray(landing_hips, dtype=np.float32), axis=0)
    landing_sway_vals = [
        float(np.linalg.norm(p - landing_center_mean) / torso_scale)
        for p in landing_hips
        if _valid_point(p) and _valid_point(landing_center_mean)
    ]

    landing_step_vals: list[float] = []
    for prev, cur in zip(landing_ankles[:-1], landing_ankles[1:]):
        if _valid_point(prev) and _valid_point(cur):
            landing_step_vals.append(float(np.linalg.norm(cur - prev) / torso_scale))

    features = {
        "num_frames": int(t_total),
        "inverted_frame_idx": int(inv_idx),
        "inverted_frame_ratio": float(inv_idx / max(1, t_total - 1)),
        "bridge_frame_idx": int(bridge_idx),
        "bridge_frame_ratio": float(bridge_idx / max(1, t_total - 1)),
        "landing_start_idx": int(landing_start_idx),
        "handstand_alignment_min": _safe_min(handstand_top_scores),
        "handstand_alignment_p10": _safe_percentile(handstand_top_scores, 10),
        "elbow_angle_min_deg": _safe_min(elbow_angles),
        "elbow_angle_p10_deg": _safe_percentile(elbow_angles, 10),
        "knee_angle_min_deg": _safe_min(knee_angles),
        "knee_angle_p10_deg": _safe_percentile(knee_angles, 10),
        "shoulder_opening_min_deg": _safe_min(shoulder_open_window),
        "shoulder_opening_p10_deg": _safe_percentile(shoulder_opening_angles, 10),
        "bridge_hip_angle_min_deg": _safe_min(hip_arch_window),
        "bridge_span_min": _safe_min(bridge_spans_window),
        "bridge_drop_from_inverted": bridge_drop_from_inverted,
        "landing_torso_tilt_mean": _safe_mean(landing_torso_tilt),
        "landing_torso_tilt_max": _safe_max(landing_torso_tilt),
        "landing_step_mean": _safe_mean(landing_step_vals),
        "landing_step_max": _safe_max(landing_step_vals),
        "landing_foot_separation_mean": _safe_mean(landing_foot_sep),
        "landing_foot_separation_max": _safe_max(landing_foot_sep),
        "landing_hip_sway_mean": _safe_mean(landing_sway_vals),
        "landing_hip_sway_max": _safe_max(landing_sway_vals),
        "landing_shoulder_height_over_hip_mean": _safe_mean(landing_shoulder_height),
        "landing_head_height_over_hip_mean": _safe_mean(landing_head_height),
        "confidence_mean": _safe_mean(mean_confidences),
    }
    return features


def _save_rows_to_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_reference_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_videos": 0, "feature_stats": {}}

    ignore = {"video_name", "video_path"}
    feature_names = [k for k in rows[0].keys() if k not in ignore]
    summary: dict[str, Any] = {"num_videos": len(rows), "feature_stats": {}}

    for name in feature_names:
        values = [float(r[name]) for r in rows if np.isfinite(r[name])]
        if not values:
            continue
        summary["feature_stats"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return summary


def build_reference_features_for_folder(
    video_folder: str,
    processed_folder: str,
    backend: str = "yolo",
    yolo_model: str | None = None,
    openpose_bin: str | None = None,
    smooth_window: int = 5,
    use_smoothing: bool = True,
    layout: str = "coco17",
    file_prefix: str = "reference_forward_walkover_good",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.makedirs(processed_folder, exist_ok=True)

    video_paths = sorted(
        os.path.join(video_folder, name)
        for name in os.listdir(video_folder)
        if name.lower().endswith((".mp4", ".avi", ".mov"))
    )
    if not video_paths:
        raise FileNotFoundError(f"В папке нет видео: {video_folder}")

    rows: list[dict[str, Any]] = []
    for video_path in video_paths:
        landmarks = extract_pose_from_video(
            video_path,
            show_video=False,
            fill_missing=True,
            backend=backend,  # type: ignore[arg-type]
            openpose_bin=openpose_bin,
            yolo_model=yolo_model,
        )
        if landmarks is None:
            continue

        processed = landmarks
        if use_smoothing:
            processed = smooth_pose_sequence(
                landmarks,
                layout=layout,
                window_size=smooth_window,
            )
            if processed is None:
                continue

        normalized = normalize_poses(processed, layout=layout)
        if normalized is None:
            continue

        safe_name = os.path.basename(video_path).replace(" ", "_")
        np.save(
            os.path.join(processed_folder, f"{safe_name}_landmarks_{layout}.npy"),
            landmarks,
        )
        np.save(
            os.path.join(processed_folder, f"{safe_name}_smoothed_{layout}.npy"),
            processed,
        )
        np.save(
            os.path.join(processed_folder, f"{safe_name}_normalized_{layout}.npy"),
            normalized,
        )

        features = extract_forward_walkover_features_from_sequences(
            processed,
            normalized,
            layout=layout,
        )
        features["video_name"] = os.path.basename(video_path)
        features["video_path"] = video_path
        rows.append(features)

    summary = summarize_reference_features(rows)

    csv_path = os.path.join(processed_folder, f"{file_prefix}_features.csv")
    json_path = os.path.join(processed_folder, f"{file_prefix}_summary.json")
    _save_rows_to_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return rows, summary
