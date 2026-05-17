from __future__ import annotations

"""Метаданные раскладок ключевых точек для нормализации и сохранения."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseLayout:
    name: str
    num_keypoints: int
    feat_per_kp: int  # например 3: x, y, confidence
    left_shoulder: int
    right_shoulder: int
    left_hip: int
    right_hip: int


# COCO Keypoints (как в Ultralytics YOLO-pose): 17 точек
COCO17 = PoseLayout(
    name="coco17",
    num_keypoints=17,
    feat_per_kp=3,
    left_shoulder=5,
    right_shoulder=6,
    left_hip=11,
    right_hip=12,
)

# OpenPose BODY_25: порядок как в выводе --write_json
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
BODY25 = PoseLayout(
    name="body25",
    num_keypoints=25,
    feat_per_kp=3,
    left_shoulder=5,
    right_shoulder=2,
    left_hip=12,
    right_hip=9,
)

LAYOUTS = {
    "coco17": COCO17,
    "body25": BODY25,
}
