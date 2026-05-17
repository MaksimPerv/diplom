from __future__ import annotations

import os
from typing import Literal

import numpy as np

from src.pose_openpose import OpenPoseCLIExtractor
from src.pose_yolo import YOLOPoseExtractor

PoseBackend = Literal["yolo", "openpose"]


def _get_extractor(
    backend: PoseBackend,
    openpose_bin: str | None = None,
    yolo_model: str | None = None,
):
    b = backend.lower().strip()
    if b == "yolo":
        return YOLOPoseExtractor(model_name=yolo_model)
    if b == "openpose":
        return OpenPoseCLIExtractor(openpose_bin=openpose_bin)
    raise ValueError(f"Неизвестный backend: {backend}. Используйте 'yolo' или 'openpose'.")


def extract_pose_from_video(
    video_path: str,
    show_video: bool = False,
    fill_missing: bool = True,
    backend: PoseBackend | None = None,
    openpose_bin: str | None = None,
    yolo_model: str | None = None,
) -> np.ndarray | None:
    """
    Извлечение последовательности поз: одна строка на кадр.

    backend по умолчанию: переменная окружения POSE_BACKEND ('yolo' или 'openpose'),
    иначе 'yolo'.

    Размерность строки: 17*3 (YOLO, COCO) или 25*3 (OpenPose BODY_25), см. layout у экстрактора.
    """
    b = (
        backend
        or os.environ.get("POSE_BACKEND", "yolo").lower().strip()
    )
    extractor = _get_extractor(
        b,
        openpose_bin=openpose_bin,
        yolo_model=yolo_model,
    )  # type: ignore[arg-type]
    return extractor.extract_pose_from_video(
        video_path,
        show_video=show_video,
        fill_missing=fill_missing,
    )


if __name__ == "__main__":
    test_video = os.path.join("data", "raw_videos", "test.mp4")
    if os.path.exists(test_video):
        os.environ.setdefault("POSE_BACKEND", "yolo")
        lm = extract_pose_from_video(test_video, show_video=False)
        if lm is not None:
            print(f"Тест: кадров {len(lm)}, форма {lm.shape}")
    else:
        print("Для теста: data/raw_videos/test.mp4")
