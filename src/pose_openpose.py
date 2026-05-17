from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

from src.pose_schema import BODY25, PoseLayout


def _default_openpose_binary() -> str:
    return os.environ.get("OPENPOSE_BIN", "")


class OpenPoseCLIExtractor:
    """
    Запуск готовой сборки OpenPose (CLI) с --write_json.

    Задайте полный путь к исполняемому файлу в переменной OPENPOSE_BIN, например:
      export OPENPOSE_BIN=/home/user/openpose/build/examples/openpose/openpose.bin
    """

    def __init__(self, openpose_bin: str | None = None):
        self.openpose_bin = (openpose_bin or _default_openpose_binary()).strip()
        self.layout: PoseLayout = BODY25

    def extract_pose_from_video(
        self,
        video_path: str,
        show_video: bool = False,
        fill_missing: bool = True,
    ) -> np.ndarray | None:
        if not self.openpose_bin:
            raise RuntimeError(
                "Не задан OPENPOSE_BIN — полный путь к openpose.bin или OpenPoseDemo.exe"
            )
        if not os.path.isfile(self.openpose_bin):
            raise FileNotFoundError(f"OPENPOSE_BIN не найден: {self.openpose_bin}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        feat_dim = self.layout.num_keypoints * self.layout.feat_per_kp
        pat = re.compile(r"(\d+)_keypoints\.json$")

        with tempfile.TemporaryDirectory(prefix="openpose_json_") as tmp:
            cmd = [
                self.openpose_bin,
                "--video",
                video_path,
                "--write_json",
                tmp,
                "--display",
                "1" if show_video else "0",
                "--render_pose",
                "1" if show_video else "0",
            ]
            print(f"[OpenPose] {os.path.basename(video_path)}")
            subprocess.run(cmd, check=True)

            per_index: dict[int, np.ndarray] = {}
            for name in os.listdir(tmp):
                m = pat.search(name)
                if not m:
                    continue
                idx = int(m.group(1))
                path = os.path.join(tmp, name)
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                people = data.get("people") or []
                if not people:
                    continue
                flat = people[0].get("pose_keypoints_2d") or []
                if len(flat) < feat_dim:
                    continue
                arr = np.asarray(flat[:feat_dim], dtype=np.float32).reshape(-1, 3)
                if w > 0 and h > 0:
                    arr[:, 0] /= float(w)
                    arr[:, 1] /= float(h)
                per_index[idx] = arr.ravel()

        if not per_index:
            print("OpenPose не вернул ни одного JSON с людьми")
            return None

        max_idx = max(per_index.keys())
        total = n_frames if n_frames > 0 else max_idx + 1

        rows: list[np.ndarray] = []
        last_row: np.ndarray | None = None
        for i in range(total):
            if i in per_index:
                row = per_index[i]
                last_row = row.copy()
            elif fill_missing and last_row is not None:
                row = last_row.copy()
            else:
                row = np.full(feat_dim, np.nan, dtype=np.float32)
            rows.append(row)

        return np.stack(rows, axis=0)
