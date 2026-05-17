from __future__ import annotations

import os

import cv2
import numpy as np

from src.pose_schema import COCO17, PoseLayout


class YOLOPoseExtractor:
    """Оценка позы через Ultralytics YOLO-pose (скелет COCO, 17 точек)."""

    def __init__(self, model_name: str | None = None):
        from ultralytics import YOLO

        weights = model_name or os.environ.get("YOLO_POSE_MODEL", "yolov8s-pose.pt")
        self.model = YOLO(weights)
        self.layout: PoseLayout = COCO17
        self.model_name = weights

    def extract_pose_from_video(
        self,
        video_path: str,
        show_video: bool = False,
        fill_missing: bool = True,
    ) -> np.ndarray | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        feat_dim = self.layout.num_keypoints * self.layout.feat_per_kp

        rows: list[np.ndarray] = []
        last_row: np.ndarray | None = None
        frame_count = 0

        print(f"[YOLO-pose] {os.path.basename(video_path)} ({w}x{h}), model={self.model_name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            res = self.model.predict(frame, verbose=False)[0]
            row = np.full(feat_dim, np.nan, dtype=np.float32)

            kpts = res.keypoints
            if (
                kpts is not None
                and kpts.xy is not None
                and kpts.xy.numel() > 0
                and kpts.xy.shape[0] > 0
            ):
                xy = kpts.xy[0].cpu().numpy()
                if kpts.conf is None:
                    conf = np.ones(self.layout.num_keypoints, dtype=np.float32)
                else:
                    conf = kpts.conf[0].cpu().numpy()
                if w > 0 and h > 0:
                    xy_norm = xy.astype(np.float64, copy=True)
                    xy_norm[:, 0] /= float(w)
                    xy_norm[:, 1] /= float(h)
                else:
                    xy_norm = xy

                parts: list[float] = []
                for i in range(self.layout.num_keypoints):
                    parts.extend(
                        [
                            float(xy_norm[i, 0]),
                            float(xy_norm[i, 1]),
                            float(conf[i]),
                        ]
                    )
                row = np.asarray(parts, dtype=np.float32)
                last_row = row.copy()

                if show_video:
                    cv2.imshow("YOLO pose — Q", res.plot())
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                if fill_missing and last_row is not None:
                    row = last_row.copy()

            rows.append(row)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  кадров: {frame_count}")

        cap.release()
        cv2.destroyAllWindows()

        if not rows:
            return None
        return np.stack(rows, axis=0)
