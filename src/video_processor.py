from __future__ import annotations

import os

import numpy as np

from src.pose_detection import extract_pose_from_video
from src.pose_schema import LAYOUTS, PoseLayout

COCO17_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

BODY25_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
]


def _get_connections(layout: str | PoseLayout) -> list[tuple[int, int]]:
    name = layout.name if isinstance(layout, PoseLayout) else layout.lower()
    if name == "body25":
        return BODY25_CONNECTIONS
    return COCO17_CONNECTIONS


def _get_scale(m: np.ndarray, lay: PoseLayout, prev_scale: float | None = None) -> float:
    shoulders = m[[lay.left_shoulder, lay.right_shoulder], :2]
    hips = m[[lay.left_hip, lay.right_hip], :2]

    shoulder_width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    hip_width = float(np.linalg.norm(hips[0] - hips[1]))
    shoulder_center = shoulders.mean(axis=0)
    hip_center = hips.mean(axis=0)
    torso_height = float(np.linalg.norm(shoulder_center - hip_center))

    candidates = [shoulder_width, hip_width, torso_height]
    finite_candidates = [v for v in candidates if np.isfinite(v)]
    scale = max(finite_candidates) if finite_candidates else 0.0

    min_scale = 0.05
    if scale < min_scale:
        if prev_scale is not None and np.isfinite(prev_scale) and prev_scale >= min_scale:
            return prev_scale
        return min_scale
    return scale


def _best_frame(seq: np.ndarray, feat_per_kp: int) -> np.ndarray | None:
    best_row = None
    best_score = -np.inf
    for row in seq:
        if np.isnan(row).any():
            continue
        pts = row.reshape(-1, feat_per_kp)
        score = float(np.nanmean(pts[:, 2])) if feat_per_kp > 2 else 1.0
        if score > best_score:
            best_score = score
            best_row = pts
    return best_row


def _robust_limits(xy: np.ndarray) -> tuple[float, float, float, float]:
    valid = xy[np.isfinite(xy).all(axis=1)]
    if len(valid) == 0:
        return -1.0, 1.0, -1.0, 1.0

    x_low, x_high = np.percentile(valid[:, 0], [2, 98])
    y_low, y_high = np.percentile(valid[:, 1], [2, 98])
    if abs(x_high - x_low) < 1e-6:
        x_low -= 1.0
        x_high += 1.0
    if abs(y_high - y_low) < 1e-6:
        y_low -= 1.0
        y_high += 1.0

    pad_x = max(0.1, (x_high - x_low) * 0.15)
    pad_y = max(0.1, (y_high - y_low) * 0.15)
    return (
        float(x_low - pad_x),
        float(x_high + pad_x),
        float(y_low - pad_y),
        float(y_high + pad_y),
    )


def _draw_connections(ax, xy: np.ndarray, conf: np.ndarray, connections: list[tuple[int, int]], conf_thr: float = 0.15) -> None:
    for a, b in connections:
        if a >= len(xy) or b >= len(xy):
            continue
        if not (
            np.isfinite(xy[a]).all()
            and np.isfinite(xy[b]).all()
            and conf[a] >= conf_thr
            and conf[b] >= conf_thr
        ):
            continue
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color="gray",
            linewidth=1.5,
            alpha=0.7,
        )


def _conf_color_bgr(conf: float) -> tuple[int, int, int]:
    conf = float(np.clip(conf, 0.0, 1.0))
    # BGR: низкая уверенность -> красный, высокая -> зеленый.
    red = int(255 * (1.0 - conf))
    green = int(255 * conf)
    return (0, green, red)


def smooth_pose_sequence(
    pose_sequence: np.ndarray | None,
    layout: str | PoseLayout = "coco17",
    window_size: int = 5,
) -> np.ndarray | None:
    """
    Временное сглаживание координат ключевых точек.

    Сглаживаются только x/y-координаты; confidence усредняется по окну.
    Для каждой точки используется взвешивание по confidence и по расстоянию до центра окна.
    """
    if pose_sequence is None or len(pose_sequence) == 0:
        return None

    if window_size <= 1:
        return pose_sequence.astype(np.float32, copy=True)

    if isinstance(layout, str):
        lay = LAYOUTS[layout.lower()]
    else:
        lay = layout

    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    kernel = np.arange(1, radius + 2, dtype=np.float32)
    kernel = np.concatenate([kernel, kernel[-2::-1]])

    seq = pose_sequence.reshape(len(pose_sequence), lay.num_keypoints, lay.feat_per_kp).astype(np.float32, copy=True)
    smoothed = seq.copy()

    for kp_idx in range(lay.num_keypoints):
        xy = seq[:, kp_idx, :2]
        conf = seq[:, kp_idx, 2] if lay.feat_per_kp > 2 else np.ones(len(seq), dtype=np.float32)
        valid_xy = np.isfinite(xy).all(axis=1)
        valid_conf = np.isfinite(conf)
        valid = valid_xy & valid_conf

        for t in range(len(seq)):
            left = max(0, t - radius)
            right = min(len(seq), t + radius + 1)
            local_xy = xy[left:right]
            local_conf = conf[left:right]
            local_valid = valid[left:right]
            local_kernel = kernel[radius - (t - left): radius + (right - t)]

            if not np.any(local_valid):
                continue

            weights = local_kernel[local_valid] * np.clip(local_conf[local_valid], 1e-3, 1.0)
            if np.sum(weights) <= 1e-8:
                continue

            smoothed[t, kp_idx, 0] = np.average(local_xy[local_valid, 0], weights=weights)
            smoothed[t, kp_idx, 1] = np.average(local_xy[local_valid, 1], weights=weights)
            if lay.feat_per_kp > 2:
                smoothed[t, kp_idx, 2] = float(np.average(local_conf[local_valid], weights=local_kernel[local_valid]))

    return smoothed.reshape(len(seq), -1).astype(np.float32, copy=False)


def render_pose_overlay_video(
    video_path: str,
    pose_sequence: np.ndarray | None,
    output_path: str,
    layout: str | PoseLayout = "coco17",
    conf_thr: float = 0.15,
) -> None:
    import cv2

    if pose_sequence is None or len(pose_sequence) == 0:
        raise ValueError("Нет данных позы для наложения на видео")

    if isinstance(layout, str):
        lay = LAYOUTS[layout.lower()]
    else:
        lay = layout

    seq = pose_sequence.reshape(len(pose_sequence), lay.num_keypoints, lay.feat_per_kp)
    connections = _get_connections(lay)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Не удалось определить размеры видео")

    os_dir = os.path.dirname(output_path)
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps if fps > 0 else 25.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось открыть VideoWriter для {output_path}")

    frame_idx = 0
    total = len(seq)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < total:
            pts = seq[frame_idx]
            xy = pts[:, :2]
            conf = pts[:, 2] if lay.feat_per_kp > 2 else np.ones(lay.num_keypoints, dtype=np.float32)

            pixel_xy = np.empty_like(xy, dtype=np.float32)
            pixel_xy[:, 0] = xy[:, 0] * width
            pixel_xy[:, 1] = xy[:, 1] * height

            for a, b in connections:
                if (
                    a < len(pixel_xy)
                    and b < len(pixel_xy)
                    and np.isfinite(pixel_xy[a]).all()
                    and np.isfinite(pixel_xy[b]).all()
                    and conf[a] >= conf_thr
                    and conf[b] >= conf_thr
                ):
                    pa = tuple(np.round(pixel_xy[a]).astype(int))
                    pb = tuple(np.round(pixel_xy[b]).astype(int))
                    cv2.line(frame, pa, pb, (180, 180, 180), 2, cv2.LINE_AA)

            for kp_idx in range(len(pixel_xy)):
                if not np.isfinite(pixel_xy[kp_idx]).all() or conf[kp_idx] < conf_thr:
                    continue
                px, py = np.round(pixel_xy[kp_idx]).astype(int)
                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    _conf_color_bgr(float(conf[kp_idx])),
                    -1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"frame {frame_idx + 1}/{max(total, frame_idx + 1)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def normalize_poses(
    poses_sequence: np.ndarray | None,
    layout: str | PoseLayout = "coco17",
) -> np.ndarray | None:
    """
    Центрирование относительно середины таза и масштаб по расстоянием между плечами.
    Каждая строка: num_kp * 3 (x, y, confidence); x,y нормированы по размеру кадра [0,1].
    Для углов и масштаба используются только первые две координаты каждой точки.
    """
    if poses_sequence is None or len(poses_sequence) == 0:
        return None

    if isinstance(layout, str):
        lay = LAYOUTS[layout.lower()]
    else:
        lay = layout

    n_feat = lay.num_keypoints * lay.feat_per_kp
    normalized: list[np.ndarray] = []
    prev_scale: float | None = None

    for pose in poses_sequence:
        if np.isnan(pose).any():
            normalized.append(np.full(n_feat, np.nan, dtype=np.float32))
            continue

        m = pose.reshape(-1, lay.feat_per_kp).astype(np.float64, copy=True)
        hip_c = (
            m[lay.left_hip, :2] + m[lay.right_hip, :2]
        ) / 2.0
        m[:, :2] -= hip_c

        scale = _get_scale(m, lay, prev_scale=prev_scale)
        prev_scale = scale
        m[:, :2] /= scale

        normalized.append(m.astype(np.float32).ravel())

    return np.asarray(normalized)


def visualize_pose_sequence(
    pose_sequence: np.ndarray | None,
    sequence_name: str = "Pose",
    layout: str | PoseLayout = "coco17",
) -> None:
    import matplotlib.pyplot as plt

    if pose_sequence is None or len(pose_sequence) == 0:
        print("Нет данных для визуализации")
        return

    if isinstance(layout, str):
        lay = LAYOUTS[layout.lower()]
    else:
        lay = layout

    sample = _best_frame(pose_sequence, lay.feat_per_kp)
    if sample is None:
        print("Все кадры содержат пропуски — нечего отображать")
        return

    xy = sample[:, :2]
    conf = sample[:, 2] if lay.feat_per_kp > 2 else np.ones(len(sample))
    connections = _get_connections(lay)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    _draw_connections(ax1, xy, conf, connections)
    ax1.scatter(xy[:, 0], xy[:, 1], c=conf, cmap="viridis", s=20)
    x_min, x_max, y_min, y_max = _robust_limits(xy)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_max, y_min)
    ax1.set_title(f"{sequence_name} — 2D скелет (цвет = confidence)")
    ax1.set_xlabel("x (норм.)")
    ax1.set_ylabel("y (норм.)")
    ax1.grid(True)
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(xy[:, 0], xy[:, 1], conf, c=conf, cmap="viridis", s=20)
    ax2.set_title(f"{sequence_name} — x, y, confidence")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("conf")

    plt.tight_layout()
    plt.show()

    print(f"Длина последовательности: {len(pose_sequence)}, форма: {pose_sequence.shape}")


def animate_pose_sequence(
    pose_sequence: np.ndarray | None,
    sequence_name: str = "Pose",
    layout: str | PoseLayout = "coco17",
    interval_ms: int = 80,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if pose_sequence is None or len(pose_sequence) == 0:
        print("Нет данных для анимации")
        return

    if isinstance(layout, str):
        lay = LAYOUTS[layout.lower()]
    else:
        lay = layout

    seq = pose_sequence.reshape(len(pose_sequence), lay.num_keypoints, lay.feat_per_kp)
    connections = _get_connections(lay)

    finite_xy = seq[:, :, :2].reshape(-1, 2)
    finite_xy = finite_xy[np.isfinite(finite_xy).all(axis=1)]
    if finite_xy.size == 0:
        print("Все кадры содержат пропуски — анимация невозможна")
        return

    x_min, x_max, y_min, y_max = _robust_limits(finite_xy)

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], s=30, c=[], cmap="viridis", vmin=0.0, vmax=1.0)
    lines = [ax.plot([], [], color="gray", linewidth=1.5, alpha=0.7)[0] for _ in connections]
    title = ax.set_title(sequence_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    def _frame_data(frame_idx: int):
        frame = seq[frame_idx]
        valid_mask = np.isfinite(frame[:, :2]).all(axis=1)
        xy = frame[valid_mask, :2]
        conf = frame[valid_mask, 2] if lay.feat_per_kp > 2 else np.ones(np.sum(valid_mask))
        return frame[:, :2], frame[:, 2] if lay.feat_per_kp > 2 else np.ones(len(frame)), xy, conf

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([], dtype=np.float32))
        for line in lines:
            line.set_data([], [])
        title.set_text(f"{sequence_name} | кадр 0/{len(seq)}")
        return [scatter, title, *lines]

    def update(frame_idx: int):
        full_xy, full_conf, xy, conf = _frame_data(frame_idx)
        scatter.set_offsets(xy if len(xy) else np.empty((0, 2)))
        scatter.set_array(conf.astype(np.float32, copy=False))
        for line, (a, b) in zip(lines, connections):
            if (
                a < len(full_xy)
                and b < len(full_xy)
                and np.isfinite(full_xy[a]).all()
                and np.isfinite(full_xy[b]).all()
                and full_conf[a] >= 0.15
                and full_conf[b] >= 0.15
            ):
                line.set_data(
                    [full_xy[a, 0], full_xy[b, 0]],
                    [full_xy[a, 1], full_xy[b, 1]],
                )
            else:
                line.set_data([], [])
        title.set_text(f"{sequence_name} | кадр {frame_idx + 1}/{len(seq)}")
        return [scatter, title, *lines]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(seq),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path:
        writer = PillowWriter(fps=max(1, int(round(1000 / max(1, interval_ms)))))
        anim.save(save_path, writer=writer)
        print(f"Анимация сохранена: {save_path}")

    plt.show()


def process_video_folder(
    video_folder: str,
    backend: str | None = None,
    openpose_bin: str | None = None,
) -> dict:
    import glob
    import os

    video_files = (
        glob.glob(os.path.join(video_folder, "*.mp4"))
        + glob.glob(os.path.join(video_folder, "*.avi"))
        + glob.glob(os.path.join(video_folder, "*.mov"))
    )

    b = (backend or os.environ.get("POSE_BACKEND", "yolo")).lower()
    layout_name = "body25" if b == "openpose" else "coco17"
    results = {}
    for video_file in video_files:
        print(f"Обработка: {os.path.basename(video_file)}")
        landmarks = extract_pose_from_video(
            video_file,
            show_video=False,
            fill_missing=True,
            backend=backend,  # type: ignore[arg-type]
            openpose_bin=openpose_bin,
        )
        if landmarks is not None:
            results[video_file] = normalize_poses(landmarks, layout=layout_name)

    return results
