from __future__ import annotations

import argparse
import os

import numpy as np

from src.feature_extraction import build_reference_features_for_folder
from src.pose_detection import extract_pose_from_video
from src.video_processor import (
    animate_pose_sequence,
    normalize_poses,
    render_pose_overlay_video,
    smooth_pose_sequence,
    visualize_pose_sequence,
)


def _layout_for_backend(backend: str) -> str:
    return "body25" if backend.lower() == "openpose" else "coco17"


def _infer_layout_from_path(path: str, fallback: str) -> str:
    lower = path.lower()
    if "body25" in lower:
        return "body25"
    if "coco17" in lower:
        return "coco17"
    return fallback


def main() -> None:
    p = argparse.ArgumentParser(description="Извлечение позы: YOLO-pose или OpenPose CLI")
    p.add_argument(
        "--video",
        default=None,
        help="Путь к видеофайлу. Если не указан, берется первый файл из data/raw_videos",
    )
    p.add_argument(
        "--view-npy",
        default=None,
        help="Путь к .npy файлу для просмотра сохраненной последовательности поз",
    )
    p.add_argument(
        "--build-reference",
        default=None,
        help="Путь к папке с эталонными видео для построения сводки признаков",
    )
    p.add_argument(
        "--backend",
        choices=["yolo", "openpose"],
        default=os.environ.get("POSE_BACKEND", "yolo"),
        help="yolo — Ultralytics (по умолчанию); openpose — бинарник, см. OPENPOSE_BIN",
    )
    p.add_argument(
        "--openpose-bin",
        default=None,
        help="Путь к openpose.bin / OpenPoseDemo.exe (иначе OPENPOSE_BIN)",
    )
    p.add_argument(
        "--yolo-model",
        default=os.environ.get("YOLO_POSE_MODEL", "yolov8s-pose.pt"),
        help="Весы YOLO pose, например yolov8s-pose.pt или yolov8m-pose.pt",
    )
    p.add_argument("--show", action="store_true", help="Показать окно с наложением скелета")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Показать график позы через matplotlib после обработки",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Показать анимацию последовательности поз кадр за кадром",
    )
    p.add_argument(
        "--save-animation",
        default=None,
        help="Путь для сохранения анимации в GIF, например data/processed/front_walkover.gif",
    )
    p.add_argument(
        "--save-overlay",
        default=None,
        help="Путь для сохранения видео с наложенным скелетом, например data/processed/front_walkover_overlay.mp4",
    )
    p.add_argument(
        "--reference-prefix",
        default="reference_forward_walkover_good",
        help="Префикс файлов для CSV/JSON со сводкой эталонной техники",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Размер окна сглаживания по времени для ключевых точек",
    )
    p.add_argument(
        "--no-smooth",
        action="store_true",
        help="Отключить временное сглаживание ключевых точек",
    )
    args = p.parse_args()

    selected_modes = [
        bool(args.video),
        bool(args.view_npy),
        bool(args.build_reference),
    ]
    if sum(selected_modes) > 1:
        print("Используйте только один режим: --video, --view-npy или --build-reference")
        return

    os.environ["POSE_BACKEND"] = args.backend
    os.environ["YOLO_POSE_MODEL"] = args.yolo_model

    video_folder = "data/raw_videos"
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if args.build_reference is not None:
        if not os.path.isdir(args.build_reference):
            print(f"Папка не найдена: {args.build_reference}")
            return

        layout = _layout_for_backend(args.backend)
        print(f"Построение эталонных признаков из папки: {args.build_reference}")
        if args.backend == "yolo":
            print(f"YOLO модель: {args.yolo_model}")
        print(
            "Сглаживание: "
            + ("отключено" if args.no_smooth else f"включено, окно={args.smooth_window}")
        )

        rows, summary = build_reference_features_for_folder(
            video_folder=args.build_reference,
            processed_folder="data/processed",
            backend=args.backend,
            yolo_model=args.yolo_model,
            openpose_bin=args.openpose_bin,
            smooth_window=args.smooth_window,
            use_smoothing=not args.no_smooth,
            layout=layout,
            file_prefix=args.reference_prefix,
        )
        print(f"Обработано видео: {len(rows)}")
        print(
            f"Сохранен CSV: data/processed/{args.reference_prefix}_features.csv"
        )
        print(
            f"Сохранен JSON: data/processed/{args.reference_prefix}_summary.json"
        )
        if summary.get("feature_stats"):
            print("Доступные признаки в сводке:")
            for key in summary["feature_stats"].keys():
                print(f"  - {key}")
        return

    if args.view_npy is not None:
        if not os.path.isfile(args.view_npy):
            print(f"Файл не найден: {args.view_npy}")
            return

        layout = _infer_layout_from_path(args.view_npy, _layout_for_backend(args.backend))
        arr = np.load(args.view_npy)
        print(f"Загружен файл: {args.view_npy}")
        print(f"Форма массива: {arr.shape}, dtype={arr.dtype}, layout={layout}")

        if arr.ndim != 2:
            print("Ожидался двумерный массив формата (число_кадров, число_признаков)")
            return

        finite_ratio = float(np.isfinite(arr).sum()) / float(arr.size) if arr.size else 0.0
        print(f"Доля конечных значений: {finite_ratio:.3f}")

        visualize_pose_sequence(
            arr,
            sequence_name=os.path.basename(args.view_npy),
            layout=layout,
        )
        if args.animate or args.save_animation:
            animate_pose_sequence(
                arr,
                sequence_name=os.path.basename(args.view_npy),
                layout=layout,
                save_path=args.save_animation,
            )
        return

    if args.video is not None:
        first_video = args.video
        if not os.path.isfile(first_video):
            print(f"Файл не найден: {first_video}")
            return
    else:
        video_files = [
            f
            for f in os.listdir(video_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
        if not video_files:
            print("Нет видео в data/raw_videos (.mp4, .avi, .mov)")
            return
        first_video = os.path.join(video_folder, sorted(video_files)[0])

    layout = _layout_for_backend(args.backend)
    print(f"Файл: {os.path.basename(first_video)}, backend={args.backend}, layout={layout}")
    if args.backend == "yolo":
        print(f"YOLO модель: {args.yolo_model}")
    print(
        "Сглаживание: "
        + ("отключено" if args.no_smooth else f"включено, окно={args.smooth_window}")
    )

    try:
        landmarks = extract_pose_from_video(
            first_video,
            show_video=args.show,
            fill_missing=True,
            backend=args.backend,
            openpose_bin=args.openpose_bin,
            yolo_model=args.yolo_model,
        )
        if landmarks is None:
            print("Не удалось извлечь позу")
            return

        print(f"Кадров: {len(landmarks)}, форма: {landmarks.shape}")

        processed = landmarks
        if not args.no_smooth:
            processed = smooth_pose_sequence(
                landmarks,
                layout=layout,
                window_size=args.smooth_window,
            )
            if processed is None:
                print("Не удалось сгладить последовательность поз")
                return

        normalized = normalize_poses(processed, layout=layout)
        if normalized is None:
            print("Не удалось нормализовать последовательность поз")
            return

        safe_name = os.path.basename(first_video).replace(" ", "_")
        raw_out = os.path.join(
            "data",
            "processed",
            f"{safe_name}_landmarks_{layout}.npy",
        )
        norm_out = os.path.join(
            "data",
            "processed",
            f"{safe_name}_normalized_{layout}.npy",
        )
        smooth_out = os.path.join(
            "data",
            "processed",
            f"{safe_name}_smoothed_{layout}.npy",
        )
        np.save(raw_out, landmarks)
        np.save(smooth_out, processed)
        np.save(norm_out, normalized)
        print(f"Сохранены ключевые точки: {raw_out}")
        print(f"Сохранены сглаженные точки: {smooth_out}")
        print(f"Сохранены нормализованные позы: {norm_out}")

        if args.save_overlay:
            print("Сохранение видео с наложенным скелетом...")
            render_pose_overlay_video(
                first_video,
                processed,
                args.save_overlay,
                layout=layout,
            )
            print(f"Сохранено видео с оверлеем: {args.save_overlay}")

        if args.plot:
            print("Визуализация (matplotlib)...")
            visualize_pose_sequence(
                normalized,
                f"{safe_name} ({args.backend}, normalized)",
                layout=layout,
            )
        if args.animate or args.save_animation:
            print("Анимация последовательности поз...")
            animate_pose_sequence(
                normalized,
                f"{safe_name} ({args.backend}, normalized)",
                layout=layout,
                save_path=args.save_animation,
            )
    except Exception as e:
        print(f"Ошибка: {e}")
        if args.backend == "openpose":
            print(
                "Подсказка: соберите OpenPose из исходников или portable-сборку, "
                "затем export OPENPOSE_BIN=/полный/путь/к/openpose.bin"
            )


if __name__ == "__main__":
    main()
