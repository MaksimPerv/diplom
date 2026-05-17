#!/usr/bin/env python3
"""
Обучение моделей оценки техники переворота вперед по разметке CSV + видео.

Шкала классов (class_id):
  0 — Плохо
  1 — Удовлетворительно
  2 — Хорошо
  3 — Отлично

Для обучения классификатора нужны примеры минимум двух разных class_id.
Если в data/labels.csv только один класс,
используйте флаг --smoke-test: подставляются временные метки [3,2,3,...]
только чтобы проверить, что конвейер и обучение работают.

Примеры:
  ./.venv/bin/python train_score_model.py --prepare-only
  ./.venv/bin/python train_score_model.py --smoke-test --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.score_classes import QUALITY_CLASS_NAMES_RU
from src.score_dataset import build_samples_from_labels
from src.score_training import (
    save_metrics,
    save_sklearn_pipeline,
    save_torch_bundle,
    train_sklearn_mlp_tabular,
    train_torch_gru_classifier,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _resample_sequence_batch(X_seq: np.ndarray, target_len: int) -> np.ndarray:
    """Меняет длину уже подготовленной последовательности без повторного запуска YOLO."""
    if X_seq.shape[1] == target_len:
        return X_seq.astype(np.float32, copy=False)

    n, old_len, dim = X_seq.shape
    t_src = np.arange(old_len, dtype=np.float32)
    t_dst = np.linspace(0.0, float(old_len - 1), target_len, dtype=np.float32)
    out = np.zeros((n, target_len, dim), dtype=np.float32)
    for i in range(n):
        for d in range(dim):
            out[i, :, d] = np.interp(t_dst, t_src, X_seq[i, :, d])
    return out


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    return {str(int(cls)): int(np.sum(y == cls)) for cls in sorted(np.unique(y))}


def _save_predictions_csv(
    metrics: dict[str, Any],
    filenames: list[str],
    path: str,
) -> None:
    rows = []
    for idx, true, pred in zip(metrics["val_indices"], metrics["y_true"], metrics["y_pred"]):
        true_i = int(true)
        pred_i = int(pred)
        rows.append(
            {
                "filename": filenames[int(idx)] if int(idx) < len(filenames) else "",
                "true_class": true_i,
                "true_name": QUALITY_CLASS_NAMES_RU.get(true_i, str(true_i)),
                "pred_class": pred_i,
                "pred_name": QUALITY_CLASS_NAMES_RU.get(pred_i, str(pred_i)),
                "delta": pred_i - true_i,
                "is_correct": int(true_i == pred_i),
                "is_adjacent_error": int(true_i != pred_i and abs(pred_i - true_i) == 1),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _confusion_notes(metrics: dict[str, Any]) -> list[str]:
    cm = np.asarray(metrics.get("confusion_matrix", []), dtype=int)
    labels = [int(x) for x in metrics.get("class_labels", [])]
    notes: list[tuple[int, str]] = []
    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            if i != j and cm.size and cm[i, j] > 0:
                notes.append(
                    (
                        int(cm[i, j]),
                        f"{cm[i, j]} видео класса {true_cls} предсказаны как {pred_cls}",
                    )
                )
    notes.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in notes[:5]]


def _write_experiment_report(
    *,
    out_dir: str,
    meta: dict[str, Any],
    y: np.ndarray,
    metrics_by_model: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    lines = [
        "# Отчет по эксперименту оценки переворота вперед",
        "",
        "## Датасет",
        "",
        f"- Видео: {len(y)}",
        f"- Разметка: {meta.get('labels_file', 'data/labels.csv')}",
        f"- Скелет: {meta.get('layout', 'coco17')}",
        f"- Длина последовательности: {meta.get('seq_len')}",
        f"- Табличные признаки: {meta.get('feature_dim_tab')}",
        f"- Признаки последовательности: {meta.get('feature_dim_seq')}",
        "",
        "Распределение классов:",
        "",
    ]
    for cls, count in _class_distribution(y).items():
        name = QUALITY_CLASS_NAMES_RU.get(int(cls), cls)
        lines.append(f"- {cls} ({name}): {count}")

    lines.extend(["", "## Основные модели", ""])
    lines.append("| Модель | Accuracy | Macro F1 |")
    lines.append("|---|---:|---:|")
    for model_name, metrics in metrics_by_model.items():
        lines.append(
            f"| {model_name} | {metrics['val_accuracy']:.3f} | {metrics['val_macro_f1']:.3f} |"
        )

    lines.extend(["", "## Диагностика ошибок", ""])
    for model_name, metrics in metrics_by_model.items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("Confusion matrix:")
        lines.append("")
        labels = [str(x) for x in metrics.get("class_labels", [])]
        cm = metrics.get("confusion_matrix", [])
        lines.append("| true \\ pred | " + " | ".join(labels) + " |")
        lines.append("|---" + "|---:" * len(labels) + "|")
        for label, row in zip(labels, cm):
            lines.append("| " + label + " | " + " | ".join(str(x) for x in row) + " |")
        notes = _confusion_notes(metrics)
        if notes:
            lines.append("")
            lines.append("Наиболее частые ошибки:")
            for note in notes:
                lines.append(f"- {note}")
        lines.append("")

    if comparisons:
        lines.extend(["## Устойчивая оценка и сравнение параметров", ""])
        lines.append("| Модель | seq_len | seed | Accuracy | Macro F1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in comparisons:
            lines.append(
                "| {model} | {seq_len} | {seed} | {accuracy:.3f} | {macro_f1:.3f} |".format(
                    **row
                )
            )

        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in comparisons:
            key = f"{row['model']} seq_len={row['seq_len']}"
            by_model.setdefault(key, []).append(row)
        lines.extend(["", "Средние значения по запускам:", ""])
        lines.append("| Вариант | Accuracy mean | Macro F1 mean |")
        lines.append("|---|---:|---:|")
        for key, rows in by_model.items():
            acc = float(np.mean([r["accuracy"] for r in rows]))
            f1 = float(np.mean([r["macro_f1"] for r in rows]))
            lines.append(f"| {key} | {acc:.3f} | {f1:.3f} |")

    lines.extend(
        [
            "",
            "## Вывод для диплома",
            "",
            "Первый устойчивый эксперимент показывает, что задача решается лучше случайного выбора класса, но соседние уровни качества часто смешиваются. Это ожидаемо для небольшой выборки с ручной разметкой: классы 1/2 и 2/3 могут отличаться несколькими субъективно оцененными ошибками. Для дальнейшего повышения качества нужны расширение набора данных, уточнение разметки спорных видео и более точное выделение фаз элемента.",
            "",
            "YOLO-модель для подготовки этого датасета: `yolov8n-pose.pt`. Сравнение с `yolov8s-pose.pt` требует повторного извлечения поз и занимает примерно столько же, сколько первичная подготовка датасета.",
            "",
        ]
    )

    report_path = os.path.join(out_dir, "experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение оценки техники по видео и labels.csv")
    p.add_argument(
        "--labels",
        default="data/labels.csv",
        help="CSV с колонками filename, class_id",
    )
    p.add_argument("--video-dir", default="data/raw_videos", help="Папка с видео из filename")
    p.add_argument(
        "--out-dir",
        default="data/processed/ml_artifacts",
        help="Куда сохранить веса и метрики",
    )
    p.add_argument("--backend", default="yolo", choices=["yolo", "openpose"])
    p.add_argument("--yolo-model", default=os.environ.get("YOLO_POSE_MODEL", "yolov8s-pose.pt"))
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=96, help="Длина ресемплинга последовательности")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument(
        "--eval-seeds",
        default="42,7,13",
        help="Seed для повторной устойчивой оценки через запятую",
    )
    p.add_argument(
        "--compare-seq-lens",
        default="64,96,128",
        help="Длины последовательности для сравнения GRU без повторного YOLO",
    )
    p.add_argument(
        "--comparison-epochs",
        type=int,
        default=30,
        help="Число эпох для быстрых сравнительных запусков",
    )
    p.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Не запускать повторные сравнения по seed/seq_len",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Временно выставить class_id как [3,2,3,...] для проверки обучения (не для отчёта)",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Только извлечь позы и сохранить dataset.npz, без обучения",
    )
    p.add_argument("--no-gru", action="store_true", help="Не обучать GRU (только sklearn MLP)")
    p.add_argument(
        "--from-npz",
        default=None,
        help="Путь к dataset.npz: не гонять YOLO, взять X_seq/X_tab оттуда, class_id — из CSV (после --smoke-test)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.labels):
        print(f"Нет файла разметки: {args.labels}")
        sys.exit(1)

    df = pd.read_csv(args.labels)
    labels_path = args.labels

    if args.smoke_test:
        print(
            "ВНИМАНИЕ: --smoke-test — искусственные метки классов для проверки кода. "
            "Для диплома соберите реальные примеры разных уровней."
        )
        alt = np.array([3, 2] * (len(df) // 2 + 1))[: len(df)]
        df = df.copy()
        df["class_id"] = alt
        labels_path = os.path.join(args.out_dir, "_smoke_labels.csv")
        df.to_csv(labels_path, index=False)

    unique_classes = sorted(int(x) for x in df["class_id"].unique().tolist())
    names = [QUALITY_CLASS_NAMES_RU[i] for i in unique_classes]
    print("Классы в разметке:", unique_classes, names)

    os.environ.setdefault("POSE_BACKEND", args.backend)
    os.environ.setdefault("YOLO_POSE_MODEL", args.yolo_model)

    if args.from_npz:
        if not os.path.isfile(args.from_npz):
            print(f"Файл не найден: {args.from_npz}")
            sys.exit(1)
        print(f"Загрузка признаков из {args.from_npz} (YOLO не запускается)...")
        z = np.load(args.from_npz)
        df_y = pd.read_csv(labels_path)
        y = df_y["class_id"].astype(int).to_numpy()
        if len(y) != len(z["X_seq"]):
            print("Число строк в CSV не совпадает с размером dataset.npz")
            sys.exit(1)
        meta_path = os.path.join(os.path.dirname(args.from_npz), "dataset_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {
                "layout": "coco17",
                "seq_len": int(z["X_seq"].shape[1]),
                "feature_dim_seq": int(z["X_seq"].shape[2]),
                "feature_dim_tab": int(z["X_tab"].shape[1]),
                "num_classes": 4,
                "paths": [],
                "filenames": df_y["filename"].tolist(),
            }
        bundle = {
            "X_seq": z["X_seq"],
            "X_tab": z["X_tab"],
            "y": y,
            "meta": meta,
        }
    else:
        print("Извлечение поз и построение выборки (может занять время, YOLO на CPU/GPU)...")
        bundle = build_samples_from_labels(
            labels_path,
            args.video_dir,
            layout="coco17",
            backend=args.backend,
            yolo_model=args.yolo_model,
            openpose_bin=os.environ.get("OPENPOSE_BIN"),
            smooth_window=args.smooth_window,
            seq_len=args.seq_len,
            use_smoothing=True,
        )

    ds_path = os.path.join(args.out_dir, "dataset.npz")
    np.savez_compressed(
        ds_path,
        X_seq=bundle["X_seq"],
        X_tab=bundle["X_tab"],
        y=bundle["y"],
    )
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        meta = dict(bundle["meta"])
        meta["labels_file"] = labels_path
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {ds_path}")

    if args.prepare_only:
        print("Режим --prepare-only: обучение пропущено.")
        return

    y = bundle["y"]
    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(
            "\nОбучение классификатора невозможно: в разметке только один class_id.\n"
            "Добавьте видео с другими оценками (0–2) в data/labels.csv или запустите с --smoke-test.\n"
        )
        sys.exit(2)

    print("\n--- Sklearn MLP (табличные признаки переворота вперед) ---")
    pipe, m_tab = train_sklearn_mlp_tabular(bundle["X_tab"], bundle["y"], random_state=42)
    print("val_accuracy:", m_tab["val_accuracy"])
    print("val_macro_f1:", m_tab["val_macro_f1"])
    if m_tab.get("small_sample_train_equals_val"):
        print(
            "(метрики на очень малой выборке: train и val совпадают, "
            "цифры не для научного вывода, только проверка кода)"
        )
    print(m_tab["report"])
    save_sklearn_pipeline(pipe, os.path.join(args.out_dir, "baseline_mlp_tabular.joblib"))
    save_metrics(m_tab, os.path.join(args.out_dir, "metrics_baseline.json"))
    _save_predictions_csv(
        m_tab,
        list(bundle["meta"].get("filenames", [])),
        os.path.join(args.out_dir, "predictions_baseline.csv"),
    )

    metrics_by_model = {"MLP tabular": m_tab}

    if not args.no_gru:
        print("\n--- PyTorch BiGRU (последовательность нормализованных x,y) ---")
        try:
            gru_bundle, m_seq = train_torch_gru_classifier(
                bundle["X_seq"],
                bundle["y"],
                num_classes=int(bundle["meta"]["num_classes"]),
                epochs=args.epochs,
                random_state=42,
            )
        except ImportError as e:
            print("PyTorch не найден:", e)
            sys.exit(1)
        print("val_accuracy:", m_seq["val_accuracy"])
        print("val_macro_f1:", m_seq["val_macro_f1"])
        if m_seq.get("small_sample_train_equals_val"):
            print(
                "(метрики на очень малой выборке: train и val совпадают, "
                "цифры не для научного вывода, только проверка кода)"
            )
        print(m_seq["report"])
        save_torch_bundle(gru_bundle, os.path.join(args.out_dir, "gru_classifier"))
        save_metrics(m_seq, os.path.join(args.out_dir, "metrics_gru.json"))
        _save_predictions_csv(
            m_seq,
            list(bundle["meta"].get("filenames", [])),
            os.path.join(args.out_dir, "predictions_gru.csv"),
        )
        metrics_by_model["BiGRU seq"] = m_seq

    comparisons: list[dict[str, Any]] = []
    if not args.skip_comparison and n_classes >= 2:
        seeds = _parse_int_list(args.eval_seeds)
        seq_lens = _parse_int_list(args.compare_seq_lens)
        print("\n--- Повторная оценка устойчивости ---")
        for seed in seeds:
            _, m = train_sklearn_mlp_tabular(bundle["X_tab"], bundle["y"], random_state=seed)
            comparisons.append(
                {
                    "model": "MLP tabular",
                    "seq_len": int(bundle["X_seq"].shape[1]),
                    "seed": seed,
                    "accuracy": float(m["val_accuracy"]),
                    "macro_f1": float(m["val_macro_f1"]),
                }
            )
            print(
                f"MLP seed={seed}: "
                f"accuracy={m['val_accuracy']:.3f}, macro_f1={m['val_macro_f1']:.3f}"
            )

        if not args.no_gru:
            for seq_len in seq_lens:
                X_seq_cmp = _resample_sequence_batch(bundle["X_seq"], seq_len)
                for seed in seeds:
                    _, m = train_torch_gru_classifier(
                        X_seq_cmp,
                        bundle["y"],
                        num_classes=int(bundle["meta"]["num_classes"]),
                        epochs=args.comparison_epochs,
                        random_state=seed,
                    )
                    comparisons.append(
                        {
                            "model": "BiGRU seq",
                            "seq_len": seq_len,
                            "seed": seed,
                            "accuracy": float(m["val_accuracy"]),
                            "macro_f1": float(m["val_macro_f1"]),
                        }
                    )
                    print(
                        f"BiGRU seq_len={seq_len} seed={seed}: "
                        f"accuracy={m['val_accuracy']:.3f}, macro_f1={m['val_macro_f1']:.3f}"
                    )

        pd.DataFrame(comparisons).to_csv(
            os.path.join(args.out_dir, "comparison_results.csv"),
            index=False,
        )
        with open(os.path.join(args.out_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)

    _write_experiment_report(
        out_dir=args.out_dir,
        meta=bundle["meta"],
        y=bundle["y"],
        metrics_by_model=metrics_by_model,
        comparisons=comparisons,
    )

    print(f"\nГотово. Артефакты в папке: {args.out_dir}")


if __name__ == "__main__":
    main()
