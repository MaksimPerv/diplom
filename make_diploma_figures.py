#!/usr/bin/env python3
"""
Построение графиков для практической части диплома.

Скрипт использует уже полученные результаты экспериментов и не запускает YOLO.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("data/processed/diploma_figures")
ARTIFACT_DIR = Path("data/processed/front_walkover_prepare_test")
CLASS_NAMES = {
    0: "0 — плохо",
    1: "1 — удовлетворительно",
    2: "2 — хорошо",
    3: "3 — отлично",
}


def _save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_class_distribution() -> None:
    labels = pd.read_csv("data/labels.csv")
    counts = labels["class_id"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = [CLASS_NAMES[int(cls)] for cls in counts.index]
    bars = ax.bar(x, counts.values, color="#4C78A8")
    ax.set_title("Распределение обучающей выборки по классам качества")
    ax.set_xlabel("Класс качества")
    ax.set_ylabel("Количество видео")
    ax.bar_label(bars, padding=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "figure_3_1_class_distribution.png")


def plot_model_metrics() -> None:
    baseline = json.loads((ARTIFACT_DIR / "metrics_baseline.json").read_text(encoding="utf-8"))
    gru = json.loads((ARTIFACT_DIR / "metrics_gru.json").read_text(encoding="utf-8"))
    names = ["MLP", "BiGRU"]
    accuracy = [baseline["val_accuracy"], gru["val_accuracy"]]
    macro_f1 = [baseline["val_macro_f1"], gru["val_macro_f1"]]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#4C78A8")
    b2 = ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color="#F58518")
    ax.set_title("Сравнение качества MLP и BiGRU")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Значение метрики")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(b1, fmt="%.3f", padding=3)
    ax.bar_label(b2, fmt="%.3f", padding=3)
    _save(fig, "figure_3_2_model_metrics.png")


def plot_confusion_matrix() -> None:
    metrics = json.loads((ARTIFACT_DIR / "metrics_baseline.json").read_text(encoding="utf-8"))
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    labels = [str(x) for x in metrics["class_labels"]]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Матрица ошибок MLP-классификатора")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "figure_3_3_confusion_matrix_mlp.png")


def plot_new_video_probabilities() -> None:
    preds = pd.read_csv(ARTIFACT_DIR / "new_video_predictions.csv")
    # Для диплома берем три новых ролика, без старого RPReplay.
    preds = preds[preds["filename"].isin(["IMG_6023 2.MOV", "IMG_6023.MOV", "IMG_6024.MOV"])]
    prob_cols = ["prob_class_0", "prob_class_1", "prob_class_2", "prob_class_3"]
    probs = preds[prob_cols].to_numpy(dtype=float)

    x = np.arange(len(preds))
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(preds))
    colors = ["#E45756", "#F58518", "#4C78A8", "#54A24B"]
    for idx, col in enumerate(prob_cols):
        values = probs[:, idx]
        ax.bar(x, values, bottom=bottom, label=f"Класс {idx}", color=colors[idx])
        bottom += values
    ax.set_title("Вероятности классов для новых видеозаписей")
    ax.set_xticks(x, preds["filename"], rotation=15, ha="right")
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "figure_3_4_new_video_probabilities.png")


def plot_stability_comparison() -> None:
    df = pd.read_csv(ARTIFACT_DIR / "comparison_results.csv")
    grouped = (
        df.groupby(["model", "seq_len"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"))
        .sort_values(["model", "seq_len"])
    )
    grouped["variant"] = grouped["model"] + ", seq_len=" + grouped["seq_len"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(grouped["variant"], grouped["macro_f1_mean"], color="#72B7B2")
    ax.set_title("Устойчивая оценка моделей по среднему Macro F1")
    ax.set_ylabel("Macro F1 mean")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=18)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "figure_3_5_stability_macro_f1.png")


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    plot_class_distribution()
    plot_model_metrics()
    plot_confusion_matrix()
    plot_new_video_probabilities()
    plot_stability_comparison()


if __name__ == "__main__":
    main()
