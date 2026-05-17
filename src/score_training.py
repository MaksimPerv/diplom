"""
Обучение baseline (sklearn) и последовательностной модели (PyTorch GRU) для оценки класса техники.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Возвращает (Xtr, Xva, ytr, yva, val_idx, same_as_train).
    Если same_as_train=True, валидация совпадает с обучением (только для очень малых N).
    """
    n = len(y)
    if n <= 4:
        return X, X, y, y, np.arange(n), True

    values, counts = np.unique(y, return_counts=True)
    stratify = y if len(values) >= 2 and np.min(counts) >= 2 else None
    idx = np.arange(n)
    Xtr, Xva, ytr, yva, _, idx_va = train_test_split(
        X,
        y,
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return Xtr, Xva, ytr, yva, idx_va, False


def _oversample_minority_classes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Выравнивает классы в train-части простым повторением редких классов."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) == np.max(counts):
        return X, y

    target = int(np.max(counts))
    indices: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        extra = rng.choice(cls_idx, size=target - len(cls_idx), replace=True)
        indices.append(np.concatenate([cls_idx, extra]))

    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def _class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = float(np.sum(counts)) / (float(np.sum(nonzero)) * counts[nonzero])
    return weights


def _metrics_from_predictions(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    class_labels: list[int],
    val_indices: np.ndarray,
    small_n: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    metrics: dict[str, Any] = {
        "val_accuracy": float(accuracy_score(yt, yp)),
        "val_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "report": classification_report(yt, yp, labels=class_labels, zero_division=0),
        "report_dict": classification_report(
            yt,
            yp,
            labels=class_labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(yt, yp, labels=class_labels).tolist(),
        "class_labels": class_labels,
        "y_true": yt.tolist(),
        "y_pred": yp.tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "small_sample_train_equals_val": small_n,
    }
    if extra:
        metrics.update(extra)
    return metrics


def train_sklearn_mlp_tabular(
    X_tab: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...] | None = None,
    max_iter: int = 500,
    random_state: int = 42,
    balance_classes: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    Xtr, Xva, ytr, yva, idx_va, small_n = _split_train_val(
        X_tab,
        y,
        test_size=max(0.2, 1.0 / max(len(y), 4)),
        random_state=random_state,
    )
    if balance_classes and not small_n:
        Xtr, ytr = _oversample_minority_classes(Xtr, ytr, random_state=random_state)

    if hidden_layer_sizes is None:
        hidden_layer_sizes = (32,) if len(Xtr) < 10 else (64, 32)
    early_stop = len(Xtr) >= 10

    pipe: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=early_stop,
                    validation_fraction=0.15 if early_stop else 0.0,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xva)
    metrics = _metrics_from_predictions(
        yva,
        y_pred,
        class_labels=sorted(int(v) for v in np.unique(y)),
        val_indices=idx_va,
        small_n=small_n,
        extra={
            "random_state": random_state,
            "balance_classes": balance_classes,
            "hidden_layer_sizes": list(hidden_layer_sizes),
        },
    )
    return pipe, metrics


def train_torch_gru_classifier(
    X_seq: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    hidden_size: int = 64,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str | None = None,
    random_state: int = 42,
    use_class_weights: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(random_state)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_state)

    Xtr, Xva, ytr, yva, idx_va, small_n = _split_train_val(
        X_seq,
        y,
        test_size=max(0.2, 1.0 / max(len(y), 4)),
        random_state=random_state,
    )

    class SeqGRU(nn.Module):
        def __init__(self, in_dim: int, hidden: int, n_cls: int) -> None:
            super().__init__()
            self.gru = nn.GRU(
                in_dim,
                hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.fc = nn.Linear(hidden * 2, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, h = self.gru(x)
            last = torch.cat([h[-2], h[-1]], dim=1)
            return self.fc(last)

    in_dim = X_seq.shape[-1]
    model = SeqGRU(in_dim, hidden_size, num_classes).to(device)

    ds_tr = TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(ytr).long(),
    )
    ds_va = TensorDataset(
        torch.from_numpy(Xva).float(),
        torch.from_numpy(yva).long(),
    )
    dl_tr = DataLoader(ds_tr, batch_size=min(batch_size, len(ds_tr)), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=min(batch_size, len(ds_va)), shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if use_class_weights:
        weights = torch.from_numpy(_class_weights(ytr, num_classes)).float().to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_state: dict[str, Any] | None = None
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        all_pred: list[int] = []
        all_true: list[int] = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                all_pred.extend(pred)
                all_true.extend(yb.numpy().tolist())
        f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())

    metrics = _metrics_from_predictions(
        all_true,
        all_pred,
        class_labels=list(range(num_classes)),
        val_indices=idx_va,
        small_n=small_n,
        extra={
            "epochs": epochs,
            "device": device,
            "random_state": random_state,
            "use_class_weights": use_class_weights,
        },
    )
    meta = {
        "model": "SeqGRU",
        "in_dim": in_dim,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "seq_len": int(X_seq.shape[1]),
    }
    bundle = {"model": model, "meta": meta, "device": device}
    return bundle, metrics


def save_sklearn_pipeline(pipe: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(pipe, path)


def save_torch_bundle(bundle: dict[str, Any], path_prefix: str) -> None:
    import torch

    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    torch.save(bundle["model"].state_dict(), path_prefix + "_state.pt")
    meta = dict(bundle["meta"])
    meta["device_saved"] = bundle["device"]
    with open(path_prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_metrics(metrics: dict[str, Any], path: str) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value

    out = json_safe(metrics)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
