"""
Шкала качества исполнения элемента (классификация для ML-части диплома).

class_id — целое 0..3 (чем выше, тем лучше техника по смыслу шкалы).
"""

from __future__ import annotations

QUALITY_CLASS_NAMES_RU: dict[int, str] = {
    0: "Плохо",
    1: "Удовлетворительно",
    2: "Хорошо",
    3: "Отлично",
}

NUM_QUALITY_CLASSES: int = 4


def validate_class_ids(ids: list[int]) -> None:
    bad = [i for i in ids if i not in QUALITY_CLASS_NAMES_RU]
    if bad:
        raise ValueError(
            f"class_id должен быть в диапазоне 0..{NUM_QUALITY_CLASSES - 1}, получено: {bad}"
        )
