# Автоматическая оценка переворота вперёд по анализу позы

Дипломный проект: система оценивает качество выполнения акробатического элемента **переворот вперёд** по видео.  
Используются **YOLO-pose** (ключевые точки COCO17), признаки техники и классификация качества по шкале **0–3**.

## Структура проекта

```text
diplom/
├── src/                    # модули: поза, признаки, обучение
├── data/
│   ├── labels.csv          # разметка: filename → class_id (0–3)
│   ├── raw_videos/         # обучающие видео 
│   ├── test_videos/        # новые видео для оценки 
│   └── processed/          # датасет, модели, метрики 
├── main.py                 # демо: извлечение позы и визуализация
├── train_score_model.py    # подготовка данных и обучение MLP / BiGRU
├── predict_video.py        # оценка нового видео
├── make_diploma_figures.py # графики для отчёта
└── requirements.txt
```

## Установка

```bash
cd diplom
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Подготовка данных

`class_id`: 0 — плохо, 1 — удовлетворительно, 2 — хорошо, 3 — отлично.

3. Подготовка датасета (долго: YOLO по всем видео):

```bash
python train_score_model.py --prepare-only
```

4. Обучение:

```bash
python train_score_model.py
```

Артефакты сохраняются в `data/processed/front_walkover_prepare_test/` (модель MLP, метрики, отчёт).

## Оценка нового видео

```bash
# положите ролик в data/test_videos/
python predict_video.py
# или один файл:
python predict_video.py --video data/test_videos/example.MOV
```

## Демо извлечения позы

```bash
python main.py --video path/to/video.mov --backend yolo
```

## OpenPose

```bash
export OPENPOSE_BIN=/path/to/openpose/build/examples/openpose/openpose.bin
python main.py --video video.mov --backend openpose
```

