# Shumodav — сравнение методов шумоподавления изображений

Курсовая работа. Приложение на Python, которое сравнивает методы шумоподавления: пространственные, частотные (FFT), вейвлетные и нейросетевые. Состоит из:

- **GUI на PyQt5**: загрузка картинки, добавление шума пяти типов, выбор метода и его параметров, спектры, таблица сравнения всех методов;
- **CLI-бенчмарка**, который подбирает параметры, прогоняет все методы на Set12 и строит итоговые таблицы и графики (`results/`);
- **кода обучения собственной CNN** (DnCNN-Lite) и ноутбука для Google Colab.

Методы: среднее, медианный, Гаусса, билатеральный, нелокальных средних (NLM), BM3D, ФНЧ в частотной области (идеальный / гауссов / Баттерворта), режекторный FFT-фильтр, вейвлеты (DWT/SWT, soft/hard, BayesShrink), готовые DnCNN и DRUNet, наша CNN.
Шумы: гауссов, «соль и перец», спекл, Пуассона, периодический.

Результаты и выводы — в [`docs/results.md`](docs/results.md), таблицы и графики — в [`results/`](results/).

## Установка

Нужен Python 3.11–3.13 и [poetry](https://python-poetry.org/).

```bash
git clone https://github.com/mkass420/shumodav.git
cd shumodav
poetry install                 # основные зависимости
poetry install -E bm3d         # (по желанию) метод BM3D
```

Нейросетевые методы (DnCNN, DRUNet, наша CNN) требуют PyTorch. CPU-сборку ставьте так (в poetry torch не прописан, а `poetry install -E neural` может поставить тяжёлую CUDA-версию):

```bash
poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Без `torch` и `bm3d` всё остальное работает: соответствующие методы просто не появляются в списках.

### Данные и веса

```bash
poetry run python scripts/get_data.py          # Train400, BSD68, Set12 -> data/
poetry run python scripts/download_weights.py  # веса KAIR (DnCNN, DRUNet, ~135 МБ) -> weights/
```

Веса нашей CNN (`weights/custom_cnn.pth`, 0.7 МБ) лежат в репозитории.

## Запуск GUI

```bash
poetry run python -m shumodav.gui.app     # или: poetry run python run.py
```

Слева: загрузка/сохранение, шум (тип, параметры, seed, режим «Без шума» для уже зашумлённых картинок), метод (список по семействам, параметры строятся автоматически, кнопка «Лучшие параметры» подставляет результаты бенчмарка), метрики PSNR/SSIM и время. Справа три вкладки: «Изображения» (оригинал / шумное / результат с общим зумом колесом), «Визуализация» (спектры и визуализация метода: маска фильтра, найденные пики, вейвлет-мозаика), «Сравнение» (кнопка «Прогнать все методы», сортируемая таблица, двойной клик выбирает метод).

Замечания: на Wayland при проблемах с Qt-плагином запускайте с `QT_QPA_PLATFORM=xcb`. Расчёты идут в отдельном процессе: медленный метод (BM3D, NLM, DRUNet) на большой картинке можно прервать кнопкой «Отмена расчёта», а новый запрос вытесняет расчёт, идущий дольше 2 с. Если процесс расчёта аварийно завершится (например, не хватит памяти), окно останется живым и покажет ошибку.

## Бенчмарк

```bash
poetry run python -m shumodav.bench.tune --quick     # быстрая проверка подбора параметров
poetry run python -m shumodav.bench.tune             # подбор параметров на BSD68 -> results/best_params.json
poetry run python -m shumodav.bench.run --quick      # быстрая проверка прогона
poetry run python -m shumodav.bench.run              # прогон на Set12 -> results/results.csv
poetry run python -m shumodav.bench.run --methods custom_cnn --noises gaussian   # только часть
poetry run python -m shumodav.bench.report           # таблицы и графики -> results/
```

Методология: `tune` берёт первые 4 картинки BSD68, `run` — все 12 картинок Set12 (серые, центральный кроп 256×256); шумная картинка генерируется один раз для каждой пары (картинка, шум, уровень), seed — `crc32` от имени, так что запуски воспроизводимы. `run` резюмируется: уже посчитанные строки (с теми же параметрами) пропускаются. Перед долгими запусками сначала прогоните `--quick`. Фоновый запуск:

```bash
nohup sh -c 'poetry run python -m shumodav.bench.tune > results/tune.log 2>&1 && poetry run python -m shumodav.bench.run > results/run.log 2>&1' &
```

В `results/`: `summary_psnr.md`, `summary_ssim.md`, `best_by_noise.md`, `heatmap_*.png`, `gain_psnr.png`, `speed_vs_quality.png`, `examples/<шум>_mid.png`, `train_curve.png` и `train_log.csv`.

## Обучение своей CNN

Сеть DnCNN-Lite (10 слоёв, 48 каналов, ≈168 тыс. параметров) учится на патчах Train400 со случайным шумом четырёх типов. Обучение рассчитано на GPU Google Colab: [`docs/TRAINING_COLAB.md`](docs/TRAINING_COLAB.md) и ноутбук [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

```bash
poetry run python -m shumodav.nn.train --smoke --out /tmp/smoke/smoke.pth   # проверка за ~20 с на CPU
poetry run python -m shumodav.nn.train --iters 3000 --batch 32 --device cpu \
    --ckpt-dir weights/ckpt --out weights/custom_cnn.pth --resume          # сокращённое обучение на CPU
```

## Структура проекта

```
src/shumodav/
  core/         типы (Image, ParamSpec), реестры, шумы, метрики, pipeline, ввод-вывод
  denoisers/    spatial.py, frequency.py, wavelet.py, bm3d_.py, neural.py
  nn/           kair/ (вендоринг KAIR, MIT), model.py, data.py, train.py
  bench/        tune.py, run.py, report.py
  gui/          app.py, main_window.py, widgets.py, param_panel.py
scripts/        get_data.py, download_weights.py
notebooks/      train_colab.ipynb
docs/           results.md, report_draft.md, slides.md, TRAINING_COLAB.md, plans/
results/        таблицы, графики, CSV
tests/
```

Контракты: изображение — `float32` в [0, 1], форма `HxW` или `HxWx3` (RGB); метод не изменяет вход и возвращает новый массив той же формы. Параметры метода описываются `ParamSpec`, GUI и бенчмарк строятся по реестру и ничего не знают о конкретных методах.

## Как добавить новый метод

Создайте класс в любом модуле из `src/shumodav/denoisers/` (или новом, добавив его в `denoisers/__init__.py`):

```python
import cv2

from shumodav.core.registry import Denoiser, register_denoiser
from shumodav.core.types import ParamSpec


@register_denoiser
class BoxBlur(Denoiser):
    name = "box_blur"          # уникальное имя метода
    label = "Усреднение 5×5"   # подпись в GUI
    family = "spatial"         # spatial | frequency | wavelet | neural
    params = (ParamSpec("ksize", "Размер ядра", "int", 5, 3, 15, 2, grid=(3, 5, 7)),)

    def denoise(self, img, ksize=5):
        return cv2.blur(img, (ksize, ksize))  # вход не менять, вернуть float32 [0, 1]
```

Метод сразу появится в GUI (с панелью параметров по `params`), в таблице сравнения и в бенчмарке; `grid` задаёт значения для подбора в `tune`. Необязательно: `supports_color = False` (тогда цветные картинки обрабатываются по каналам), `is_available()` (если нужны внешние зависимости), `visualize()` (картинка для вкладки «Визуализация»).

## Тесты

```bash
QT_QPA_PLATFORM=offscreen poetry run pytest -q
```

Часть тестов пропускается, если нет `torch`, данных или весов.

## Лицензии

Код сетей DnCNN и DRUNet (`src/shumodav/nn/kair/`) — из репозитория [KAIR](https://github.com/cszn/KAIR) (К. Чжан, MIT), лицензия лежит рядом. Тестовые данные Train400/BSD68/Set12 скачиваются из репозитория [DnCNN](https://github.com/cszn/DnCNN).
