# Shumodav v2 — шумоподавление изображений: сравнение методов

Курсовая работа. Приложение на Python для сравнения методов шумоподавления: пространственных, частотных (FFT), вейвлетных и нейросетевых. Есть GUI на PyQt5 и CLI-бенчмарк, который формирует итоговую таблицу эффективности.

**Срок — один день.** Поэтому приоритеты такие:
1. Работающий код важнее красивого кода.
2. Простое решение важнее универсального.
3. Ничего сверх описанного здесь не делать.

Работа идёт **блоками** (см. раздел «Блоки работ»). Пользователь пишет «выполни блок N». Тогда:
1. Прочитай этот файл целиком.
2. Сделай только блок N.
3. Проверь критерии «Готово, когда».
4. Отметь блок в чек-листе прогресса.
5. Сделай коммит.

---

## Чек-лист прогресса

Отмечай `[x]`, когда блок закончен. Если что-то пропущено, допиши короткую заметку.

- [x] Блок 0 — подготовка репозитория и данных (venv на Python 3.12; CLAUDE.md перенесён в корень; план — docs/plans/00-block0-setup.md)
- [x] Блок 1 — ядро: типы, реестр, шумы, метрики, pipeline, тесты (план — docs/plans/01-block1-core.md)
- [x] Блок 2 — классические методы: пространственные, FFT, notch, вейвлеты, NLM, BM3D (notch: порог считается от локального фона спектра, а не от глобальной медианы; nlm default h=20; план — docs/plans/02-block2-classic.md)
- [x] Блок 3 — нейросети: DnCNN/DRUNet (готовые веса), своя CNN, скрипт обучения, Colab-ноутбук (drunet: параметры auto_sigma + sigma; torch ставится отдельно, CPU-сборкой; план — docs/plans/03-block3-neural.md)
- [ ] Блок 4 — бенчмарк: tune / run / report
- [ ] Блок 5 — GUI
- [ ] Блок 6 — интеграция обученной CNN, финальный прогон и отчёт
- [ ] Блок 7 — документация: README, материалы для записки и презентации

---

## Контекст: что было раньше

- Старая версия лежит под git-тегом `v1-legacy`. Её не правим. Читать старые файлы так: `git show v1-legacy:src/denoise_app/core/wavelet_processor.py`.
- Что переносим из старой версии (логику, а не структуру):
  - `core/wavelet_processor.py` — DWT/SWT, soft/hard, обработка в YCbCr с усиленным порогом для цветовых каналов, bilateral-постобработка, мозаика коэффициентов;
  - `core/image_processor.py` — маски FFT-фильтров (ideal/gaussian/butterworth), паддинг через `BORDER_REFLECT`;
  - `ui/widgets.py` — `ZoomableImageView` (зум колесом, синхронизация) и `SpectrumViewer`.
- Что **не** переносим: GPU/OpenCL (не работало), `batch_processor`, `performance_profiler`, `DetailEnhancer`, смешивание, авто-оптимизацию. Последнюю заменяет подбор параметров из бенчмарка.

---

## Стек и окружение

- Python ≥ 3.11, менеджер зависимостей — **poetry**. Имя пакета `shumodav`, код в `src/shumodav/`.
- Основные зависимости: `numpy`, `scipy`, `opencv-python`, `pywavelets`, `scikit-image`, `matplotlib`, `pandas`, `tqdm`, `pyqt5`.
- Опциональные группы (extras):
  - `neural` = `torch` (на ноутбуке CPU-сборка);
  - `bm3d` = `bm3d`.
- Без опциональных зависимостей приложение должно работать: соответствующие методы просто не появляются в списке (`is_available() == False`).
- CPU-сборку torch ставить так: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Опиши это в README и не пытайся прописать в poetry.
- **GPU-кода в проекте нет.** GPU используется только в Colab для обучения.

---

## Структура

```
CLAUDE.md
README.md
pyproject.toml
docs/TRAINING_COLAB.md        # инструкция по обучению CNN в Colab (для человека)
notebooks/train_colab.ipynb   # ноутбук для Colab (Блок 3)
scripts/
  get_data.py                 # скачивает датасеты в data/
  download_weights.py         # скачивает веса KAIR в weights/
data/                         # gitignore
  train400/  bsd68/  set12/
weights/                      # gitignore, кроме weights/README.md
results/                      # CSV и PNG коммитим (они маленькие), кэш шумных картинок не коммитим
src/shumodav/
  __init__.py                 # пустой! без импортов подмодулей
  core/
    types.py                  # Image, ParamSpec
    registry.py               # реестры шумов и методов
    noise.py                  # 5 типов шума + пресеты уровней
    metrics.py                # PSNR, SSIM, таймер
    pipeline.py               # run_denoiser(), apply_noise(), поканальная обработка
    io.py                     # загрузка и сохранение, перевод в gray/float
  denoisers/
    __init__.py               # импортирует все модули, чтобы сработала регистрация
    spatial.py                # mean, median, gaussian, bilateral, nlm
    frequency.py              # fft_lowpass, fft_notch
    wavelet.py                # wavelet (DWT/SWT)
    bm3d_.py                  # bm3d (опционально)
    neural.py                 # dncnn, drunet, custom_cnn
  nn/
    kair/                     # вендоринг из KAIR (MIT): basicblock.py, network_dncnn.py, network_unet.py + LICENSE
    model.py                  # DnCNNLite — наша сеть
    data.py                   # датасет патчей с шумом на лету
    train.py                  # CLI обучения
  bench/
    tune.py  run.py  report.py
  gui/
    app.py  main_window.py  widgets.py  param_panel.py
tests/
  test_noise.py  test_metrics.py  test_denoisers_smoke.py
```

---

## Контракты (это главное: их не нарушать)

### Изображения
- `Image = np.ndarray`, **float32, диапазон [0, 1]**, форма `HxW` (серое) или `HxWx3` (**RGB**).
- Функции **не изменяют вход** и возвращают новый массив той же формы, float32, обрезанный в [0, 1].
- OpenCV читает BGR. Переводим в RGB в `core/io.py` и больше нигде этим не занимаемся.

### ParamSpec (`core/types.py`)
```python
@dataclass(frozen=True)
class ParamSpec:
    name: str                      # ключ в **kwargs
    label: str                     # подпись в GUI, по-русски
    kind: Literal["float", "int", "choice", "bool"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None      # для int: шаг (напр. 2 для нечётных ядер)
    choices: tuple | None = None
    log: bool = False              # логарифмический слайдер
    grid: tuple | None = None      # значения для подбора в bench/tune.py; None = не перебирать
    visible_if: dict | None = None # {"kind": "butterworth"}: показывать только при этом значении другого параметра
```

### Шум (`core/noise.py`)
```python
class NoiseModel(ABC):
    name: ClassVar[str]; label: ClassVar[str]
    params: ClassVar[tuple[ParamSpec, ...]]
    levels: ClassVar[dict[str, dict]]       # {"low": {...}, "mid": {...}, "high": {...}}
    def apply(self, img: Image, rng: np.random.Generator, **p) -> Image: ...
```
- `apply` должен работать на **любом** float-массиве (2D, 3D, патч 50×50). На этом построено обучение сети.
- Детерминизм: одинаковые `rng` seed + вход дают одинаковый результат.

| name | label | параметры | low / mid / high |
|---|---|---|---|
| `gaussian` | Гауссов | `sigma` в единицах 0..255 (внутри делим на 255) | 15 / 25 / 50 |
| `salt_pepper` | Соль и перец | `amount` (доля пикселей), `ratio`=0.5 | 0.05 / 0.10 / 0.20 |
| `speckle` | Спекл | `sigma`: `img + img * N(0, sigma)` | 0.10 / 0.20 / 0.30 |
| `poisson` | Пуассона | `peak`: `poisson(img * peak) / peak` | 100 / 30 / 10 |
| `periodic` | Периодический | `amplitude`, `freq` (циклов/пиксель), `angle` (градусы) | amp 0.10 / 0.20 / 0.30, freq 0.1, angle 45 |

### Метод (`core/registry.py`)
```python
class Denoiser(ABC):
    name: ClassVar[str]; label: ClassVar[str]
    family: ClassVar[Literal["spatial", "frequency", "wavelet", "neural"]]
    params: ClassVar[tuple[ParamSpec, ...]]
    supports_color: ClassVar[bool] = True     # False → pipeline обрабатывает каналы по отдельности
    def is_available(self) -> bool: return True
    def denoise(self, img: Image, **p) -> Image: ...
    def visualize(self, img: Image, **p) -> np.ndarray | None: return None  # спектр с маской, мозаика и т.п.
```
- Регистрация через декораторы `@register_noise` / `@register_denoiser`. Функции доступа: `get_noise(name)`, `get_denoiser(name)`, `list_denoisers(available_only=True)`.
- Параметры, которых нет в `**p`, берутся из `default`. Лишние ключи игнорируются.
- Тяжёлые модели (torch) загружаются **лениво** при первом `denoise` и кэшируются в экземпляре.
- **Методы ничего не знают об исходной чистой картинке.** Например, DRUNet получает `sigma` как параметр, а при `sigma="auto"` оценивает её через `skimage.restoration.estimate_sigma`.

### Pipeline (`core/pipeline.py`)
- `apply_noise(img, noise_name, params, seed) -> Image`
- `run_denoiser(img, denoiser_name, params) -> (result, seconds)`. Здесь же поканальная обработка для `supports_color=False`.
- `evaluate(clean, noisy, result) -> dict(psnr_noisy, psnr, ssim_noisy, ssim)`

### Метрики (`core/metrics.py`)
- PSNR и SSIM из `skimage.metrics`, `data_range=1.0`. Для цветных картинок `channel_axis=-1`.
- Время — `time.perf_counter()`. Для нейросетей перед замером делается прогрев (один прогон).

---

## Список методов

| name | family | ключевые параметры (grid для подбора) | заметки |
|---|---|---|---|
| `mean` | spatial | `ksize` 3..15 нечёт (3,5,7) | `cv2.blur` |
| `median` | spatial | `ksize` (3,5,7) | `cv2.medianBlur` требует uint8 или ksize ≤ 5 для float32: при ksize > 5 переводить в uint8 |
| `gaussian` | spatial | `sigma` (0.5,1,1.5,2,3) | `cv2.GaussianBlur`, ksize = 0 (авто) |
| `bilateral` | spatial | `d` (5,9), `sigma_color` (0.05,0.1,0.2), `sigma_space` (3,5) | `cv2.bilateralFilter` на float32 |
| `nlm` | spatial | `h` (5,10,15,20,30) в единицах 0..255 | `cv2.fastNlMeansDenoising` / `...Colored`, работает в uint8 |
| `bm3d` | spatial | `sigma` (10,15,25,35,50) в единицах 0..255 | пакет `bm3d`, опциональный |
| `fft_lowpass` | frequency | `kind` ideal/gaussian/butterworth, `d0` (0.05..0.5), `order` (1,2,4; visible_if butterworth), `padding` 0.1 | маска из v1; `visualize` → лог-спектр с наложенной маской |
| `fft_notch` | frequency | `threshold` (k·медиана), `radius` (2,4,8), `exclude_center` | ищет пики в лог-спектре вне центральной зоны и гасит их гауссовыми «дырками» (симметрично). `visualize` → спектр с отмеченными пиками |
| `wavelet` | wavelet | `wavelet` (db1,db2,db4,sym4,coif1,bior1.3; grid db4,sym4), `level` (1..4; grid 2,3), `threshold` (0.02..0.3), `mode` soft/hard, `swt` bool (grid True), `threshold_rule` manual/bayes, `chroma_strength`, `post_smooth` | перенести из v1. `bayes` = BayesShrink по уровням (оценка σ по MAD на HH1); `visualize` → мозаика |
| `dncnn` | neural | — | KAIR `dncnn_gray_blind.pth` (nb=20, act_mode='R'), для цвета `dncnn_color_blind.pth`; `supports_color=True` (выбор весов по числу каналов) |
| `drunet` | neural | `sigma` "auto" или 5..75 | KAIR `drunet_gray.pth`: `UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False)`. На вход подаётся картинка + карта шума `sigma/255`. Паддинг до кратного 8. `supports_color=False` |
| `custom_cnn` | neural | — | наша `DnCNNLite` из `weights/custom_cnn.pth`. `is_available` = есть файл весов. `supports_color=False` |

Все нейросетевые методы работают на CPU через `torch.no_grad()`.

---

## Данные

`scripts/get_data.py`:
1. Делает `git clone --depth 1 https://github.com/cszn/DnCNN` во временную папку.
2. Копирует:
   - `TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400` → `data/train400` (400 серых 180×180);
   - `testsets/BSD68` → `data/bsd68`;
   - `testsets/Set12` → `data/set12`.
3. Удаляет временную папку. Если папки уже есть, ничего не делает.

Роли датасетов (важно для честной методологии; так и пишем в записке):
- **train400** — обучение своей CNN;
- **bsd68** — валидация CNN и подбор параметров классических методов (`tune` берёт первые 4 картинки);
- **set12** — финальное тестирование в бенчмарке. Ни на чём не обучались и не подбирались.

`scripts/download_weights.py` скачивает с `https://github.com/cszn/KAIR/releases/download/v1.0/{name}` файлы `dncnn_gray_blind.pth`, `dncnn_color_blind.pth`, `drunet_gray.pth` (~130 МБ) в `weights/`.

---

## Наша CNN (подробно — в `docs/TRAINING_COLAB.md`)

- `DnCNNLite(depth=10, channels=48)`: Conv+ReLU → (Conv+BN+ReLU)×8 → Conv. Residual learning: `forward(x) = x - net(x)`. Вход и выход — 1 канал.
- `nn/data.py`: `PatchDataset` берёт случайный патч 50×50 из train400, случайный поворот или отражение (8 вариантов), затем **случайный тип шума из {gaussian, salt_pepper, speckle, poisson}** со случайным уровнем из диапазона. Шум накладывается через `core.noise`, тот же код, что в бенчмарке. Диапазоны:
  - gaussian sigma U[5, 55];
  - salt_pepper amount U[0.02, 0.25];
  - speckle sigma U[0.05, 0.35];
  - poisson peak лог-равномерно [5, 150].
- `nn/train.py`, CLI (argparse):
  - аргументы: `--data data/train400 --val data/bsd68 --iters 15000 --batch 128 --lr 1e-3 --out weights/custom_cnn.pth --ckpt-dir <dir> --resume --smoke --device auto`;
  - Adam, MSE-loss, MultiStepLR (×0.1 на 60% и 85% итераций), AMP на CUDA;
  - каждые 1000 итераций: валидация на первых 10 картинках bsd68 для **каждого** типа шума на уровне mid, печать PSNR. Сохраняется чекпойнт (модель + оптимизатор + scheduler + iter) и лучшая модель по среднему PSNR;
  - в конце сохраняются `train_log.csv` (iter, loss, val_psnr по шумам) и `train_curve.png` рядом с весами;
  - `--smoke`: 50 итераций, батч 8, валидация на 2 картинках, всё на CPU. Должно отработать меньше чем за минуту;
  - `--resume` продолжает обучение с последнего чекпойнта в `--ckpt-dir`. Если чекпойнта нет, начинает с нуля без ошибки;
  - `train_log.csv` и `train_curve.png` дописываются при каждой валидации, а не только в конце, чтобы после обрыва сессии они не терялись.
- `shumodav.core` и `shumodav.nn` **не должны импортировать** PyQt5, pywt, bm3d, matplotlib на уровне модуля (matplotlib — только внутри функции рисования графика). Это нужно, чтобы в Colab хватило `pip install -e . --no-deps`.

---

## Бенчмарк

- `python -m shumodav.bench.tune [--quick]` — перебор по `grid` всех методов с непустой сеткой, для каждого шума × уровня, на 4 картинках bsd68 (центр-кроп 256). Максимизируем средний PSNR. Результат: `results/best_params.json` (`{noise: {level: {method: params}}}`). Перебор — декартово произведение, но не больше 60 комбинаций на метод; если больше, случайная выборка 60 с фиксированным seed.
- `python -m shumodav.bench.run [--methods ...] [--noises ...] [--quick]`:
  - картинки set12, центр-кроп 256, серые;
  - шумная картинка генерируется **один раз** на (картинка, шум, уровень) с seed = hash(...). Все методы получают один и тот же вход;
  - параметры берутся из `best_params.json`, а если записи нет — из `default`;
  - строки дописываются в `results/results.csv`: `image, noise, level, method, family, params_json, psnr_noisy, psnr, ssim_noisy, ssim, seconds`;
  - **резюмирование:** строки, которые уже есть в CSV, пропускаются. Прогресс — через `tqdm`;
  - `--quick`: 2 картинки, только уровень mid.
- `python -m shumodav.bench.report`, результаты в `results/`:
  - `summary_psnr.md`, `summary_ssim.md` — таблицы «метод × (шум, уровень)», среднее по картинкам, лучший результат в каждом столбце выделен **жирным**;
  - `heatmap_psnr.png`, `heatmap_ssim.png` — то же самое в виде тепловой карты, аннотированные значения;
  - `gain_psnr.png` — выигрыш PSNR относительно шумной картинки;
  - `speed_vs_quality.png` — scatter: X = время (лог-шкала), Y = средний PSNR, подписи методов, цвет по family;
  - `best_by_noise.md` — лучший метод и лучший «быстрый» (< 0.05 с) для каждого шума;
  - `examples/<noise>_mid.png` — сетка кропов 96×96 одной картинки (`05.png`): чистая, шумная и каждый метод с PSNR в подписи.

---

## GUI

- `python -m shumodav.gui.app` (и `run.py` в корне для совместимости).
- Слева панель:
  - «Загрузить» / «Сохранить результат»;
  - чекбокс «Работать в оттенках серого»;
  - группа «Шум»: combo из реестра, авто-параметры, seed, чекбокс «Без шума (картинка уже шумная)»;
  - группа «Метод»: combo, сгруппированный по family, авто-параметры;
  - кнопка «Лучшие параметры» — подставляет значения из `best_params.json` для текущего шума и уровня (ближайший уровень по параметрам). Неактивна, если файла нет;
  - метрики: PSNR/SSIM шумной и результата, время. Если включено «Без шума», метрики показываются как «—».
- `param_panel.py`: `ParamPanel(specs)` строит виджеты по `ParamSpec`:
  - float → слайдер + `QDoubleSpinBox`;
  - int → `QSpinBox` с шагом;
  - choice → `QComboBox`;
  - bool → `QCheckBox`.
  
  Учитывает `visible_if`. Сигнал `changed(dict)`.
- Справа вкладки:
  1. «Изображения» — оригинал / шумное / результат, синхронный зум (перенести `ZoomableImageView`).
  2. «Визуализация» — лог-спектры трёх картинок и `denoiser.visualize()`, если метод его возвращает.
  3. «Сравнение» — кнопка «Прогнать все методы»: в фоне запускает все доступные методы с лучшими (или дефолтными) параметрами и показывает сортируемую `QTableWidget` с колонками PSNR/SSIM/время. По двойному клику метод выбирается.
- Обработка в `QThread` с debounce 200 мс. У каждого запроса есть id, устаревшие результаты отбрасываются.
- Ошибки метода показываются в статус-строке, а не роняют приложение.

---

## Правила работы

- После каждого блока: `pytest -q` зелёный, затем коммит `блок N: <кратко>`.
- Не добавляй зависимости сверх списка без явной просьбы пользователя.
- Не скачивай ничего, кроме описанного в «Данные» и весов KAIR.
- Не делай GPU-ускорение, не возвращай код из v1, который помечен как «не переносим».
- Строки для пользователя (GUI, подписи графиков) — по-русски. Имена в коде — по-английски. Комментарии — коротко, по-русски.
- Если что-то не получается за разумное время (например, `bm3d` не ставится), сделай метод недоступным через `is_available()`, допиши заметку в чек-лист и иди дальше. **Не застревай.**
- Перед длинными операциями (tune/run на полном наборе) сначала прогони `--quick`.

---

## Блоки работ

### Блок 0 — подготовка (≈15 мин)
1. Проверь, что тег `v1-legacy` существует. Если нет, поставь его на текущий HEAD ветки `main`. Работаем в ветке `v2`.
2. Удали старый `src/denoise_app/` из рабочей копии (в истории он остаётся). Заведи структуру из раздела «Структура», `pyproject.toml` с зависимостями и extras, `.gitignore` (`data/`, `weights/*.pth`, `results/cache/`, `__pycache__`, `*.log`).
3. Напиши `scripts/get_data.py` и запусти его.
4. Напиши `scripts/download_weights.py`, но **не запускай** (это блок 3).

**Готово, когда:** `poetry install` проходит; в `data/` лежат train400 (400 файлов), bsd68 (68), set12 (12).

### Блок 1 — ядро (≈1 ч)
`core/types.py`, `registry.py`, `noise.py` (все 5 шумов и `levels`), `metrics.py`, `pipeline.py`, `io.py`. Тесты:
- каждый шум детерминирован при одном seed и сохраняет форму и dtype на 2D, 3D и патче 50×50;
- у gaussian sigma=25 стандартное отклонение разности ≈ 25/255 (±10%) на серой картинке 0.5;
- PSNR(x, x) = inf; PSNR при известном MSE совпадает с формулой;
- заглушка-метод `identity` (только для тестов, в `tests/`) проходит через `run_denoiser` и `evaluate`.

**Готово, когда:** `pytest -q` зелёный.

### Блок 2 — классические методы (≈1.5 ч)
Все методы из таблицы с family spatial / frequency / wavelet и `bm3d`. Вейвлеты и маски FFT переносятся из `v1-legacy`. Smoke-тест: каждый доступный метод на сером 64×64 и цветном 64×64 возвращает ту же форму, float32, значения в [0, 1], и работает без исключений с параметрами по умолчанию. Дополнительно две проверки по смыслу:
- `fft_lowpass` не меняет среднюю яркость больше чем на 1%;
- `fft_notch` на картинке с `periodic`-шумом даёт PSNR выше шумной минимум на 5 дБ.

**Готово, когда:** тесты зелёные, а `list_denoisers()` печатает все методы этого блока.

### Блок 3 — нейросети (≈1.5 ч)
1. Вендоринг KAIR: `basicblock.py`, `network_dncnn.py`, `network_unet.py` в `nn/kair/` вместе с LICENSE (MIT, Kai Zhang). Импорты внутри поправить на относительные.
2. Запусти `scripts/download_weights.py`.
3. `denoisers/neural.py`: `dncnn`, `drunet` (с `sigma="auto"`), `custom_cnn`.
4. `nn/model.py`, `nn/data.py`, `nn/train.py` — строго по разделу «Наша CNN».
5. `notebooks/train_colab.ipynb` — ячейки из раздела «Ячейки ноутбука» в `docs/TRAINING_COLAB.md`, по порядку, с markdown-заголовками между ними.
6. Проверки:
   - `python -m shumodav.nn.train --smoke --out /tmp/smoke.pth` отрабатывает;
   - dncnn и drunet на `set12/05.png` с gaussian σ=25 дают PSNR > 29 дБ;
   - smoke-тест методов зелёный (custom_cnn пропускается, если весов нет).
7. Коммит и **`git push`**: ноутбук в Colab клонирует репозиторий с GitHub. Ветку `v2` пушим. В ноутбуке используется `-b v2`.

**Готово, когда:** всё из п.6 выполнено, push прошёл. Скажи пользователю: «Можно запускать обучение в Colab по docs/TRAINING_COLAB.md».

### Блок 4 — бенчмарк (≈1.5 ч)
`bench/tune.py`, `bench/run.py`, `bench/report.py` по разделу «Бенчмарк». Порядок:
1. `tune --quick` → `run --quick` → `report`. Проверь, что графики читаемые: открой PNG и посмотри на них.
2. Затем дай пользователю команды полного прогона в фоне (сам не жди их окончания):
   ```
   nohup python -m shumodav.bench.tune > results/tune.log 2>&1 && \
   nohup python -m shumodav.bench.run > results/run.log 2>&1 &
   ```

**Готово, когда:** quick-версии всех трёх команд отработали, в `results/` есть все файлы из списка.

### Блок 5 — GUI (≈2 ч)
Всё из раздела «GUI». `ZoomableImageView` и `SpectrumViewer` переносятся из `v1-legacy:src/denoise_app/ui/widgets.py`. Проверка без экрана: `QT_QPA_PLATFORM=offscreen` — окно создаётся, программно загружается `set12/05.png`, прогоняются все методы, нет исключений.

**Готово, когда:** offscreen-проверка проходит. Попроси пользователя запустить приложение у себя и прокликать его.

### Блок 6 — интеграция своей CNN (≈30 мин, после Colab)
1. Пользователь кладёт `custom_cnn.pth`, `train_log.csv`, `train_curve.png` в `weights/`.
2. Проверь загрузку весов и smoke-тест.
3. `python -m shumodav.bench.run --methods custom_cnn` (дописывает в CSV) → `report`.
4. Скопируй `train_curve.png` в `results/`.

**Готово, когда:** custom_cnn есть во всех таблицах и графиках.

### Блок 7 — документация (≈1 ч)
1. `README.md`: установка (включая CPU-torch), получение данных и весов, запуск GUI, бенчмарка и обучения, структура проекта, как добавить новый метод (пример на 15 строк).
2. `docs/results.md` — **выводы** по фактическим цифрам из `results/`:
   - лучший метод для каждого шума;
   - где классика конкурирует с нейросетями;
   - готовые DnCNN/DRUNet на негауссовом шуме против нашей CNN;
   - notch на периодическом шуме;
   - качество против скорости.
   
   Только факты из таблиц, никаких выдуманных чисел.
3. `docs/report_draft.md` — черновик разделов пояснительной записки:
   - описание архитектуры приложения;
   - методы (кратко, с формулами: маска Баттерворта, soft-threshold, BayesShrink, residual learning);
   - методика эксперимента (датасеты и их роли, уровни шума, метрики, подбор параметров);
   - результаты (ссылки на таблицы и графики);
   - выводы.
4. `docs/slides.md` — тексты для обновлённых слайдов. Обязательно поправить: на старом слайде «Методы фильтрации» вместо «идеальный ФВЧ» должно быть «идеальный ФНЧ» (для шумоподавления пропускаем низкие частоты). Слайды «план на год» заменить на «что сделано» и результаты.
