# План: Блок 3 — нейросети (DnCNN/DRUNet, своя CNN, обучение, Colab-ноутбук)

## Context
Блоки 0–2 готовы (ветка `v2`, 80 тестов зелёные, 9 классических методов). Блок 3 по `CLAUDE.md` добавляет нейросетевые методы `dncnn`, `drunet`, `custom_cnn`, код обучения своей сети (`nn/`) и ноутбук для Colab, затем пушит `v2` на GitHub (Colab клонирует репозиторий). Ориентир по обучению — `docs/TRAINING_COLAB.md` (разделы 2 и 5).

Что проверено заранее:
- `torch` в окружении **нет**; GPU нет (всё на CPU). Ставить по спеке: `poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu` (в poetry не прописываем — extra `neural` там уже есть, но повторный `poetry install` может подтянуть CUDA-сборку; после установки torch не запускать `poetry install -E neural`).
- Веса KAIR доступны: `dncnn_gray_blind.pth` (2.6 МБ), `dncnn_color_blind.pth` (2.7 МБ), `drunet_gray.pth` (130 МБ).
- Исходники KAIR (`models/basicblock.py`, `network_dncnn.py`, `network_unet.py`, `LICENSE` — MIT, Kai Zhang) берём с `raw.githubusercontent.com/cszn/KAIR/master/`. В двух сетях `import models.basicblock as B` → заменить на `from . import basicblock as B`. У `UNetRes.forward` паддинг до кратного 8 **закомментирован** → делаем его сами в денойзере; выход `UNetRes` — уже чистая картинка (без вычитания).
- Репозиторий `mkass420/shumodav` публичный, на GitHub пока только `main`; `gh` авторизован.

Первым действием при исполнении: скопировать этот план в `docs/plans/03-block3-neural.md`.

## Шаги

### 1. Окружение и веса
- Установить CPU-torch (команда выше), проверить `import torch`.
- Написанный в Блоке 0 `scripts/download_weights.py` запустить (по спеке в этом блоке). Файлы в `weights/` не коммитятся (`weights/*.pth` в `.gitignore`).

### 2. Вендоринг KAIR — `src/shumodav/nn/kair/`
`basicblock.py`, `network_dncnn.py`, `network_unet.py`, `LICENSE`; заголовок с источником. Правка только относительных импортов.

### 3. `src/shumodav/nn/model.py`
`DnCNNLite(depth=10, channels=48)`: Conv(1→48)+ReLU → (Conv+BN+ReLU)×8 → Conv(48→1); `forward(x) = x - self.net(x)`. Ожидаемо ≈168–170 тыс. параметров.

### 4. `src/shumodav/denoisers/neural.py` (+ добавить в `denoisers/__init__.py`)
- `torch` импортируется **лениво** внутри методов; модель грузится при первом `denoise` и кэшируется в экземпляре (отдельно для gray/color у `dncnn`). `torch.no_grad()`, `torch.set_grad_enabled` не трогаем глобально.
- Путь к весам: `WEIGHTS_DIR = <корень репо>/weights` (`Path(__file__).parents[3]`), можно переопределить env `SHUMODAV_WEIGHTS`.
- `dncnn`: без параметров, `supports_color=True`; по числу каналов входа: gray → `DnCNN(1,1,nc=64,nb=20,act_mode='R')` + `dncnn_gray_blind.pth`, color → `DnCNN(3,3,...)` + `dncnn_color_blind.pth`. `is_available` = torch импортируется и нужный файл весов есть.
- `drunet`: `sigma` — параметр (choice/float: `"auto"` или 5..75; в `ParamSpec` kind=`"float"` + отдельный bool-подобный выбор не нужен — делаю `kind="choice"`-несовместимости не вводить: `sigma` float 5..75 с default `25`, а `"auto"` принимается как значение в `**p`; в GUI позже добавлю чекбокс; **решение при реализации**: два параметра — `auto_sigma` (bool, default True) и `sigma`, `visible_if={"auto_sigma": False}`; `denoise` также принимает `sigma="auto"` для совместимости со спекой). Авто-оценка: `skimage.restoration.estimate_sigma(img)·255`, обрезка в [5, 75]. `UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False)`; вход = конкатенация картинки и карты `sigma/255`; reflect-паддинг до кратного 8, затем обрезка. `supports_color=False`.
- `custom_cnn`: `DnCNNLite` из `weights/custom_cnn.pth` (plain `state_dict`), `is_available` = файл есть, `supports_color=False`.
- Общий хелпер: `_to_tensor(img)` / `_to_numpy(t)`; прогон `net(x)` целиком (полносвёрточные сети, любой размер).

### 5. Обучение — `src/shumodav/nn/`
**`data.py`** — `PatchDataset(images, length, patch=50, seed)` (`torch.utils.data.Dataset`; `images` — список 2D float32). `__getitem__(i)`: `rng = default_rng((seed, i))`; случайный кроп 50×50, одно из 8 преобразований (поворот k·90° × флип), случайный тип шума из {gaussian, salt_pepper, speckle, poisson} через `core.noise.get_noise(...).apply` с уровнем из диапазонов спеки (gaussian σ U[5,55]; s&p amount U[0.02,0.25]; speckle σ U[0.05,0.35]; poisson peak log-uniform [5,150]). Возвращает `(noisy, clean)` тензоры 1×50×50. Загрузка картинок — `core.io.load_image(gray=True)`.

**`train.py`** — CLI (argparse) строго по спеке: `--data --val --iters 15000 --batch 128 --lr 1e-3 --out --ckpt-dir --resume --smoke --device auto` (+ `--seed 0`).
- Adam, MSE, `MultiStepLR(milestones=[0.6·iters, 0.85·iters], gamma=0.1)`, шаг планировщика каждую итерацию; AMP (`autocast`+`GradScaler`) только на CUDA; `DataLoader` с 2 воркерами на CUDA, 0 на CPU.
- Каждые 1000 итераций: валидация на первых 10 картинках BSD68 для каждого из 4 шумов на уровне mid (фиксированный seed на пару картинка×шум, чтобы значения были сравнимы), печать строки `iter N | loss L | val PSNR: gaussian … | salt_pepper … | speckle … | poisson …`; сохранение чекпойнта `ckpt_last.pth` (модель, оптимизатор, scheduler, scaler, iter, best) в `--ckpt-dir` и лучшей модели (по среднему PSNR) как plain `state_dict` в `--out`.
- При каждой валидации перезаписываются `train_log.csv` (iter, loss, val_psnr_<noise>…, val_psnr_mean) и `train_curve.png` (matplotlib импортируется **внутри функции**, backend Agg) — в каталоге `--out`.
- `--resume`: при наличии `ckpt_last.pth` продолжает (включая историю CSV), иначе молча стартует с нуля. В конце — валидация на последней итерации, печать времени обучения.
- `--smoke`: 50 итераций, батч 8, 2 картинки валидации, CPU, 0 воркеров, валидация каждые 25 итераций (чтобы код валидации/сохранения реально отработал), <1 мин.
- `core` и `nn` не импортируют PyQt5, pywt, bm3d, matplotlib на уровне модуля.

### 6. Ноутбук — `notebooks/train_colab.ipynb`
Собрать скриптом на `json` (nbformat v4, `metadata.accelerator = "GPU"`): 8 ячеек из раздела 5 `docs/TRAINING_COLAB.md` по порядку, перед каждой markdown-заголовок. Единственное отличие от документа: в ячейке 3 `pip install -q -e . --no-deps --ignore-requires-python` (страховка от `python <3.14` в `pyproject.toml`, если Colab обновит Python). Клонирование — `-b v2`.

### 7. Тесты
- `tests/test_nn.py` (с `pytest.importorskip("torch")`): `DnCNNLite` — форма на 1×1×64×64, число параметров 160–180 тыс.; `PatchDataset` на синтетических картинках — форма, диапазон [0,1], детерминизм по индексу; smoke обучения через `train.main(["--smoke", ..., "--out", tmp])` (skip, если нет `data/`) — создаёт `.pth`, `train_log.csv`, `train_curve.png`; проверка `--resume` (второй запуск не падает).
- `tests/test_neural_quality.py` (skip без torch/весов/данных): `dncnn` и `drunet` на `data/set12/05.png` с gaussian σ=25 (seed фиксирован) дают PSNR > 29 дБ.
- Существующий `test_denoisers_smoke.py` автоматически подхватит доступные нейро-методы (по `list_denoisers()`); `custom_cnn` пропускается, пока нет весов.

## Завершение блока
1. `poetry run python -m shumodav.nn.train --smoke --out /tmp/smoke.pth` отрабатывает (<1 мин); `poetry run pytest -q` зелёный (включая PSNR > 29 для dncnn/drunet).
2. Отметить `[x] Блок 3` в чек-листе `CLAUDE.md`, коммит `блок 3: нейросети` (+ `Co-Authored-By`).
3. `git push -u origin v2` (ветки на GitHub ещё нет; репозиторий публичный, `gh` авторизован). Веса и данные в коммит не попадают.
4. Сообщить пользователю: «Можно запускать обучение в Colab по docs/TRAINING_COLAB.md».

## Проверка
- Smoke-обучение, pytest (выше).
- Ручной прогон: `dncnn`, `drunet` (auto и sigma=25) на `set12/05.png` при gaussian σ=25 и на salt_pepper/speckle — вывести PSNR и время (ожидаю ≈29–30+ дБ на гауссе).
- После push: `gh api repos/mkass420/shumodav/contents/notebooks/train_colab.ipynb?ref=v2` возвращает файл; ссылка на ноутбук из `TRAINING_COLAB.md` рабочая.
