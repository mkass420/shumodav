# План: Блок 0 — подготовка репозитория и данных

## Context
По `docs/CLAUDE.md` проект Shumodav v2 переписывается с нуля в пакет `shumodav`. Первый этап — Блок 0: тег на старую версию, ветка `v2`, чистая структура, `pyproject.toml`, `.gitignore`, скрипты данных/весов, скачанные датасеты.

Текущее состояние:
- тега `v1-legacy` нет; `main` == `develop` == `13cf2a8`; remote `origin` = github.com/mkass420/shumodav;
- в develop незакоммиченный рефакторинг `src/denoise_app` — пользователь решил **выбросить**;
- `.gitignore` игнорирует `*.png/*.jpg/...` — это мешает коммитить `results/*.png` (спека требует);
- системный Python 3.14, есть и 3.12 → для venv берём **3.12** (надёжнее для torch/bm3d/pyqt5);
- `CLAUDE.md` лежит в `docs/`, а по структуре должен быть в корне.

## Шаги
1. **Git**
   - `git reset --hard HEAD` + удалить незакоммиченное в `src/` (`.idea/`, `docs/`, `shumodav.pdf` — неотслеживаемые, не трогаем).
   - `git tag v1-legacy main`; `git switch -c v2 main`.
2. **Очистка и структура**
   - `git rm -r src/denoise_app run.py` (и прочий старый код v1 в корне, если есть).
   - `git mv`/перенос `docs/CLAUDE.md` → `CLAUDE.md` в корень; `docs/TRAINING_COLAB.md` остаётся.
   - Создать каркас: `src/shumodav/__init__.py` (пустой), `core/`, `denoisers/`, `nn/kair/`, `bench/`, `gui/` с пустыми `__init__.py`; `tests/`, `scripts/`, `results/`, `weights/README.md` (что и откуда класть). Модули-файлы самих блоков не создаём заранее.
3. **pyproject.toml** (poetry, PEP 621 или `[tool.poetry]` — как удобнее poetry 2.3):
   - имя `shumodav`, `packages = [{include="shumodav", from="src"}]`, `python >=3.11`;
   - deps: numpy, scipy, opencv-python, pywavelets, scikit-image, matplotlib, pandas, tqdm, pyqt5;
   - extras: `neural = [torch]`, `bm3d = [bm3d]` (optional);
   - dev: pytest; убрать старый script `denoise-app`, black/mypy-настройки не нужны.
   - Удалить старый `poetry.lock`, сгенерировать новый.
4. **.gitignore**: заменить на короткий: `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.venv/`, `.pytest_cache/`, `*.log`, `data/`, `weights/*.pth`, `results/cache/`, `.idea/`. Убрать игнор `*.png` и т.п.
5. **`scripts/get_data.py`**: если `data/train400`, `data/bsd68`, `data/set12` уже есть — выход; иначе `git clone --depth 1 https://github.com/cszn/DnCNN` во `tempfile.mkdtemp()`, `shutil.copytree` трёх папок по путям из спеки, удаление временной папки (`shutil.rmtree` в `finally`). Печать числа файлов.
6. **`scripts/download_weights.py`**: скачать через `urllib.request` `dncnn_gray_blind.pth`, `dncnn_color_blind.pth`, `drunet_gray.pth` с `https://github.com/cszn/KAIR/releases/download/v1.0/{name}` в `weights/`, пропуская существующие. **Не запускать.**
7. **Установка и данные**: `poetry env use python3.12 && poetry install`; `poetry run python scripts/get_data.py`.
8. Отметить `[x] Блок 0` в чек-листе `CLAUDE.md` (с заметкой про Python 3.12, если пригодится) и закоммитить `блок 0: подготовка репозитория и данных` в `v2`. Push — не в этом блоке (по спеке push в Блоке 3); тег тоже пока локальный.

## Проверка
- `poetry install` проходит без ошибок; `poetry run python -c "import shumodav, cv2, pywt, skimage, PyQt5"` работает.
- Подсчёт файлов: `data/train400` = 400, `data/bsd68` = 68, `data/set12` = 12.
- `git tag -l` показывает `v1-legacy`; `git show v1-legacy:src/denoise_app/core/wavelet_processor.py` читается.
- `git status` чистый (кроме `.idea/`, `shumodav.pdf`), `data/` не попадает в индекс.
