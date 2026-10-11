# План: Блок 4 — бенчмарк (tune / run / report)

## Context
Блоки 0–3 готовы и запушены (`v2` — ветка по умолчанию, 89 тестов зелёные). Доступны 12 методов: 9 классических + `dncnn`, `drunet` (`custom_cnn` появится после Colab, Блок 6). Блок 4 по `CLAUDE.md` даёт три команды, которые формируют итоговую таблицу эффективности для записки: подбор параметров (`tune`), прогон на тестовом наборе (`run`) и отчёт (`report`). Обучение в Colab идёт параллельно и этому блоку не мешает; `custom_cnn` позже дописывается в тот же CSV командой `run --methods custom_cnn`.

Методология (из спеки): `tune` — первые 4 картинки BSD68, `run` — Set12 (ничего на нём не подбиралось), все картинки серые, центр-кроп 256, шумная картинка генерируется один раз на (картинка, шум, уровень) и подаётся всем методам.

Первым действием при исполнении: скопировать этот план в `docs/plans/04-block4-bench.md`.

## Файлы (`src/shumodav/bench/`)

**`common.py`** — общее для трёх команд:
- пути (`ROOT`, `RESULTS`, `DATA`) и список уровней/шумов в порядке спеки (gaussian, salt_pepper, speckle, poisson, periodic × low, mid, high);
- `load_images(folder, n=None, first=None)` → `list[(имя, серый center_crop 256)]` через `core.io.load_image/center_crop` (уже есть); `first` — имя файла, которое ставится первым (для `05.png`);
- `noise_seed(image, noise, level)` — **`zlib.crc32` от строки** (не `hash()`: он рандомизируется между запусками и сломал бы воспроизводимость и резюмирование);
- `noisy_for(img, name, noise, level)` = `core.pipeline.apply_noise(img, noise, get_noise(noise).levels[level], seed)`;
- `to_jsonable(params)` (numpy → python).

**`tune.py`** — `python -m shumodav.bench.tune [--quick]`
- Картинки: первые 4 из `data/bsd68` (по имени). Методы: доступные из `list_denoisers()` **с непустой сеткой** (нейросети без `grid` пропускаются сами).
- `grid_combos(denoiser, cap=60, seed=0)`: декартово произведение `ParamSpec.grid` по параметрам с сеткой, остальные — `default`. Параметры, чей `visible_if` не выполнен (например, `order` при `kind != butterworth`), фиксируются на `default` и комбинации-дубликаты схлопываются (не гоняем одно и то же). Если комбинаций > 60 — `random.Random(0).sample(..., 60)`.
- Для каждого (шум × уровень): шумные картинки считаются один раз; для каждого метода выбирается комбинация с максимальным средним PSNR (`core.pipeline.run_denoiser` + `core.metrics.psnr`). Прогресс — `tqdm`.
- Результат `results/best_params.json` вида `{noise: {level: {method: params}}}`. Запись **со слиянием** с уже существующим файлом (`quick` не затирает остальное) и после каждого шума (прогресс не теряется при обрыве).
- `--quick`: 2 картинки, только уровень `mid`, не более 8 комбинаций на метод.

**`run.py`** — `python -m shumodav.bench.run [--methods ...] [--noises ...] [--quick]`
- Картинки: `data/set12`, порядок: `05.png` первой, затем остальные по имени. `--quick`: первые 2 (т. е. `05.png` + одна) и только уровень `mid`.
- Для каждого (картинка, шум, уровень): одна шумная картинка → все выбранные доступные методы. Параметры: `best_params[noise][level][method]`, иначе `default`; в CSV пишутся **разрешённые** (`resolve_params`) параметры.
- Строки дописываются в `results/results.csv` (колонки: `image, noise, level, method, family, params_json, psnr_noisy, psnr, ssim_noisy, ssim, seconds`), построчно с `flush`, шапка — только в новом файле.
- Резюмирование: строка пропускается, если в CSV уже есть тот же `(image, noise, level, method, params_json)`. Ключ включает параметры, поэтому после нового `tune` метод переcчитается, а устаревшая строка не пропадёт — `report` оставляет **последнюю** строку на (image, noise, level, method).
- Ошибка одного метода ловится, печатается и не роняет прогон (строка не пишется, при повторном запуске попробует снова).
- Логика вынесена в функцию `run_benchmark(images, noises, levels, methods, params, csv_path)`, чтобы тестировать без `data/`.

**`report.py`** — `python -m shumodav.bench.report` (читает `results/results.csv`, пишет в `results/`)
- Данные: `pandas`, дедупликация по `keep="last"`; среднее по картинкам.
- `summary_psnr.md`, `summary_ssim.md`: таблица «метод × (шум, уровень)», строка «Шумная (без обработки)» для справки, лучший в каждом столбце **жирным** (таблицу в markdown собираем вручную — `tabulate` не в зависимостях).
- `heatmap_psnr.png`, `heatmap_ssim.png`: `imshow` с аннотациями значений; цвет нормируется по столбцам (уровни шума несопоставимы), подписи по-русски.
- `gain_psnr.png`: выигрыш PSNR (`psnr − psnr_noisy`), тепловая карта с расходящейся палитрой, центр в 0.
- `speed_vs_quality.png`: scatter, X = среднее время (лог-шкала), Y = средний PSNR по всем шумам/уровням, подписи методов, цвет по `family`.
- `best_by_noise.md`: для каждого шума лучший метод по среднему PSNR и лучший «быстрый» (среднее время < 0.05 с) с PSNR и временем.
- `examples/<noise>_mid.png`: для каждого шума на уровне `mid` — картинка `05.png` (или первая из CSV): заново генерируется шумная (тот же seed), каждый метод запускается с `params_json` из CSV, кроп 96×96 из центра; сетка «чистая, шумная, методы…», подпись — метод и **PSNR полной картинки из CSV**.
- matplotlib с backend `Agg` импортируется внутри функций; шрифт DejaVu Sans поддерживает кириллицу.

## Тесты — `tests/test_bench.py` (без `data/`, на синтетике)
- `noise_seed` стабилен и различает аргументы; `grid_combos`: декартово произведение, дедупликация по `visible_if`, лимит 60 и детерминизм выборки.
- `run_benchmark` на 2 синтетических картинках 64×64 и методах `mean`, `median`: пишет ожидаемое число строк; повторный вызов не добавляет строк; смена параметров добавляет новые.
- `report`: на маленьком DataFrame/CSV строит таблицу, лучший в столбце выделен `**`.

## Порядок выполнения
1. `tune --quick` → `run --quick` → `report`. Открыть полученные PNG (Read) и проверить читаемость: подписи, нет наезжающего текста, цвета, `examples/*` показывают осмысленные картинки; поправить вёрстку по результату.
2. `pytest -q` зелёный; отметить `[x] Блок 4` в `CLAUDE.md`; коммит `блок 4: бенчмарк` (+ `Co-Authored-By`). Результаты quick-прогона в коммит не включаем (итоговые `results/` коммитятся после полного прогона); push не делаем.
3. Дать пользователю команды полного прогона в фоне (сам не жду; оценка: tune ≈ 10–20 мин, run ≈ 10–15 мин на CPU, дольше всего `bm3d` и `drunet`):
   ```
   cd ~/code/TOZI && nohup sh -c 'poetry run python -m shumodav.bench.tune > results/tune.log 2>&1 && poetry run python -m shumodav.bench.run > results/run.log 2>&1' &
   ```
   `*.log` в `.gitignore`. Команды безопасно перезапускать: `tune` сливается с существующим JSON, `run` резюмируется.

## Проверка
- Quick-версии всех трёх команд отработали, в `results/` есть: `best_params.json`, `results.csv`, `summary_psnr.md`, `summary_ssim.md`, `heatmap_psnr.png`, `heatmap_ssim.png`, `gain_psnr.png`, `speed_vs_quality.png`, `best_by_noise.md`, `examples/<noise>_mid.png` для 5 шумов.
- Повторный `run --quick` ничего не дописывает (резюмирование).
- Санити по цифрам: на gaussian mid у `bm3d`/`drunet`/`dncnn` PSNR заметно выше классики, у `fft_notch` на periodic — лучшие значения среди классики, `median` лидирует на salt_pepper среди классических.
