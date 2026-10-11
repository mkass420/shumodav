# План: Блок 1 — ядро (типы, реестр, шумы, метрики, pipeline, io, тесты)

## Context
Блок 0 сделан (ветка `v2`, каркас `src/shumodav/`, данные в `data/`). Блок 1 по `CLAUDE.md` даёт фундамент, на который опираются все остальные блоки: контракты изображений, `ParamSpec`, реестры шумов и методов, 5 моделей шума, метрики и pipeline. Код пишем просто, без лишней универсальности (срок — день). `core` не импортирует PyQt5, pywt, bm3d, torch, matplotlib.

Первым действием при исполнении: скопировать этот план в `docs/plans/01-block1-core.md` (история планов хранится в docs).

## Файлы (все в `src/shumodav/core/`, тесты в `tests/`)

1. **`types.py`**: `Image = np.ndarray`; `ParamSpec` — frozen dataclass ровно как в `CLAUDE.md` (name, label, kind, default, min, max, step, choices, log, grid, visible_if).

2. **`registry.py`**
   - `class Denoiser(ABC)` с контрактом из спеки (`name`, `label`, `family`, `params`, `supports_color=True`, `is_available()`, абстрактный `denoise(img, **p)`, `visualize()` → None).
   - Словари `_NOISES`, `_DENOISERS`; декораторы `register_noise` / `register_denoiser` хранят **экземпляр** класса, повторная регистрация того же имени перезаписывает (чтобы не падать при повторном импорте в pytest).
   - `get_noise(name)`, `get_denoiser(name)` (понятная `KeyError`), `list_noises()`, `list_denoisers(available_only=True)`.
   - Общий хелпер `resolve_params(specs, p) -> dict`: берёт `default` для отсутствующих ключей, лишние ключи отбрасывает. Используется и в pipeline, и позже в методах.

3. **`noise.py`**: `class NoiseModel(ABC)` (`name`, `label`, `params`, `levels`, `apply(img, rng, **p)`). Все `apply` работают на любом float-массиве (2D, 3D, патч 50×50), возвращают новый float32, обрезанный в [0, 1], и берут случайность только из `rng`.
   - `gaussian`: `img + rng.normal(0, sigma/255, img.shape)`; levels 15/25/50.
   - `salt_pepper`: маска по `shape[:2]` (один пиксель — все каналы): `u = rng.random(shape[:2])`; `u < amount*ratio` → 1 (соль), `amount*ratio <= u < amount` → 0 (перец); levels 0.05/0.10/0.20, `ratio=0.5`.
   - `speckle`: `img + img * rng.normal(0, sigma, shape)`; 0.10/0.20/0.30.
   - `poisson`: `rng.poisson(img * peak) / peak`; peak 100/30/10 (вход клипнуть ≥0 перед poisson).
   - `periodic`: `amplitude * sin(2π·freq·(x·cosθ + y·sinθ) + phase)`, фаза из `rng.uniform(0, 2π)`, добавляется ко всем каналам; levels amp 0.10/0.20/0.30, freq 0.1, angle 45.
   - Каждая модель — `ParamSpec`-ы с `label` по-русски и `grid=None`; декоратор `@register_noise`.

4. **`metrics.py`**: `psnr(a, b)` → `skimage.metrics.peak_signal_noise_ratio(a, b, data_range=1.0)` (обернуть в `np.errstate`, чтобы PSNR(x, x) = inf без варнингов); `ssim(a, b)` с `channel_axis=-1` для 3D; `Timer` — контекст-менеджер на `time.perf_counter()` с полем `.seconds`.

5. **`io.py`**: `load_image(path, gray=False)` — `cv2.imread` → BGR→RGB → float32/255 (при `gray=True` читать `IMREAD_GRAYSCALE`); `save_image(path, img)` — clip, ×255, round, uint8, RGB→BGR; `to_gray(img)` (RGB→gray через `cv2.cvtColor`, 2D остаётся как есть); `center_crop(img, size)` — понадобится бенчмарку (мелочь, но удобно положить сюда сразу). Явная ошибка, если файл не прочитался.

6. **`pipeline.py`**
   - `apply_noise(img, noise_name, params, seed)` → `default_rng(seed)`, `resolve_params`, вызов `apply`.
   - `run_denoiser(img, denoiser_name, params) -> (result, seconds)`: `resolve_params`; если `img.ndim == 3` и `not supports_color` — по каналам и склейка; прогрев (один прогон до замера) для `family == "neural"`; время `perf_counter`; результат `np.clip(..., 0, 1).astype(np.float32)`. Вход не меняется.
   - `evaluate(clean, noisy, result)` → `dict(psnr_noisy, psnr, ssim_noisy, ssim)`.

## Тесты (`tests/`)
- `test_noise.py`: для каждого шума × формы (64×64, 64×64×3, 50×50) — детерминизм при одном seed, форма и dtype float32, диапазон [0, 1], вход не мутируется; gaussian σ=25 на серой 0.5: std разности ≈ 25/255 ±10% (картинку брать крупную, напр. 256×256, и учесть клиппинг — 0.5 ± 3σ укладывается в [0,1]); доля испорченных пикселей salt_pepper ≈ amount.
- `test_metrics.py`: `psnr(x, x) == inf`; при известном MSE совпадает с `10·log10(1/mse)`; ssim(x, x) ≈ 1; `Timer` > 0.
- `test_pipeline.py`: тестовая заглушка `identity` (`@register_denoiser`, family spatial, `supports_color=False` для проверки поканальной ветки, плюс вариант с цветом) проходит `run_denoiser` и `evaluate`; `resolve_params` подставляет defaults и игнорирует лишние ключи; `list_denoisers()` содержит `identity`; вход не мутируется.
- (маленький) `test_io.py`: save → load round-trip на временной папке, ошибка для несуществующего файла.

## Завершение блока
- `poetry run pytest -q` зелёный.
- Отметить `[x] Блок 1` в чек-листе `CLAUDE.md`.
- Коммит `блок 1: ядро` (с `Co-Authored-By` по системному напоминанию). Push не делаем.

## Проверка
- `poetry run pytest -q` — все тесты проходят.
- `poetry run python -c "import sys, shumodav.core.pipeline; print([m for m in ('PyQt5','pywt','bm3d','torch','matplotlib') if m in sys.modules])"` → пустой список (core не тянет тяжёлые зависимости).
- Быстрый ручной прогон: загрузить `data/set12/05.png`, `apply_noise` gaussian mid, `evaluate(clean, noisy, noisy)` — PSNR шумной ≈ 20 дБ для σ=25.
