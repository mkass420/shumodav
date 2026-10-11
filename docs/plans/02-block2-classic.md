# План: Блок 2 — классические методы (spatial, FFT, notch, wavelet, NLM, BM3D)

## Context
Блоки 0–1 готовы (ветка `v2`, ядро в `src/shumodav/core/`, 29 тестов зелёные). Блок 2 по `CLAUDE.md` добавляет все неиспользующие нейросети методы из таблицы «Список методов»: `mean`, `median`, `gaussian`, `bilateral`, `nlm`, `bm3d`, `fft_lowpass`, `fft_notch`, `wavelet`. Вейвлеты и маски FFT переносятся из `v1-legacy` по логике (`git show v1-legacy:src/denoise_app/core/wavelet_processor.py` и `.../image_processor.py` — я их прочитал), структура новая. Контракты: float32 [0,1], вход не мутируется, параметры берутся из `ParamSpec.default`, лишние ключи игнорируются (`resolve_params` из `core/registry.py` вызывает pipeline). Код простой, без GPU, `DetailEnhancer`/blend не переносим.

Первым действием при исполнении: скопировать этот план в `docs/plans/02-block2-classic.md`.

Окружение: OpenCV 5.0.0.93, numpy 2.4, PyWavelets 1.9, scikit-image 0.26; `bm3d` не установлен.

## Файлы (`src/shumodav/denoisers/`)

**`__init__.py`** — импортирует `spatial, frequency, wavelet, bm3d_` (регистрация). Импортируется явно из тестов, GUI, bench (core его не импортирует, чтобы не тянуть pywt в Colab).

**`spatial.py`** (`@register_denoiser`, family `spatial`, все `supports_color=True`, cv2 принимает 1 и 3 канала):
- `mean`: `ksize` int 3..15 шаг 2 (grid 3,5,7); `cv2.blur`. Чётное ksize приводить к нечётному (`k | 1`).
- `median`: `ksize` (grid 3,5,7); float32 напрямую при ksize ≤ 5, при > 5 через uint8 и обратно.
- `gaussian`: `sigma` 0.1..5 (grid 0.5,1,1.5,2,3); `cv2.GaussianBlur(ksize=(0,0))`.
- `bilateral`: `d` (5,9), `sigma_color` (0.05,0.1,0.2), `sigma_space` (3,5); `cv2.bilateralFilter` на float32.
- `nlm`: `h` 1..50 в единицах 0..255 (grid 5,10,15,20,30); uint8 → `fastNlMeansDenoising` (серое) / `fastNlMeansDenoisingColored` (цвет, h и hColor одинаковые), окна 7/21 константами. **Риск:** в OpenCV 5.0 функция могла переехать/измениться — проверить в самом начале; запасной вариант — `skimage.restoration.denoise_nl_means` (`h = h/255`, `fast_mode=True`).
- Общий хелпер `_to_u8` / `_from_u8` внутри файла.

**`frequency.py`**
- `fft_lowpass`: `kind` choice ideal/gaussian/butterworth (default butterworth), `d0` 0.05..0.5 (default 0.2), `order` int (1,2,4; `visible_if={"kind": "butterworth"}`), `padding` 0.1. Маска из v1 (расстояние от центра `h//2, w//2`, нормированное на расстояние до угла); паддинг `cv2.copyMakeBorder(..., BORDER_REFLECT)`; `scipy.fft.fft2/ifft2` + `fftshift/ifftshift` поканально; обрезка паддинга. Центр маски = 1 (DC сохраняется). `visualize` → RGB float32: лог-спектр серого, область маски подсвечена/затемнена.
- `fft_notch`: параметры `threshold` (k × медиана амплитуды; default подберу эмпирически на тесте, grid около 10,20,40), `radius` (2,4,8), `exclude_center` (float, доля `min(h,w)`, default 0.05, чтобы не гасить НЧ-центр). Алгоритм: серый → окно Ханна (только для детекции) → `fftshift(fft2)` → пики = локальные максимумы (`scipy.ndimage.maximum_filter`) с амплитудой > `threshold·median` вне центральной зоны, не более ~20 сильнейших, для каждого добавляется симметричный пик → маска = ∏(1 − exp(−d²/2r²)) → применить к спектру **исходного** (без окна) изображения поканально. `visualize` → лог-спектр с отмеченными кругами пиками.
- Общие функции (`_log_spectrum`, `_filter_channels`) — в этом же файле.

**`wavelet.py`** (family `wavelet`, `supports_color=True`):
- Параметры по таблице: `wavelet` choice (db1,db2,db4,sym4,coif1,bior1.3; default db4), `level` 1..4 (default 2), `threshold` 0.02..0.3 (default 0.1), `mode` soft/hard, `swt` bool (default False), `threshold_rule` manual/bayes, `chroma_strength` 0..1 (default 0.5), `post_smooth` bool.
- Логика из v1: серое — один канал; цвет → `cv2.cvtColor(RGB2YCrCb)` (float32), каналы Cb/Cr: порог × (1 + 5·chroma_strength) и подавление коэффициентов × (1 − 0.8·chroma_strength); ЯЛ не трогаем. DWT: `wavedec2/waverec2` (обрезка до исходного размера); SWT: паддинг до кратного `2^level` (`edge`), `swt2/iswt2`. LL не пороговится.
- `threshold_rule`: `manual` — абсолютный порог; `bayes` — BayesShrink по подзонам: σ_n = MAD(HH)/0.6745 (DWT — по HH самого тонкого уровня; SWT — по HH каждого уровня отдельно, т. к. SWT без нормировки), σ_x = sqrt(max(σ_y² − σ_n², 0)), T = σ_n²/σ_x (если σ_x = 0 — T = max|c|). Для цветовых каналов T тоже умножается на chroma-буст.
- `level` клампится к максимально допустимому для размера картинки и фильтра (`pywt.dwtn_max_level` / `swt_max_level`), чтобы 64×64 и нечётные размеры не падали.
- `post_smooth`: `cv2.bilateralFilter(d=5, sigmaColor=0.08, sigmaSpace=5)` (значения v1).
- `visualize` → мозаика из `_coeffs_to_image` v1 (лог-амплитуды, нормировка 0..1) на сером изображении.

**`bm3d_.py`** — `bm3d`, family `spatial`, `sigma` 5..75 (grid 10,15,25,35,50). `import bm3d` лениво внутри `is_available()` / `denoise()` (try/except ImportError); серое — `bm3d.bm3d(img, sigma_psd=sigma/255)`, цвет — `bm3d.bm3d_rgb`. Сначала пробую `poetry install -E bm3d`; если не ставится за разумное время — метод просто недоступен, заметка в чек-листе, идём дальше.

## Тесты — `tests/test_denoisers_smoke.py`
Тестовые картинки без `data/`: `skimage.data.camera()` (серая 128×128 кроп, цвет — стек/`skimage.data.astronaut()` кроп), чтобы тесты работали и в чистом клоне.
- Для каждого `list_denoisers()` (кроме тестового `identity*` — они безвредны): серое 64×64 и цветное 64×64×3 через `run_denoiser` с параметрами по умолчанию → форма, float32, [0,1], вход не изменён; `visualize` возвращает `None` или ndarray.
- Доп. прогоны: `median` с ksize=7; `wavelet` с `swt=True`, `threshold_rule="bayes"`, `mode="hard"`, `post_smooth=True`, нечётный размер 63×65, `level=4` на 64×64 (кламп); `fft_lowpass` для всех трёх `kind`.
- По смыслу: `fft_lowpass` меняет среднюю яркость < 1%; `fft_notch` на `periodic` mid поднимает PSNR ≥ 5 дБ; `median` повышает PSNR на `salt_pepper`; `wavelet`, `gaussian`, `nlm` не ухудшают PSNR на `gaussian` σ=25 (простые санити-проверки).

## Завершение блока
1. `poetry run pytest -q` зелёный.
2. `poetry run python -c "import shumodav.denoisers as _; from shumodav.core.registry import list_denoisers; print([d.name for d in list_denoisers()])"` — печатает все методы блока (с `bm3d`, если установился).
3. Отметить `[x] Блок 2` в чек-листе `CLAUDE.md` (заметка про bm3d, если недоступен), коммит `блок 2: классические методы` (+ `Co-Authored-By` по системному напоминанию). Push не делаем.

## Проверка
- Тесты выше.
- Ручной прогон на `data/set12/05.png` (серое), gaussian σ=25: напечатать PSNR всех методов с дефолтами — ожидаю заметный прирост у `nlm`, `wavelet`, `bilateral`, `bm3d`; для `periodic` mid — у `fft_notch` (> +5 дБ).
- Посмотреть `visualize()` для `fft_lowpass`, `fft_notch`, `wavelet`, сохранив PNG во временную папку (scratchpad) и открыв их.
