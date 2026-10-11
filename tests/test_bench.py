import json

import numpy as np
import pandas as pd

import shumodav.denoisers  # noqa: F401
from shumodav.bench.common import noise_seed
from shumodav.bench.report import md_table, noisy_row, pivot
from shumodav.bench.run import run_benchmark
from shumodav.bench.tune import grid_combos
from shumodav.core.registry import get_denoiser


def test_noise_seed_stable_and_distinct():
    assert noise_seed("05.png", "gaussian", "mid") == noise_seed("05.png", "gaussian", "mid")
    assert noise_seed("05.png", "gaussian", "mid") != noise_seed("05.png", "gaussian", "low")


def test_grid_combos_product_and_hidden_dedup():
    combos = grid_combos(get_denoiser("mean"))
    assert [c["ksize"] for c in combos] == [3, 5, 7]
    # order скрыт при kind != butterworth → комбинации схлопываются: 1 + 1 + 3 вместо 9 по (kind, order)
    lp = grid_combos(get_denoiser("fft_lowpass"))
    assert len({json.dumps(c, sort_keys=True) for c in lp}) == len(lp)
    n_d0 = len(get_denoiser("fft_lowpass").params[1].grid)
    assert len(lp) == n_d0 * (1 + 1 + 3)


def test_grid_combos_cap_deterministic():
    d = get_denoiser("wavelet")
    a, b = grid_combos(d, cap=5), grid_combos(d, cap=5)
    assert len(a) == 5 and a == b


def _images():
    rng = np.random.default_rng(0)
    return [(f"{i}.png", rng.random((64, 64)).astype(np.float32)) for i in range(2)]


def test_run_benchmark_resume(tmp_path):
    csv = tmp_path / "r.csv"
    args = (_images(), ["gaussian", "salt_pepper"], ["mid"], ["mean", "median"])
    assert run_benchmark(*args, {}, csv) == 2 * 2 * 1 * 2
    assert run_benchmark(*args, {}, csv) == 0  # резюмирование
    params = {"gaussian": {"mid": {"mean": {"ksize": 5}}}}
    assert run_benchmark(*args, params, csv) == 2  # изменились параметры mean на gaussian/mid для 2 картинок
    df = pd.read_csv(csv)
    assert list(df.columns)[:4] == ["image", "noise", "level", "method"]


def test_report_bold_best(tmp_path):
    csv = tmp_path / "r.csv"
    run_benchmark(_images(), ["gaussian"], ["mid"], ["mean", "median"], {}, csv)
    df = pd.read_csv(csv)
    md = md_table(pivot(df, "psnr"), noisy_row(df, "psnr_noisy"), 2)
    assert md.count("**") == 2  # одна ячейка (один столбец) выделена
