"""Подбор параметров: python -m shumodav.bench.tune [--quick]"""
import argparse
import itertools
import json
import random

import numpy as np
from tqdm import tqdm

import shumodav.denoisers  # noqa: F401
from ..core.metrics import psnr
from ..core.pipeline import run_denoiser
from ..core.registry import list_denoisers
from .common import DATA, LEVELS, NOISES, RESULTS, load_images, noisy_for, to_jsonable


def grid_combos(denoiser, cap: int = 60, seed: int = 0) -> list[dict]:
    """Декартово произведение сеток; скрытые (visible_if) параметры фиксируются на default."""
    defaults = {s.name: s.default for s in denoiser.params}
    gridded = [s for s in denoiser.params if s.grid]
    combos, seen = [], set()
    for values in itertools.product(*[s.grid for s in gridded]):
        p = dict(defaults)
        p.update({s.name: v for s, v in zip(gridded, values)})
        for s in gridded:
            if s.visible_if and any(p.get(k) != v for k, v in s.visible_if.items()):
                p[s.name] = s.default
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            combos.append(p)
    if len(combos) > cap:
        combos = random.Random(seed).sample(combos, cap)
    return combos


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    images = load_images(DATA / "bsd68", n=2 if a.quick else 4)
    levels = ("mid",) if a.quick else LEVELS
    cap = 8 if a.quick else 60
    methods = [d for d in list_denoisers() if any(s.grid for s in d.params)]
    path = RESULTS / "best_params.json"
    RESULTS.mkdir(exist_ok=True)
    best = json.loads(path.read_text()) if path.exists() else {}

    for noise in NOISES:
        for level in levels:
            noisy = [noisy_for(img, name, noise, level) for name, img in images]
            for d in tqdm(methods, desc=f"{noise}/{level}", leave=False):
                scores = []
                for p in grid_combos(d, cap):
                    s = np.mean([psnr(img, run_denoiser(n, d.name, p)[0]) for (_, img), n in zip(images, noisy)])
                    scores.append((s, p))
                top = max(scores, key=lambda t: t[0])
                best.setdefault(noise, {}).setdefault(level, {})[d.name] = to_jsonable(top[1])
                print(f"{noise}/{level} {d.name}: {top[0]:.2f} дБ {top[1]}", flush=True)
        path.write_text(json.dumps(best, indent=1, ensure_ascii=False))  # после каждого шума


if __name__ == "__main__":
    main()
