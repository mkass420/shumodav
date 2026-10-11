"""Прогон на Set12: python -m shumodav.bench.run [--methods ...] [--noises ...] [--quick]"""
import argparse
import csv
import json

from tqdm import tqdm

import shumodav.denoisers  # noqa: F401
from ..core.pipeline import evaluate, run_denoiser
from ..core.registry import get_denoiser, list_denoisers, resolve_params
from .common import DATA, LEVELS, NOISES, RESULTS, load_images, noisy_for, to_jsonable

COLUMNS = ["image", "noise", "level", "method", "family", "params_json",
           "psnr_noisy", "psnr", "ssim_noisy", "ssim", "seconds"]


def _done_keys(csv_path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["image"], r["noise"], r["level"], r["method"], r["params_json"]) for r in csv.DictReader(f)}


def run_benchmark(images, noises, levels, methods, params, csv_path) -> int:
    """Дописывает строки в csv_path, пропуская уже посчитанные. Возвращает число новых строк."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(csv_path)
    new = 0
    total = len(images) * len(noises) * len(levels)
    with open(csv_path, "a", newline="") as f, tqdm(total=total, desc="run") as bar:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if f.tell() == 0:
            w.writeheader()
        for name, clean in images:
            for noise in noises:
                for level in levels:
                    bar.update(1)
                    noisy = None
                    for m in methods:
                        d = get_denoiser(m)
                        p = to_jsonable(resolve_params(d.params, params.get(noise, {}).get(level, {}).get(m)))
                        pj = json.dumps(p, sort_keys=True, ensure_ascii=False)
                        if (name, noise, level, m, pj) in done:
                            continue
                        try:
                            noisy = noisy_for(clean, name, noise, level) if noisy is None else noisy
                            res, sec = run_denoiser(noisy, m, p)
                            ev = evaluate(clean, noisy, res)
                        except Exception as e:  # один сбойный метод не роняет прогон
                            print(f"ОШИБКА {m} {name} {noise}/{level}: {e}", flush=True)
                            continue
                        w.writerow({"image": name, "noise": noise, "level": level, "method": m,
                                    "family": d.family, "params_json": pj, **ev, "seconds": sec})
                        f.flush()
                        new += 1
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--noises", nargs="+", default=list(NOISES))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    images = load_images(DATA / "set12", n=2 if a.quick else None, first="05.png")
    levels = ("mid",) if a.quick else LEVELS
    available = {d.name for d in list_denoisers()}
    methods = a.methods or sorted(available, key=lambda n: get_denoiser(n).family)
    missing = [m for m in methods if m not in available]
    if missing:
        print("Недоступны, пропускаю:", ", ".join(missing))
    methods = [m for m in methods if m in available]
    pfile = RESULTS / "best_params.json"
    params = json.loads(pfile.read_text()) if pfile.exists() else {}
    n = run_benchmark(images, a.noises, levels, methods, params, RESULTS / "results.csv")
    print(f"Добавлено строк: {n}")


if __name__ == "__main__":
    main()
