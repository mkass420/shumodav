"""Отчёт: python -m shumodav.bench.report"""
import json

import numpy as np
import pandas as pd

import shumodav.denoisers  # noqa: F401
from ..core.metrics import psnr
from ..core.pipeline import run_denoiser
from ..core.registry import get_denoiser, get_noise
from .common import CROP, DATA, LEVELS, NOISES, RESULTS, load_images, noisy_for

FAST = 0.05  # секунд
FAMILY_COLORS = {"spatial": "#1f77b4", "frequency": "#ff7f0e", "wavelet": "#2ca02c", "neural": "#d62728"}
FAMILY_LABELS = {"spatial": "пространственные", "frequency": "частотные", "wavelet": "вейвлеты", "neural": "нейросети"}


def load_results(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or RESULTS / "results.csv")
    return df.drop_duplicates(["image", "noise", "level", "method"], keep="last")


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Метод × (шум, уровень), среднее по картинкам; столбцы в порядке спеки."""
    t = df.pivot_table(index="method", columns=["noise", "level"], values=value, aggfunc="mean")
    cols = [(n, l) for n in NOISES for l in LEVELS if (n, l) in t.columns]
    t = t[cols]
    order = df.drop_duplicates("method").set_index("method")["family"]
    fam_rank = {f: i for i, f in enumerate(FAMILY_COLORS)}
    return t.loc[sorted(t.index, key=lambda m: (fam_rank.get(order[m], 9), m))]


def noisy_row(df: pd.DataFrame, value: str) -> pd.Series:
    t = df.pivot_table(index=["noise", "level"], values=value, aggfunc="mean")[value]
    return t


def md_table(t: pd.DataFrame, noisy: pd.Series, digits: int) -> str:
    """Markdown-таблица; лучшее значение в каждом столбце выделено жирным."""
    labels = {n: get_noise(n).label for n in NOISES}
    head = ["Метод"] + [f"{labels[n]} {l}" for n, l in t.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    best = t.max()
    lines.append("| *Шумная* | " + " | ".join(f"*{noisy[c]:.{digits}f}*" for c in t.columns) + " |")
    for m, row in t.iterrows():
        cells = []
        for c, v in row.items():
            s = "—" if pd.isna(v) else f"{v:.{digits}f}"
            cells.append(f"**{s}**" if v == best[c] else s)
        lines.append(f"| {get_denoiser(m).label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def best_by_noise_md(df: pd.DataFrame) -> str:
    lines = ["| Шум | Лучший метод | PSNR, дБ | Лучший быстрый (< 0.05 с) | PSNR, дБ | Время, с |", "|---|---|---|---|---|---|"]
    for n in NOISES:
        d = df[df.noise == n]
        if d.empty:
            continue
        g = d.groupby("method").agg(psnr=("psnr", "mean"), sec=("seconds", "mean"))
        top = g.psnr.idxmax()
        fast = g[g.sec < FAST]
        if fast.empty:
            fl, fp, fs = "—", "—", "—"
        else:
            f = fast.psnr.idxmax()
            fl, fp, fs = get_denoiser(f).label, f"{fast.psnr[f]:.2f}", f"{fast.sec[f]:.4f}"
        lines.append(f"| {get_noise(n).label} | {get_denoiser(top).label} | {g.psnr[top]:.2f} | {fl} | {fp} | {fs} |")
    return "\n".join(lines) + "\n"


def _heatmap(t, annot, path, title, cmap, norm_by_column=True, center_zero=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = t.values.astype(float)
    if norm_by_column:  # цвет по столбцам: уровни шума несопоставимы
        lo, hi = np.nanmin(vals, 0), np.nanmax(vals, 0)
        color = (vals - lo) / np.where(hi - lo == 0, 1, hi - lo)
        vmin, vmax = 0, 1
    else:
        color, m = vals, np.nanmax(np.abs(vals))
        vmin, vmax = (-m, m) if center_zero else (np.nanmin(vals), np.nanmax(vals))
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * t.shape[1], 1.5 + 0.42 * t.shape[0]))
    im = ax.imshow(color, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if not np.isnan(vals[i, j]):
                dark = cmap == "viridis" and (color[i, j] - vmin) / (vmax - vmin) < 0.5  # белый текст на тёмном
                ax.text(j, i, f"{round(vals[i, j], 2) + 0.0:{annot}}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([f"{get_noise(n).label}\n{l}" for n, l in t.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels([get_denoiser(m).label for m in t.index], fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, label="цвет нормирован по столбцам" if norm_by_column else "")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def speed_vs_quality(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("method").agg(psnr=("psnr", "mean"), sec=("seconds", "mean"), family=("family", "first"))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fam, color in FAMILY_COLORS.items():
        s = g[g.family == fam]
        ax.scatter(s.sec, s.psnr, c=color, label=FAMILY_LABELS[fam], s=50, zorder=3)
    for m, r in g.iterrows():
        ax.annotate(get_denoiser(m).label, (r.sec, r.psnr), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set(xscale="log", xlabel="Среднее время на картинку 256×256, с", ylabel="Средний PSNR, дБ",
           title="Скорость и качество (среднее по всем шумам и уровням)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def examples(df, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    names = sorted(df.image.unique())
    name = "05.png" if "05.png" in names else names[0]
    clean = dict(load_images(DATA / "set12", first=name, n=1))[name]
    sl = slice((CROP - 96) // 2, (CROP + 96) // 2)
    for noise in NOISES:
        d = df[(df.image == name) & (df.noise == noise) & (df.level == "mid")]
        if d.empty:
            continue
        noisy = noisy_for(clean, name, noise, "mid")
        tiles = [("Чистая", clean, None), ("Шумная", noisy, d.psnr_noisy.iloc[0])]
        for _, r in d.iterrows():
            res, _ = run_denoiser(noisy, r.method, json.loads(r.params_json))
            tiles.append((get_denoiser(r.method).label, res, r.psnr))
        cols = 5
        rows = -(-len(tiles) // cols)
        fig, axs = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.9 * rows), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (title, im, p) in zip(axs.ravel(), tiles):
            ax.imshow(im[sl, sl], cmap="gray", vmin=0, vmax=1)
            ax.set_title(title if p is None else f"{title}\n{p:.2f} дБ", fontsize=8)
        fig.suptitle(f"{get_noise(noise).label}, уровень mid ({name}, кроп 96×96)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.8)
        fig.savefig(out_dir / f"{noise}_mid.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    df = load_results()
    t_psnr, t_ssim = pivot(df, "psnr"), pivot(df, "ssim")
    (RESULTS / "summary_psnr.md").write_text(md_table(t_psnr, noisy_row(df, "psnr_noisy"), 2))
    (RESULTS / "summary_ssim.md").write_text(md_table(t_ssim, noisy_row(df, "ssim_noisy"), 3))
    (RESULTS / "best_by_noise.md").write_text(best_by_noise_md(df))
    _heatmap(t_psnr, ".1f", RESULTS / "heatmap_psnr.png", "Средний PSNR, дБ (Set12)", "viridis")
    _heatmap(t_ssim, ".2f", RESULTS / "heatmap_ssim.png", "Средний SSIM (Set12)", "viridis")
    gain = t_psnr - noisy_row(df, "psnr_noisy").reindex(t_psnr.columns).values
    _heatmap(gain, "+.1f", RESULTS / "gain_psnr.png", "Выигрыш PSNR относительно шумной, дБ", "RdYlGn",
             norm_by_column=False, center_zero=True)
    speed_vs_quality(df, RESULTS / "speed_vs_quality.png")
    examples(df, RESULTS / "examples")
    print("Отчёт записан в", RESULTS)


if __name__ == "__main__":
    main()
