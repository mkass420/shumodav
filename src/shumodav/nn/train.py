"""Обучение DnCNNLite: python -m shumodav.nn.train --help"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core.metrics import psnr
from ..core.pipeline import apply_noise
from .data import NOISE_TYPES, PatchDataset, load_folder
from .model import DnCNNLite


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train400")
    ap.add_argument("--val", default="data/bsd68")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="weights/custom_cnn.pth")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


@torch.no_grad()
def validate(model, val_imgs, device) -> dict[str, float]:
    """Средний PSNR по картинкам для каждого типа шума на уровне mid."""
    from ..core.registry import get_noise

    model.eval()
    res = {}
    for k, noise in enumerate(NOISE_TYPES):
        params = get_noise(noise).levels["mid"]
        vals = []
        for j, clean in enumerate(val_imgs):
            noisy = apply_noise(clean, noise, params, seed=1000 * k + j)
            out = model(torch.from_numpy(noisy)[None, None].to(device)).clamp(0, 1)[0, 0].cpu().numpy()
            vals.append(psnr(clean, out))
        res[noise] = float(np.mean(vals))
    model.train()
    return res


def save_log(history, out_dir: Path):
    cols = ["iter", "loss"] + [f"val_psnr_{n}" for n in NOISE_TYPES] + ["val_psnr_mean"]
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(history)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [r["iter"] for r in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(its, [r["loss"] for r in history])
    ax[0].set(xlabel="итерация", ylabel="MSE (обучение)", title="Функция потерь", yscale="log")
    for n in NOISE_TYPES:
        ax[1].plot(its, [r[f"val_psnr_{n}"] for r in history], label=n)
    ax[1].set(xlabel="итерация", ylabel="PSNR, дБ", title="Валидация (BSD68, уровень mid)")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curve.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    val_every, n_val, workers = 1000, 10, 2
    if a.smoke:
        a.iters, a.batch, a.device = 50, 8, "cpu"
        val_every, n_val, workers = 25, 2, 0
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                          "cpu" if a.device == "auto" else a.device)
    if device.type == "cpu":
        workers = 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else out.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt_last.pth"

    torch.manual_seed(a.seed)
    train_imgs, val_imgs = load_folder(a.data), load_folder(a.val, n_val)
    model = DnCNNLite().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.6 * a.iters), int(0.85 * a.iters)], 0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best, history = 0, -1.0, []
    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, history = ck["iter"], ck["best"], ck["history"]
        print(f"Продолжаю с итерации {start}")

    loader = DataLoader(PatchDataset(train_imgs, (a.iters - start) * a.batch, seed=a.seed + start),
                        batch_size=a.batch, num_workers=workers, pin_memory=use_amp, drop_last=True)
    model.train()
    t0, run_loss, n_loss, it = time.perf_counter(), 0.0, 0, start

    def checkpoint(it):
        nonlocal best
        val = validate(model, val_imgs, device)
        mean = float(np.mean(list(val.values())))
        loss = run_loss / max(n_loss, 1)
        history.append({"iter": it, "loss": loss, **{f"val_psnr_{k}": v for k, v in val.items()},
                        "val_psnr_mean": mean})
        print(f"iter {it} | loss {loss:.5f} | val PSNR: " + " | ".join(f"{k} {v:.1f}" for k, v in val.items()), flush=True)
        if mean > best:
            best = mean
            torch.save(model.state_dict(), out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "iter": it, "best": best, "history": history}, ckpt_path)
        save_log(history, out.parent)

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = torch.nn.functional.mse_loss(model(noisy), clean)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        it += 1
        run_loss += loss.item(); n_loss += 1
        if it % val_every == 0 and it < a.iters:
            checkpoint(it)
            run_loss, n_loss = 0.0, 0
    if it > start:
        checkpoint(it)
    print(f"Готово за {(time.perf_counter() - t0) / 60:.1f} мин. Лучший средний PSNR: {best:.2f}")


if __name__ == "__main__":
    main()
