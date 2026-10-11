from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shumodav.nn.data import PatchDataset  # noqa: E402
from shumodav.nn.model import DnCNNLite  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_model_shape_and_size():
    m = DnCNNLite()
    assert m(torch.rand(1, 1, 64, 64)).shape == (1, 1, 64, 64)
    n = sum(p.numel() for p in m.parameters())
    assert 160_000 < n < 180_000


def test_patch_dataset():
    imgs = [np.random.default_rng(i).random((90, 90)).astype(np.float32) for i in range(3)]
    ds = PatchDataset(imgs, length=40, seed=1)
    noisy, clean = ds[5]
    assert noisy.shape == clean.shape == (1, 50, 50)
    assert noisy.dtype == torch.float32
    assert 0 <= noisy.min() and noisy.max() <= 1
    assert torch.equal(ds[5][0], noisy)  # детерминизм по индексу
    assert not torch.equal(ds[6][0], noisy)


@pytest.mark.skipif(not (ROOT / "data/train400").exists(), reason="нет данных")
def test_train_smoke_and_resume(tmp_path):
    from shumodav.nn.train import main

    args = ["--smoke", "--data", str(ROOT / "data/train400"), "--val", str(ROOT / "data/bsd68"),
            "--out", str(tmp_path / "m.pth")]
    main(args)
    for f in ("m.pth", "train_log.csv", "train_curve.png", "ckpt_last.pth"):
        assert (tmp_path / f).exists()
    main(args + ["--resume"])
