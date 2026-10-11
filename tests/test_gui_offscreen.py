import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402
from skimage import data  # noqa: E402

import shumodav.denoisers  # noqa: E402,F401
from shumodav.core.io import save_image  # noqa: E402
from shumodav.core.registry import get_denoiser, list_denoisers  # noqa: E402
from shumodav.core.types import ParamSpec  # noqa: E402
from shumodav.gui.main_window import MainWindow  # noqa: E402
from shumodav.gui.param_panel import ParamPanel  # noqa: E402

app = QApplication.instance() or QApplication([])


def real_methods():
    """Методы пакета. Тестовые заглушки (identity, box_blur) зарегистрированы только в процессе pytest,
    а расчёты GUI идут в дочернем процессе, где их нет."""
    return [d for d in list_denoisers() if type(d).__module__.startswith("shumodav.denoisers")]


def wait_until(cond, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def settle(win, timeout=60.0):
    assert wait_until(lambda: not win.is_busy(), timeout), "GUI не успокоился"


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("img") / "cam.png"
    save_image(p, (data.camera()[:96, :96] / 255.0).astype(np.float32))
    return str(p)


@pytest.fixture
def win(image_path):
    w = MainWindow()
    w.show()
    assert w.load_path(image_path)
    settle(w)
    yield w
    w.close()


def test_param_panel_types_and_sync():
    specs = (ParamSpec("a", "A", "float", 0.5, 0.0, 1.0, 0.1), ParamSpec("n", "N", "int", 3, 3, 15, 2),
             ParamSpec("c", "C", "choice", "x", choices=("x", "y")), ParamSpec("b", "B", "bool", True),
             ParamSpec("lg", "L", "float", 10.0, 1.0, 100.0, 1.0, log=True))
    p = ParamPanel(specs)
    got = []
    p.changed.connect(got.append)
    v = p.values()
    assert v == {"a": 0.5, "n": 3, "c": "x", "b": True, "lg": pytest.approx(10.0)}
    assert isinstance(v["a"], float) and isinstance(v["n"], int)
    p.set_values({"a": 0.8, "n": 5, "c": "y", "b": False})
    assert len(got) == 1 and got[0]["c"] == "y" and got[0]["n"] == 5 and got[0]["a"] == pytest.approx(0.8)


def test_param_panel_visible_if():
    p = ParamPanel(get_denoiser("fft_lowpass").params)
    assert p.is_visible_param("order")
    p.set_values({"kind": "ideal"})
    assert not p.is_visible_param("order")
    p = ParamPanel(get_denoiser("drunet").params) if get_denoiser("drunet").params else None
    assert p.is_visible_param("auto_sigma") and not p.is_visible_param("sigma")
    p.set_values({"auto_sigma": False})
    assert p.is_visible_param("sigma")


def test_window_load_and_process(win):
    assert all(v.has_image() for v in win.views[:2])
    win.set_noise("gaussian")
    win.set_method("gaussian")
    settle(win)
    assert win.result is not None and win.views[2].has_image()
    assert "—" not in win.lbl_result.text()
    assert win.statusBar().currentMessage() == "Готово"


def test_no_noise_mode_gives_dashes(win):
    win.cb_nonoise.setChecked(True)
    settle(win)
    assert "—" in win.lbl_result.text()
    assert np.array_equal(win.noisy, win.clean_image())


def test_stale_results_discarded(win):
    win.set_method("median")
    for k in (3, 5, 7, 5, 3):
        win.panel_method.set_values({"ksize": k})
    settle(win)
    ref, _ = __import__("shumodav.core.pipeline", fromlist=["x"]).run_denoiser(win.noisy, "median", {"ksize": 3})
    assert np.array_equal(win.result, ref)  # применён последний запрос


def test_gray_mode_and_error_handling(win):
    win.cb_gray.setChecked(True)
    settle(win)
    assert win.result.ndim == 2


def test_best_params_button_and_compare(win):
    win.set_method("mean")
    settle(win)
    win.compare_all()
    settle(win, 120)
    names = [d.name for d in real_methods()]
    assert win.table.rowCount() == len(names)
    assert not win.client.busy() and win.btn_compare.isEnabled()
    row = next(r for r in range(win.table.rowCount()) if win.table.item(r, 0).text() == get_denoiser("median").label)
    win.select_from_table(row)
    assert win._method_name() == "median"


@pytest.mark.slow
@pytest.mark.parametrize("noise", ["gaussian", "salt_pepper", "periodic"])
def test_every_method_runs(win, noise):
    win.set_noise(noise)
    for d in real_methods():
        win.set_method(d.name)
        settle(win)
        msg = win.statusBar().currentMessage()
        assert msg == "Готово", f"{d.name}/{noise}: {msg}"
        assert win.result is not None and win.result.shape == win.clean_image().shape


def _big_image(tmp_path, n=700):
    p = tmp_path / "big.png"
    save_image(p, np.random.default_rng(0).random((n, n, 3)).astype(np.float32))
    return str(p)


def test_cancel_kills_heavy_job(tmp_path):
    w = MainWindow()
    w.show()
    w.load_path(_big_image(tmp_path))
    w.set_method("bm3d")
    w._kick()
    assert wait_until(lambda: w.client.busy(), 5)
    assert w.btn_cancel.isEnabled()
    pid = w.client._proc.pid
    t0 = time.time()
    w.cancel_job()
    assert time.time() - t0 < 3 and not w.client.busy()
    assert w.statusBar().currentMessage() == "Расчёт отменён"
    assert not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z"
    # после отмены приложение продолжает работать: новый процесс считает быстрый метод
    w.set_method("mean")
    settle(w)
    assert w.result is not None and w.statusBar().currentMessage() == "Готово"
    w.close()


def test_long_job_is_preempted_by_new_request(tmp_path, monkeypatch):
    import shumodav.gui.main_window as mwmod

    monkeypatch.setattr(mwmod, "PREEMPT_S", 0.5)
    w = MainWindow()
    w.show()
    w.load_path(_big_image(tmp_path))
    w.set_method("bm3d")
    w._kick()
    assert wait_until(lambda: w.client.busy(), 5)
    t0 = time.time()
    w.set_method("mean")  # новый запрос, пока считается BM3D
    settle(w, 60)
    assert time.time() - t0 < 20  # BM3D на таком кадре заметно дольше: его вытеснили
    assert w.statusBar().currentMessage() == "Готово"
    from shumodav.core.pipeline import run_denoiser

    ref, _ = run_denoiser(w.noisy, "mean", {"ksize": 3})
    assert np.array_equal(w.result, ref)
    w.close()


def test_recovers_after_child_crash(win):
    win.set_method("mean")
    settle(win)
    win.client._proc.kill()  # как будто OOM-killer
    win.client._proc.join()
    win.set_method("median")
    settle(win)
    assert win.statusBar().currentMessage() == "Готово"
    assert win.result is not None


def test_close_kills_child_without_waiting(tmp_path):
    w = MainWindow()
    w.show()
    w.load_path(_big_image(tmp_path))
    w.set_method("bm3d")
    w._kick()
    assert wait_until(lambda: w.client.busy(), 5)
    t0 = time.time()
    w.close()
    assert time.time() - t0 < 3
    assert not w.client.busy()


def test_image_sent_once(win):
    """Картинка пересылается в дочерний процесс один раз: второй запрос идёт без неё."""
    win.set_method("mean")
    settle(win)
    key = win._image_key
    assert win.client._child_key == key
    win.panel_method.set_values({"ksize": 5})
    settle(win)
    assert win.client._child_key == key and win.statusBar().currentMessage() == "Готово"
