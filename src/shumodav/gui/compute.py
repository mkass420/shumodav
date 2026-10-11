"""Расчёты GUI в отдельном процессе: их можно отменить (убить процесс), а память тяжёлых методов не копится в окне.

Модуль не импортирует Qt: дочерний процесс (spawn) импортирует его заново.
"""
import multiprocessing as mp
import os
import queue
import time

from ..core.pipeline import apply_noise, evaluate, run_denoiser
from ..core.registry import get_denoiser, list_denoisers, resolve_params

POLL_S = 1.0  # как часто дочерний процесс проверяет, жив ли родитель


# ---------- сторона дочернего процесса ----------
def _trim():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)  # вернуть освобождённую память системе
    except (OSError, AttributeError):
        pass


def _noisy(req, clean):
    if req["no_noise"]:
        return clean
    return apply_noise(clean, req["noise"], req["noise_params"], req["seed"])


def process_request(req, clean) -> dict:
    out = {"noisy": None, "result": None, "metrics": None, "seconds": None, "vis": None, "error": None}
    try:
        out["noisy"] = noisy = _noisy(req, clean)
        res, sec = run_denoiser(noisy, req["method"], req["method_params"])
        out["result"], out["seconds"] = res, sec
        if not req["no_noise"]:
            out["metrics"] = evaluate(clean, noisy, res)
        d = get_denoiser(req["method"])
        try:
            out["vis"] = d.visualize(noisy, **resolve_params(d.params, req["method_params"]))
        except Exception:
            out["vis"] = None
    except Exception as e:  # включая MemoryError
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def compare_request(req, clean):
    """Генератор строк таблицы сравнения: по одному методу за раз."""
    noisy = _noisy(req, clean)
    for d in list_denoisers():
        params = req["best_params"].get(d.name) or {}
        item = {"method": d.name, "family": d.family, "params": resolve_params(d.params, params),
                "psnr": None, "ssim": None, "seconds": None, "error": None}
        try:
            res, sec = run_denoiser(noisy, d.name, params)
            item["seconds"] = sec
            if not req["no_noise"]:
                ev = evaluate(clean, noisy, res)
                item["psnr"], item["ssim"] = ev["psnr"], ev["ssim"]
        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"
        yield item


def serve(in_q, out_q):
    import shumodav.denoisers  # noqa: F401  (регистрация методов в дочернем процессе)

    parent = os.getppid()
    cache = {"key": None, "clean": None}  # последняя пересланная картинка
    while True:
        try:
            msg = in_q.get(timeout=POLL_S)
        except queue.Empty:
            if os.getppid() != parent:  # родитель умер — выходим
                break
            continue
        kind, job_id, req = msg
        if kind == "quit":
            break
        if "clean" in req:
            cache["key"], cache["clean"] = req["image_key"], req["clean"]
        clean = cache["clean"]
        if kind == "process":
            out_q.put(("result", job_id, process_request(req, clean)))
        elif kind == "compare":
            for item in compare_request(req, clean):
                out_q.put(("row", job_id, item))
            out_q.put(("compare_done", job_id, None))
        _trim()
    out_q.close()
    out_q.join_thread()


# ---------- сторона GUI ----------
class ComputeClient:
    """Управляет дочерним процессом: отправка задач, опрос ответов, отмена."""

    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._in = self._out = None
        self._child_key = None  # какая картинка уже лежит в дочернем процессе
        self.job_id = None  # идёт ли сейчас задача (и какая)
        self.started = 0.0

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._drop_queues()
        self._in, self._out = self._ctx.Queue(), self._ctx.Queue()
        self._proc = self._ctx.Process(target=serve, args=(self._in, self._out), daemon=True)
        self._proc.start()
        self._child_key = None

    def submit(self, kind: str, job_id: int, req: dict, image_key: int):
        self.start()
        req = dict(req)
        req["image_key"] = image_key
        if self._child_key == image_key:
            req.pop("clean", None)  # картинка уже в дочернем процессе — второй раз не шлём
        else:
            self._child_key = image_key
        self._in.put((kind, job_id, req))
        self.job_id, self.started = job_id, time.monotonic()

    def busy(self) -> bool:
        return self.job_id is not None

    def elapsed(self) -> float:
        return time.monotonic() - self.started if self.busy() else 0.0

    def poll(self) -> list:
        """Все пришедшие сообщения; при аварии дочернего процесса — одно сообщение ('crashed', job_id, текст)."""
        msgs = []
        if self._out is None:
            return msgs
        while True:
            try:
                msg = self._out.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            msgs.append(msg)
            if msg[0] in ("result", "compare_done") and msg[1] == self.job_id:
                self.job_id = None
        if self.busy() and not self._proc.is_alive():
            code, job = self._proc.exitcode, self.job_id
            self.kill()
            msgs.append(("crashed", job, f"процесс расчёта завершился аварийно (код {code}), возможно, не хватило памяти"))
        return msgs

    def kill(self):
        """Отмена: убить дочерний процесс; следующая задача запустит новый."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(2)
        self._proc = None
        self._drop_queues()
        self._child_key = None
        self.job_id = None

    def _drop_queues(self):
        for q in (self._in, self._out):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._in = self._out = None

    def shutdown(self):
        self.kill()
