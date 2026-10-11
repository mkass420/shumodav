import signal
import sys
import traceback

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

import shumodav.denoisers  # noqa: F401  (регистрация методов)
from .main_window import MainWindow


def main():
    # PyQt5 по умолчанию аварийно завершает процесс (abort) при любом необработанном исключении в слоте
    sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.prewarm()
    # Ctrl+C в терминале закрывает окно штатно (иначе KeyboardInterrupt прилетает в слот и роняет процесс)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QTimer()
    tick.timeout.connect(lambda: None)  # даём интерпретатору обработать сигнал
    tick.start(200)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
