import sys

from PyQt5.QtWidgets import QApplication

import shumodav.denoisers  # noqa: F401  (регистрация методов)
from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
