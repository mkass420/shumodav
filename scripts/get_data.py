"""Скачивает train400, bsd68, set12 из репозитория DnCNN в data/."""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = "https://github.com/cszn/DnCNN"
# (папка в репозитории, папка назначения)
ITEMS = [
    ("TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400", "train400"),
    ("testsets/BSD68", "bsd68"),
    ("testsets/Set12", "set12"),
]


def main() -> None:
    if all((DATA / dst).is_dir() for _, dst in ITEMS):
        print("Данные уже есть, ничего не делаю.")
        return
    DATA.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(tmp / "dncnn")], check=True)
        for src, dst in ITEMS:
            if not (DATA / dst).exists():
                shutil.copytree(tmp / "dncnn" / src, DATA / dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for _, dst in ITEMS:
        print(dst, len(list((DATA / dst).iterdir())))


if __name__ == "__main__":
    main()
