"""Isolate model storage and present SQL-derived calibration results."""

import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display


def configure(trees):
    if type(trees) is not int or not 1 <= trees <= 200:
        raise ValueError("N_ESTIMATORS must be an integer from 1 through 200")
    directory = TemporaryDirectory(prefix="vgi-lightgbm-models-")
    previous = {key: os.environ.get(key) for key in ("LIGHTGBM_MODELS_DIR", "OMP_NUM_THREADS")}
    os.environ["LIGHTGBM_MODELS_DIR"] = directory.name
    os.environ["OMP_NUM_THREADS"] = "1"
    return directory, previous


def cleanup(directory, previous):
    directory.cleanup()
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def prose(text: str) -> None:
    display(Markdown(text))


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.08, 0.90, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.82))
    fig.text(0.08, 0.03, source, fontsize=8, color="#6C7D73")
    plt.show()
