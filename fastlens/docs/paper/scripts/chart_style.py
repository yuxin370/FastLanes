"""Shared paper chart sizes and colors. Edit these to restyle all statistical figures."""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FASTLENS_ARTIFACT_ROOT", PAPER.parents[2])).resolve()
RESULTS = ROOT / "fastlens/benchmarks/paper/results"
OUTPUT = Path(os.environ.get("FASTLENS_FIGURE_OUTPUT", PAPER)).resolve()
WIDTH = 3.35
COLORS = {
    "Model only": "#bfc5cb", "PyTorch": "#7f8790", "DALI": "#e0a348",
    "CoorDL": "#6aaed6", "FFCV": "#9b86b5", "L3": "#b87968",
    "CPU-DCT": "#376f9f", "FastLens": "#43967a",
}
BLUE, ORANGE, GREEN = COLORS["CPU-DCT"], COLORS["DALI"], COLORS["FastLens"]
# Distinct soft hues for bars, with blue emphasis and warm/cool contrast.
BAR_PALETTE = (
    "#cbd1d8", "#deb0bd", "#e5d59d", "#c2d0a3",
    "#9fcdbf", "#bdafd7", "#e9be9c", "#98bedd",
)
BAR_COLORS = {
    "PyTorch": BAR_PALETTE[0], "DALI": BAR_PALETTE[1],
    "CoorDL": BAR_PALETTE[2], "FFCV": BAR_PALETTE[3],
    "FFCV JPEG": BAR_PALETTE[4], "L3": BAR_PALETTE[5],
    "CPU-DCT": BAR_PALETTE[6], "FastLens": BAR_PALETTE[7],
}
BAR_STYLE = {"edgecolor": "black", "linewidth": .4}
HATCHES = {
    "PyTorch": "---", "DALI": "//", "CoorDL": "\\\\", "FFCV": "--",
    "FFCV JPEG": "\\\\\\", "L3": "xx", "CPU-DCT": "///", "FastLens": "",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "axes.edgecolor": "#555555", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.linewidth": .55, "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": .9, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "hatch.linewidth": .25,
})


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    path = OUTPUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + extension), dpi=200)
    plt.close(fig)
