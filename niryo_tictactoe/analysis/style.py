"""
File: niryo_tictactoe/analysis/style.py
Description: Shared plotting style for all figures in docs/images.
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, GRAY, DARK = "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#94a3b8", "#0f172a"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})
