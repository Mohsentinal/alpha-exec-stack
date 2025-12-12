# research/utils_results.py
from __future__ import annotations
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"


def _stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_result_dirs() -> None:
    for d in (RESULTS_DIR, METRICS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_metrics(df: pd.DataFrame, name: str) -> Path:
    """Write a timestamped CSV + a 'latest' CSV for quick browsing."""
    ensure_result_dirs()
    ts = _stamp()
    out = METRICS_DIR / f"{name}_{ts}.csv"
    df.to_csv(out, index=False)
    latest = METRICS_DIR / f"{name}_latest.csv"
    df.to_csv(latest, index=False)
    return out


def save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str,
                 title: str, name: str) -> Path:
    """Simple matrix heatmap (no seaborn) saved to PNG."""
    ensure_result_dirs()
    pivot = df.pivot(index=index, columns=columns, values=values).sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_xticks(np.arange(pivot.shape[1]), labels=[f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[f"{r:.2f}" for r in pivot.index])
    fig.colorbar(im, ax=ax, label=values)
    fig.tight_layout()

    ts = _stamp()
    out = PLOTS_DIR / f"{name}_{ts}.png"
    fig.savefig(out)
    fig.savefig(PLOTS_DIR / f"{name}_latest.png")
    plt.close(fig)
    return out