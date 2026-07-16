# V0001I
# PROJECT CONTRACT
# NO AI-GENERATED IMAGES. PYTHON/PANDAS/MATPLOTLIB/IPYWIDGETS ONLY.

from __future__ import annotations

import io
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display, clear_output

VERSION = "V0001I"
REPO = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"

CSV_FILES = {
    "FFT frequencies": f"{RAW}/spectral/EMBOPSA_FFT_FREQUENCIES_V0001D.csv",
    "Harmonic coefficients": f"{RAW}/spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv",
    "Harmonic reconstruction": f"{RAW}/reconstruction/EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv",
    "Holdout validation": f"{RAW}/validation/EMBOPSA_HOLDOUT_VALIDATION_V0001D.csv",
    "Validation metrics": f"{RAW}/validation/EMBOPSA_VALIDATION_METRICS_V0001D.csv",
}

C = {
    "bg": "#000000",
    "panel": "#000000",
    "grid": "#263247",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "blue": "#5B8CFF",
    "violet": "#B388FF",
    "magenta": "#FF6EC7",
    "orange": "#FF9F43",
    "gold": "#FFD166",
    "green": "#35E0A1",
    "red": "#FF5C7A",
}

CACHE: dict[str, pd.DataFrame] = {}

def fetch_csv(url: str) -> pd.DataFrame:
    if url not in CACHE:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        CACHE[url] = pd.read_csv(io.StringIO(r.text))
    return CACHE[url].copy()

def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=[np.number]).columns.tolist()

def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, color=C["text"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for s in ax.spines.values():
        s.set_color(C["grid"])

def make_widget():
    file_dd = widgets.Dropdown(
        options=list(CSV_FILES.keys()),
        description="CSV",
        layout=widgets.Layout(width="420px"),
    )
    x_dd = widgets.Dropdown(description="X", layout=widgets.Layout(width="320px"))
    y_dd = widgets.Dropdown(description="Y", layout=widgets.Layout(width="320px"))
    plot_dd = widgets.Dropdown(
        options=["Line", "Scatter", "Histogram"],
        description="Plot",
        layout=widgets.Layout(width="240px"),
    )
    rows_slider = widgets.IntSlider(
        value=20,
        min=5,
        max=100,
        step=5,
        description="Rows",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )
    output = widgets.Output()

    def refresh_columns(*_):
        frame = fetch_csv(CSV_FILES[file_dd.value])
        nums = numeric_columns(frame)
        x_dd.options = nums
        y_dd.options = nums
        if nums:
            x_dd.value = nums[0]
            y_dd.value = nums[1] if len(nums) > 1 else nums[0]

    def render(*_):
        with output:
            clear_output(wait=True)
            frame = fetch_csv(CSV_FILES[file_dd.value])

            display(frame.head(rows_slider.value))

            summary = pd.DataFrame({
                "dtype": frame.dtypes.astype(str),
                "missing": frame.isna().sum(),
                "unique": frame.nunique(dropna=True),
            })
            display(summary)

            nums = numeric_columns(frame)
            if nums:
                display(frame[nums].describe().T)

            if not nums or x_dd.value is None:
                return

            plt.close("all")
            plt.ioff()
            plt.rcParams.update({
                "figure.facecolor": C["bg"],
                "axes.facecolor": C["panel"],
                "savefig.facecolor": C["bg"],
                "text.color": C["text"],
                "axes.labelcolor": C["text"],
                "xtick.color": C["text"],
                "ytick.color": C["text"],
                "font.family": "DejaVu Sans",
            })

            fig, ax = plt.subplots(figsize=(11.5, 6.5))

            if plot_dd.value == "Histogram":
                values = frame[x_dd.value].dropna().to_numpy(dtype=float)
                ax.hist(
                    values,
                    bins=40,
                    color=C["cyan"],
                    edgecolor=C["text"],
                    linewidth=0.5,
                    alpha=0.9,
                )
                style_axes(
                    ax,
                    f"{file_dd.value}: {x_dd.value} distribution",
                    x_dd.value,
                    "Count",
                )

            elif plot_dd.value == "Scatter":
                ax.scatter(
                    frame[x_dd.value],
                    frame[y_dd.value],
                    s=28,
                    color=C["magenta"],
                    edgecolor=C["text"],
                    linewidth=0.3,
                    alpha=0.85,
                )
                style_axes(
                    ax,
                    f"{file_dd.value}: {y_dd.value} versus {x_dd.value}",
                    x_dd.value,
                    y_dd.value,
                )

            else:
                ordered = frame.sort_values(x_dd.value)
                ax.plot(
                    ordered[x_dd.value],
                    ordered[y_dd.value],
                    color=C["cyan"],
                    linewidth=1.1,
                )
                style_axes(
                    ax,
                    f"{file_dd.value}: {y_dd.value} versus {x_dd.value}",
                    x_dd.value,
                    y_dd.value,
                )

            fig.tight_layout()
            display(fig)
            plt.close(fig)

    file_dd.observe(refresh_columns, names="value")
    for control in [file_dd, x_dd, y_dd, plot_dd, rows_slider]:
        control.observe(render, names="value")

    refresh_columns()
    display(
        widgets.VBox([
            widgets.HBox([file_dd, plot_dd]),
            widgets.HBox([x_dd, y_dd]),
            rows_slider,
            output,
        ])
    )
    render()

if __name__ == "__main__":
    make_widget()
# V0001I
