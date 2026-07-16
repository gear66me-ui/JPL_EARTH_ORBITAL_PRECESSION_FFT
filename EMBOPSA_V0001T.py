# EMBOPSA_V0001T
# Interactive two-point cycle measurement.
# NO AI-GENERATED IMAGES.
# Deterministic Python / NumPy / Pandas / Matplotlib / ipywidgets only.

from __future__ import annotations

import io
import sys
import subprocess
import importlib.util

if importlib.util.find_spec("ipympl") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ipympl"]
    )

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "widget")

VERSION = "V0001T"

DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

C = {
    "bg": "#000000",
    "grid": "#28364A",
    "text": "#F4F7FB",
    "cyan": "#52D6FF",
    "gold": "#FFD166",
    "red": "#FF5C7A",
    "green": "#35E0A1",
    "violet": "#B388FF",
}

response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()

df = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna().sort_values("t_kyr").reset_index(drop=True)
df["age_myr"] = -df["t_kyr"] / 1000.0
df["pole_radius"] = np.hypot(df["p"], df["q"])
df["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(df["pole_radius"], 0.0, 1.0))
)

MAX_AGE = float(df["age_myr"].max())
TODAY_I = float(df.loc[df["t_kyr"].abs().idxmin(), "inclination_deg"])

global_max_row = df.loc[df["inclination_deg"].idxmax()]
GLOBAL_MAX_I = float(global_max_row["inclination_deg"])
GLOBAL_MAX_AGE = float(global_max_row["age_myr"])

window_width = widgets.SelectionSlider(
    options=[
        ("0.5 Myr", 0.5),
        ("1 Myr", 1.0),
        ("2 Myr", 2.0),
        ("5 Myr", 5.0),
        ("10 Myr", 10.0),
        ("25 Myr", 25.0),
        ("50 Myr", 50.0),
        ("100 Myr", 100.0),
        ("250 Myr", MAX_AGE),
    ],
    value=10.0,
    description="Window",
    continuous_update=False,
    layout=widgets.Layout(width="720px"),
)

window_start = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=max(0.0, MAX_AGE - 10.0),
    step=0.1,
    description="Position",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="720px"),
)

sample_spacing = widgets.SelectionSlider(
    options=[
        ("1 kyr", 1),
        ("10 kyr", 10),
        ("25 kyr", 25),
        ("50 kyr", 50),
        ("100 kyr", 100),
        ("250 kyr", 250),
        ("1 Myr", 1000),
    ],
    value=10,
    description="Sampling",
    continuous_update=False,
    layout=widgets.Layout(width="720px"),
)

today_button = widgets.Button(description="Today")
maximum_button = widgets.Button(description="Go to maximum")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
clear_button = widgets.Button(description="Clear A–B", button_style="danger")

status = widgets.HTML()
measurement = widgets.HTML(
    value=(
        "<b>Cycle measurement:</b> Tap once for A, then tap again for B. "
        "A third tap begins a new measurement."
    )
)
output = widgets.Output()

fig = None
ax = None
click_cid = None
marker_lines = []
marker_labels = []
selected_x = []
current_subset = None

def sync_position_limit(*_):
    width = float(window_width.value)
    window_start.max = max(0.0, MAX_AGE - width)
    if window_start.value > window_start.max:
        window_start.value = window_start.max

def nearest_record(age_myr: float, subset: pd.DataFrame):
    index = (subset["age_myr"] - age_myr).abs().idxmin()
    row = subset.loc[index]
    return float(row["age_myr"]), float(row["inclination_deg"])

def clear_markers(_=None, redraw=True):
    global marker_lines, marker_labels, selected_x

    for artist in marker_lines + marker_labels:
        try:
            artist.remove()
        except Exception:
            pass

    marker_lines = []
    marker_labels = []
    selected_x = []

    measurement.value = (
        "<b>Cycle measurement:</b> Tap once for A, then tap again for B. "
        "A third tap begins a new measurement."
    )

    if redraw and fig is not None:
        fig.canvas.draw_idle()

def add_marker(x_value: float, label: str, color: str):
    global marker_lines, marker_labels

    line = ax.axvline(
        x_value,
        color=color,
        linewidth=1.8,
        linestyle="--",
        zorder=8,
    )

    y_top = ax.get_ylim()[1]
    text = ax.text(
        x_value,
        y_top,
        f" {label}: {x_value:.6f} Myr",
        color=color,
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="top",
        ha="right",
        zorder=9,
        bbox=dict(facecolor=C["bg"], edgecolor=color, alpha=0.85),
    )

    marker_lines.append(line)
    marker_labels.append(text)

def on_plot_click(event):
    global selected_x

    if event.inaxes is not ax or event.xdata is None or current_subset is None:
        return

    if len(selected_x) >= 2:
        clear_markers(redraw=False)

    x_value, inclination = nearest_record(float(event.xdata), current_subset)
    selected_x.append(x_value)

    if len(selected_x) == 1:
        add_marker(x_value, "A", C["gold"])
        measurement.value = (
            f"<b>A:</b> {x_value:,.6f} Myr before J2000 "
            f"({inclination:.6f}°). Tap the second point for B."
        )

    elif len(selected_x) == 2:
        add_marker(x_value, "B", C["violet"])

        a, b = selected_x
        delta_myr = abs(b - a)
        delta_kyr = delta_myr * 1000.0
        delta_years = delta_myr * 1_000_000.0

        measurement.value = (
            f"<b>A:</b> {a:,.6f} Myr &nbsp; | &nbsp; "
            f"<b>B:</b> {b:,.6f} Myr &nbsp; | &nbsp; "
            f"<b>Separation:</b> "
            f"{delta_years:,.0f} years "
            f"= {delta_kyr:,.3f} kyr "
            f"= {delta_myr:,.6f} Myr"
        )

    fig.canvas.draw_idle()

def render(*_):
    global fig, ax, click_cid, current_subset

    sync_position_limit()
    clear_markers(redraw=False)

    start = float(window_start.value)
    width = float(window_width.value)
    end = min(MAX_AGE, start + width)

    subset = df[
        (df["age_myr"] >= start)
        & (df["age_myr"] <= end)
    ].copy()

    current_subset = subset

    stride = max(1, int(sample_spacing.value))
    plotted = subset.iloc[::stride].copy()

    local_max_row = subset.loc[subset["inclination_deg"].idxmax()]
    local_max_i = float(local_max_row["inclination_deg"])
    local_max_age = float(local_max_row["age_myr"])

    local_min_i = float(subset["inclination_deg"].min())

    status.value = (
        f"<b>Window:</b> {start:,.3f}–{end:,.3f} Myr before J2000 &nbsp; | &nbsp; "
        f"<b>Tap the curve twice to measure a cycle.</b>"
    )

    with output:
        clear_output(wait=True)

        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

        plt.rcParams.update({
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["bg"],
            "savefig.facecolor": C["bg"],
            "text.color": C["text"],
            "axes.labelcolor": C["text"],
            "xtick.color": C["text"],
            "ytick.color": C["text"],
            "font.family": "DejaVu Sans",
        })

        fig, ax = plt.subplots(figsize=(15, 7.5))

        ax.plot(
            plotted["age_myr"],
            plotted["inclination_deg"],
            color=C["cyan"],
            linewidth=1.15,
            label="Earth orbital inclination",
        )

        ax.axhline(
            TODAY_I,
            color=C["gold"],
            linestyle=":",
            linewidth=1.0,
            label=f"J2000 = {TODAY_I:.6f}°",
        )

        ax.axhline(
            3.0,
            color=C["red"],
            linestyle=":",
            linewidth=1.0,
            label="3° reference",
        )

        ax.scatter(
            [local_max_age],
            [local_max_i],
            color=C["green"],
            edgecolor=C["text"],
            s=60,
            zorder=5,
        )

        ax.set_title(
            "Earth Orbital Inclination — Touch Two Points to Measure a Cycle",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel("Millions of years before J2000")
        ax.set_ylabel("Inclination (degrees)")
        ax.set_xlim(start, end if end > start else start + 0.001)
        ax.set_ylim(max(0.0, local_min_i - 0.2), max(3.15, local_max_i + 0.2))
        ax.grid(True, color=C["grid"], linewidth=0.55, alpha=0.8)

        for spine in ax.spines.values():
            spine.set_color(C["grid"])

        legend = ax.legend(
            loc="upper right",
            facecolor=C["bg"],
            edgecolor=C["grid"],
            framealpha=0.95,
        )

        for text in legend.get_texts():
            text.set_color(C["text"])

        click_cid = fig.canvas.mpl_connect("button_press_event", on_plot_click)

        fig.tight_layout()
        display(fig)

def go_today(_):
    window_start.value = 0.0

def go_maximum(_):
    width = float(window_width.value)
    window_start.value = min(
        max(0.0, GLOBAL_MAX_AGE - width / 2.0),
        window_start.max,
    )

def go_previous(_):
    window_start.value = max(
        0.0,
        window_start.value - float(window_width.value),
    )

def go_next(_):
    window_start.value = min(
        window_start.max,
        window_start.value + float(window_width.value),
    )

window_width.observe(render, names="value")
window_start.observe(render, names="value")
sample_spacing.observe(render, names="value")

today_button.on_click(go_today)
maximum_button.on_click(go_maximum)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
clear_button.on_click(clear_markers)

controls = widgets.VBox([
    status,
    measurement,
    window_width,
    window_start,
    sample_spacing,
    widgets.HBox([
        today_button,
        maximum_button,
        previous_button,
        next_button,
        clear_button,
    ]),
])

display(widgets.VBox([controls, output]))
render()
