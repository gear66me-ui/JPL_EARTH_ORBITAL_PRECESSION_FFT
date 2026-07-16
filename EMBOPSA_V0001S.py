# EMBOPSA_V0001S
# NO AI-GENERATED IMAGES.
# Deterministic Python / NumPy / Pandas / Matplotlib / ipywidgets only.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0001S"

DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

C = {
    "bg": "#000000",
    "grid": "#28364A",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "gold": "#FFD166",
    "red": "#FF5C7A",
    "green": "#35E0A1",
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

today_row = df.loc[df["t_kyr"].abs().idxmin()]
TODAY_I = float(today_row["inclination_deg"])

max_row = df.loc[df["inclination_deg"].idxmax()]
GLOBAL_MAX_I = float(max_row["inclination_deg"])
GLOBAL_MAX_AGE = float(max_row["age_myr"])

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
global_max_button = widgets.Button(description="Go to maximum")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
full_button = widgets.Button(description="Full 250 Myr", button_style="warning")
render_button = widgets.Button(description="Update plot", button_style="info")

status = widgets.HTML()
output = widgets.Output()

def sync_position_limit(*_):
    width = float(window_width.value)
    window_start.max = max(0.0, MAX_AGE - width)
    if window_start.value > window_start.max:
        window_start.value = window_start.max

def go_today(_=None):
    window_start.value = 0.0

def go_global_max(_=None):
    width = float(window_width.value)
    target = max(0.0, GLOBAL_MAX_AGE - width / 2.0)
    window_start.value = min(target, window_start.max)

def go_previous(_=None):
    width = float(window_width.value)
    window_start.value = max(0.0, window_start.value - width)

def go_next(_=None):
    width = float(window_width.value)
    window_start.value = min(window_start.max, window_start.value + width)

def go_full(_=None):
    window_width.value = MAX_AGE
    window_start.value = 0.0
    sample_spacing.value = 100

def choose_stride(spacing_kyr: int) -> int:
    return max(1, int(round(spacing_kyr)))

def render(_=None):
    sync_position_limit()

    start = float(window_start.value)
    width = float(window_width.value)
    end = min(MAX_AGE, start + width)

    subset = df[
        (df["age_myr"] >= start)
        & (df["age_myr"] <= end)
    ].copy()

    if subset.empty:
        with output:
            clear_output(wait=True)
            print("No data in the selected window.")
        return

    stride = choose_stride(int(sample_spacing.value))
    plotted = subset.iloc[::stride].copy()

    local_max_row = subset.loc[subset["inclination_deg"].idxmax()]
    local_max_i = float(local_max_row["inclination_deg"])
    local_max_age = float(local_max_row["age_myr"])

    local_min_row = subset.loc[subset["inclination_deg"].idxmin()]
    local_min_i = float(local_min_row["inclination_deg"])
    local_min_age = float(local_min_row["age_myr"])

    status.value = (
        f"<b>Window:</b> {start:,.3f}–{end:,.3f} Myr before J2000 &nbsp; | &nbsp; "
        f"<b>Width:</b> {end-start:,.3f} Myr &nbsp; | &nbsp; "
        f"<b>Points shown:</b> {len(plotted):,}"
    )

    with output:
        clear_output(wait=True)

        plt.close("all")
        plt.ioff()
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
            linestyle="--",
            linewidth=1.15,
            label=f"J2000 inclination = {TODAY_I:.6f}°",
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
            s=65,
            zorder=5,
        )

        ax.annotate(
            f"Window maximum\n{local_max_i:.6f}° at {local_max_age:.3f} Myr",
            xy=(local_max_age, local_max_i),
            xytext=(14, 18),
            textcoords="offset points",
            color=C["text"],
            fontsize=10,
            arrowprops=dict(color=C["green"], width=0.7, headwidth=5),
            bbox=dict(facecolor=C["bg"], edgecolor=C["grid"], alpha=0.9),
        )

        ax.set_title(
            "Earth Orbital-Plane Inclination Window Explorer",
            fontsize=17,
            fontweight="bold",
            pad=16,
        )

        ax.set_xlabel("Millions of years before J2000")
        ax.set_ylabel("Inclination (degrees)")
        ax.set_xlim(start, end if end > start else start + 0.001)

        y_low = max(0.0, local_min_i - 0.2)
        y_high = max(3.15, local_max_i + 0.2)
        ax.set_ylim(y_low, y_high)

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

        summary = (
            f"Window minimum : {local_min_i:.6f}° at {local_min_age:.3f} Myr\n"
            f"Window maximum : {local_max_i:.6f}° at {local_max_age:.3f} Myr\n"
            f"J2000 value    : {TODAY_I:.6f}°\n"
            f"Global maximum : {GLOBAL_MAX_I:.6f}° at {GLOBAL_MAX_AGE:.3f} Myr"
        )

        ax.text(
            0.015,
            0.025,
            summary,
            transform=ax.transAxes,
            color=C["text"],
            fontsize=10,
            va="bottom",
            ha="left",
            family="monospace",
            bbox=dict(facecolor=C["bg"], edgecolor=C["grid"], alpha=0.88),
        )

        fig.tight_layout()
        display(fig)
        plt.close(fig)

window_width.observe(sync_position_limit, names="value")
window_width.observe(render, names="value")
window_start.observe(render, names="value")
sample_spacing.observe(render, names="value")

today_button.on_click(go_today)
global_max_button.on_click(go_global_max)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
full_button.on_click(go_full)
render_button.on_click(render)

controls = widgets.VBox([
    status,
    window_width,
    window_start,
    sample_spacing,
    widgets.HBox([
        today_button,
        global_max_button,
        previous_button,
        next_button,
        full_button,
        render_button,
    ]),
])

display(widgets.VBox([controls, output]))
render()
