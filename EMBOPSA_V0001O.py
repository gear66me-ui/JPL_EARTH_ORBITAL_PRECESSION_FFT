# V0001O
# NO AI IMAGES — PYTHON / PANDAS / MATPLOTLIB / IPYWIDGETS ONLY

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

C = {
    "bg": "#000000",
    "grid": "#29364D",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "violet": "#B388FF",
    "gold": "#FFD166",
    "red": "#FF5C7A",
}

response = requests.get(URL, timeout=300)
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
df["z"] = np.hypot(df["p"], df["q"])
df["age_kyr"] = -df["t_kyr"]

T_MIN = int(np.floor(df["t_kyr"].min()))
T_MAX = int(np.ceil(df["t_kyr"].max()))
MAX_AGE = int(np.ceil(df["age_kyr"].max()))
CADENCE = int(round(np.median(np.diff(np.sort(df["age_kyr"].unique())))))

series = widgets.SelectMultiple(
    options=[("p(t)", "p"), ("q(t)", "q"), ("|z(t)|", "z")],
    value=("p",),
    description="Series",
    rows=3,
)

axis_mode = widgets.ToggleButtons(
    options=[
        ("Signed time", "signed"),
        ("Age before J2000", "age"),
    ],
    value="signed",
    description="Axis",
)

start_box = widgets.BoundedIntText(
    value=-1000,
    min=T_MIN,
    max=T_MAX,
    step=1,
    description="Start kyr",
    layout=widgets.Layout(width="240px"),
)

end_box = widgets.BoundedIntText(
    value=0,
    min=T_MIN,
    max=T_MAX,
    step=1,
    description="End kyr",
    layout=widgets.Layout(width="240px"),
)

full_button = widgets.Button(
    description="Full 250 Myr",
    button_style="warning",
)

recent_button = widgets.Button(
    description="Last 1 Myr",
    button_style="",
)

view = widgets.Dropdown(
    options=["Time series", "Spectrum", "Both"],
    value="Time series",
    description="View",
)

period_low = widgets.BoundedIntText(
    value=100,
    min=2,
    max=MAX_AGE,
    description="Period min",
    layout=widgets.Layout(width="220px"),
)

period_high = widgets.BoundedIntText(
    value=500,
    min=2,
    max=MAX_AGE,
    description="Period max",
    layout=widgets.Layout(width="220px"),
)

render_button = widgets.Button(
    description="Render",
    button_style="info",
)

status = widgets.HTML(
    value=(
        f"<b>La2010a:</b> {len(df):,} rows | "
        f"signed range {T_MIN:,} to {T_MAX:,} kyr | "
        f"maximum age {MAX_AGE:,} kyr "
        f"({MAX_AGE / 1000:,.3f} Myr) | cadence {CADENCE} kyr"
    )
)

output = widgets.Output()

def configure_boxes(*_):
    if axis_mode.value == "signed":
        start_box.description = "Start kyr"
        end_box.description = "End kyr"
        start_box.min = T_MIN
        start_box.max = T_MAX
        end_box.min = T_MIN
        end_box.max = T_MAX
        start_box.value = max(T_MIN, -1000)
        end_box.value = 0
    else:
        start_box.description = "Age min kyr"
        end_box.description = "Age max kyr"
        start_box.min = 0
        start_box.max = MAX_AGE
        end_box.min = 0
        end_box.max = MAX_AGE
        start_box.value = 0
        end_box.value = min(1000, MAX_AGE)

def set_full(_=None):
    if axis_mode.value == "signed":
        start_box.value = T_MIN
        end_box.value = T_MAX
    else:
        start_box.value = 0
        end_box.value = MAX_AGE

def set_recent(_=None):
    if axis_mode.value == "signed":
        start_box.value = max(T_MIN, -1000)
        end_box.value = 0
    else:
        start_box.value = 0
        end_box.value = min(1000, MAX_AGE)

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

def compute_spectrum(t_kyr, y):
    order = np.argsort(t_kyr)
    t = t_kyr[order]
    values = y[order]

    dt = float(np.median(np.diff(t)))
    centered_t = t - np.mean(t)
    trend = np.polyval(np.polyfit(centered_t, values, 1), centered_t)
    residual = values - trend

    window_function = np.hanning(len(residual))
    spectrum_values = np.fft.rfft(residual * window_function)
    frequency = np.fft.rfftfreq(len(residual), d=dt)
    amplitude = 2.0 * np.abs(spectrum_values) / max(np.sum(window_function), 1.0)

    valid = frequency > 0
    return 1.0 / frequency[valid], amplitude[valid]

def selected_subset():
    a = int(start_box.value)
    b = int(end_box.value)
    lo, hi = sorted((a, b))

    if axis_mode.value == "signed":
        subset = df[(df["t_kyr"] >= lo) & (df["t_kyr"] <= hi)].copy()
        x_column = "t_kyr"
        xlabel = "Time from J2000 (kyr)"
    else:
        subset = df[(df["age_kyr"] >= lo) & (df["age_kyr"] <= hi)].copy()
        x_column = "age_kyr"
        xlabel = "Age before J2000 (kyr)"

    return subset.sort_values(x_column), x_column, xlabel

def render(_=None):
    with output:
        clear_output(wait=True)

        subset, x_column, xlabel = selected_subset()
        if len(subset) < 8:
            print("Selected interval contains too few rows.")
            return

        selected = list(series.value) or ["p"]
        labels = {"p": "p(t)", "q": "q(t)", "z": "|z(t)|"}
        colors = {"p": C["cyan"], "q": C["violet"], "z": C["gold"]}

        audit_rows = []
        for key in selected:
            y = subset[key].to_numpy(dtype=float)
            i_min = int(np.argmin(y))
            i_max = int(np.argmax(y))
            audit_rows.append({
                "Series": labels[key],
                "Minimum": y[i_min],
                "Maximum": y[i_max],
                "Absolute maximum": np.max(np.abs(y)),
                "Peak-to-peak": np.ptp(y),
                "RMS": np.sqrt(np.mean(y * y)),
                "Epoch of minimum (kyr)": subset.iloc[i_min][x_column],
                "Epoch of maximum (kyr)": subset.iloc[i_max][x_column],
            })

        display(
            pd.DataFrame(audit_rows).style.format({
                "Minimum": "{:.9f}",
                "Maximum": "{:.9f}",
                "Absolute maximum": "{:.9f}",
                "Peak-to-peak": "{:.9f}",
                "RMS": "{:.9f}",
                "Epoch of minimum (kyr)": "{:,.0f}",
                "Epoch of maximum (kyr)": "{:,.0f}",
            })
        )

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

        x = subset[x_column].to_numpy(dtype=float)
        signed_t = subset["t_kyr"].to_numpy(dtype=float)

        for key in selected:
            y = subset[key].to_numpy(dtype=float)

            if view.value in {"Time series", "Both"}:
                fig, ax = plt.subplots(figsize=(13, 5.4))
                ax.plot(x, y, color=colors[key], linewidth=0.95)
                if axis_mode.value == "signed":
                    ax.axvline(0, color=C["red"], linestyle="--", linewidth=0.9)
                style(ax, f"La2010a {labels[key]}", xlabel, labels[key])
                fig.tight_layout()
                display(fig)
                plt.close(fig)

            if view.value in {"Spectrum", "Both"}:
                periods, amplitudes = compute_spectrum(signed_t, y)
                p_lo, p_hi = sorted((int(period_low.value), int(period_high.value)))
                mask = (periods >= p_lo) & (periods <= p_hi)
                pp = periods[mask]
                aa = amplitudes[mask]
                order = np.argsort(pp)
                pp = pp[order]
                aa = aa[order]

                fig, ax = plt.subplots(figsize=(13, 5.4))
                ax.plot(pp, aa, color=colors[key], linewidth=1.1)
                ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
                ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")
                style(ax, f"La2010a {labels[key]} spectrum", "Period (kyr)", "Amplitude")
                legend = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
                for text in legend.get_texts():
                    text.set_color(C["text"])
                fig.tight_layout()
                display(fig)
                plt.close(fig)

                if len(aa):
                    indices = np.argsort(aa)[::-1][:12]
                    peaks = pd.DataFrame({
                        "Rank": np.arange(1, len(indices) + 1),
                        "Period (kyr)": pp[indices],
                        "Amplitude": aa[indices],
                    })
                    display(peaks.style.format({
                        "Period (kyr)": "{:,.3f}",
                        "Amplitude": "{:.9e}",
                    }))

axis_mode.observe(configure_boxes, names="value")
full_button.on_click(set_full)
recent_button.on_click(set_recent)
render_button.on_click(render)

display(
    widgets.VBox([
        status,
        widgets.HBox([axis_mode, view]),
        widgets.HBox([start_box, end_box, recent_button, full_button]),
        widgets.HBox([series, widgets.VBox([period_low, period_high, render_button])]),
        output,
    ])
)

render()
