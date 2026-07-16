# V0001P
# NO AI IMAGES — PYTHON / PANDAS / SCIPY / MATPLOTLIB / IPYWIDGETS ONLY

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy.signal import butter, sosfiltfilt, periodogram, find_peaks
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
    "green": "#35E0A1",
    "magenta": "#FF6EC7",
}

r = requests.get(URL, timeout=300)
r.raise_for_status()

df = pd.read_csv(
    io.StringIO(r.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().sort_values("t_kyr").reset_index(drop=True)
df["age_kyr"] = -df["t_kyr"]
df["z"] = np.hypot(df["p"], df["q"])

MAX_AGE = int(df["age_kyr"].max())
DT = float(np.median(np.diff(np.sort(df["age_kyr"].unique()))))

series = widgets.Dropdown(
    options=[("p(t)", "p"), ("q(t)", "q"), ("|z(t)|", "z")],
    value="z",
    description="Series",
)

age_min = widgets.BoundedIntText(
    value=0,
    min=0,
    max=MAX_AGE,
    description="Age min kyr",
    layout=widgets.Layout(width="240px"),
)

age_max = widgets.BoundedIntText(
    value=min(10000, MAX_AGE),
    min=0,
    max=MAX_AGE,
    description="Age max kyr",
    layout=widgets.Layout(width="240px"),
)

period_min = widgets.BoundedFloatText(
    value=200.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Period min",
    layout=widgets.Layout(width="220px"),
)

period_max = widgets.BoundedFloatText(
    value=350.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Period max",
    layout=widgets.Layout(width="220px"),
)

view = widgets.Dropdown(
    options=[
        "Filtered cycle",
        "Spectrum",
        "Filtered + spectrum",
        "Raw + filtered",
    ],
    value="Filtered + spectrum",
    description="View",
)

full_button = widgets.Button(description="Full 250 Myr", button_style="warning")
ten_myr_button = widgets.Button(description="First 10 Myr")
one_myr_button = widgets.Button(description="First 1 Myr")
render_button = widgets.Button(description="Analyze", button_style="info")

status = widgets.HTML(
    value=(
        f"<b>La2010a loaded:</b> {len(df):,} rows | "
        f"0 to {MAX_AGE:,} kyr before J2000 | cadence {DT:g} kyr"
    )
)

output = widgets.Output()

def set_full(_=None):
    age_min.value = 0
    age_max.value = MAX_AGE

def set_10myr(_=None):
    age_min.value = 0
    age_max.value = min(10000, MAX_AGE)

def set_1myr(_=None):
    age_min.value = 0
    age_max.value = min(1000, MAX_AGE)

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

def bandpass(y, dt, pmin, pmax):
    low_f = 1.0 / pmax
    high_f = 1.0 / pmin
    nyquist = 0.5 / dt
    if high_f >= nyquist:
        raise ValueError("Selected minimum period is below the Nyquist limit.")
    sos = butter(
        4,
        [low_f / nyquist, high_f / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, y)

def spectrum(y, dt):
    f, power = periodogram(
        y,
        fs=1.0 / dt,
        detrend="linear",
        window="hann",
        scaling="spectrum",
    )
    valid = f > 0
    return 1.0 / f[valid], np.sqrt(power[valid])

def render(_=None):
    with output:
        clear_output(wait=True)

        lo, hi = sorted((int(age_min.value), int(age_max.value)))
        sub = df[(df["age_kyr"] >= lo) & (df["age_kyr"] <= hi)].copy()
        sub = sub.sort_values("age_kyr")

        if len(sub) < 64:
            print("Selected interval is too short.")
            return

        key = series.value
        label = {"p": "p(t)", "q": "q(t)", "z": "|z(t)|"}[key]
        color = {"p": C["cyan"], "q": C["violet"], "z": C["gold"]}[key]

        x = sub["age_kyr"].to_numpy(float)
        y = sub[key].to_numpy(float)
        dt = float(np.median(np.diff(x)))

        p_lo, p_hi = sorted((float(period_min.value), float(period_max.value)))
        filtered = bandpass(y, dt, p_lo, p_hi)

        periods, amplitudes = spectrum(y, dt)
        band = (periods >= p_lo) & (periods <= p_hi)
        pp = periods[band]
        aa = amplitudes[band]

        order = np.argsort(pp)
        pp = pp[order]
        aa = aa[order]

        peak_indices, _ = find_peaks(aa)
        if len(peak_indices):
            ranked = peak_indices[np.argsort(aa[peak_indices])[::-1]]
        else:
            ranked = np.argsort(aa)[::-1]

        ranked = ranked[:12]
        peak_table = pd.DataFrame({
            "Rank": np.arange(1, len(ranked) + 1),
            "Period (kyr)": pp[ranked],
            "Frequency (cycles/kyr)": 1.0 / pp[ranked],
            "Spectral amplitude": aa[ranked],
        })

        time_peaks, _ = find_peaks(filtered)
        spacings = np.diff(x[time_peaks]) if len(time_peaks) > 1 else np.array([])
        cycle_table = pd.DataFrame({
            "Metric": [
                "Filtered minimum",
                "Filtered maximum",
                "Filtered peak-to-peak",
                "Filtered RMS",
                "Median peak spacing",
                "Mean peak spacing",
                "Number of detected maxima",
            ],
            "Value": [
                np.min(filtered),
                np.max(filtered),
                np.ptp(filtered),
                np.sqrt(np.mean(filtered**2)),
                np.median(spacings) if len(spacings) else np.nan,
                np.mean(spacings) if len(spacings) else np.nan,
                len(time_peaks),
            ],
        })

        display(peak_table.style.format({
            "Period (kyr)": "{:,.6f}",
            "Frequency (cycles/kyr)": "{:.9e}",
            "Spectral amplitude": "{:.9e}",
        }))
        display(cycle_table.style.format({"Value": "{:.9f}"}))

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

        if view.value in {"Filtered cycle", "Filtered + spectrum", "Raw + filtered"}:
            fig, ax = plt.subplots(figsize=(13, 5.4))
            if view.value == "Raw + filtered":
                ax.plot(x, y, color=C["muted"], linewidth=0.55, alpha=0.45, label="Raw")
            ax.plot(
                x,
                filtered,
                color=color,
                linewidth=1.15,
                label=f"{p_lo:g}–{p_hi:g} kyr band",
            )
            style(
                ax,
                f"La2010a {label}: isolated long cycle",
                "Age before J2000 (kyr)",
                "Filtered amplitude",
            )
            legend = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
            for text in legend.get_texts():
                text.set_color(C["text"])
            fig.tight_layout()
            display(fig)
            plt.close(fig)

        if view.value in {"Spectrum", "Filtered + spectrum"}:
            fig, ax = plt.subplots(figsize=(13, 5.4))
            ax.plot(pp, aa, color=color, linewidth=1.15)
            ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
            ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")
            if len(ranked):
                ax.scatter(
                    pp[ranked],
                    aa[ranked],
                    color=C["magenta"],
                    edgecolor=C["text"],
                    s=38,
                    zorder=5,
                )
            style(
                ax,
                f"La2010a {label}: {p_lo:g}–{p_hi:g} kyr spectrum",
                "Period (kyr)",
                "Spectral amplitude",
            )
            legend = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
            for text in legend.get_texts():
                text.set_color(C["text"])
            fig.tight_layout()
            display(fig)
            plt.close(fig)

full_button.on_click(set_full)
ten_myr_button.on_click(set_10myr)
one_myr_button.on_click(set_1myr)
render_button.on_click(render)

display(
    widgets.VBox([
        status,
        widgets.HBox([series, view, render_button]),
        widgets.HBox([age_min, age_max, one_myr_button, ten_myr_button, full_button]),
        widgets.HBox([period_min, period_max]),
        output,
    ])
)

render()
