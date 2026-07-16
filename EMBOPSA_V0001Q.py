# V0001Q
# NO AI-GENERATED IMAGES.
# PYTHON / NUMPY / PANDAS / SCIPY / MATPLOTLIB / IPYWIDGETS ONLY.

from __future__ import annotations

import io
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import clear_output, display
from scipy.signal import find_peaks

VERSION = "V0001Q"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

OUT = Path("/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001Q_OUTPUT")
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "bg": "#000000",
    "grid": "#29364D",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "violet": "#B388FF",
    "gold": "#FFD166",
    "green": "#35E0A1",
    "red": "#FF5C7A",
    "magenta": "#FF6EC7",
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

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().sort_values("t_kyr").reset_index(drop=True)
df["age_kyr"] = -df["t_kyr"]
df["z_abs"] = np.hypot(df["p"], df["q"])

MAX_AGE = int(df["age_kyr"].max())
DT = float(np.median(np.diff(np.sort(df["age_kyr"].unique()))))

signal_dd = widgets.Dropdown(
    options=[
        ("p(t)", "p"),
        ("q(t)", "q"),
        ("complex z(t)=p+iq", "z_complex"),
        ("|z(t)|", "z_abs"),
    ],
    value="z_complex",
    description="Signal",
)

age_min = widgets.BoundedIntText(
    value=0,
    min=0,
    max=MAX_AGE,
    description="Age min kyr",
    layout=widgets.Layout(width="240px"),
)

age_max = widgets.BoundedIntText(
    value=MAX_AGE,
    min=0,
    max=MAX_AGE,
    description="Age max kyr",
    layout=widgets.Layout(width="240px"),
)

detrend_dd = widgets.Dropdown(
    options=[("Mean", 0), ("Linear", 1), ("Quadratic", 2), ("Cubic", 3)],
    value=1,
    description="Detrend",
)

window_dd = widgets.Dropdown(
    options=["Hann", "Blackman", "None"],
    value="Hann",
    description="Window",
)

peak_count = widgets.IntSlider(
    value=30,
    min=5,
    max=100,
    step=5,
    description="Peaks",
    continuous_update=False,
    layout=widgets.Layout(width="380px"),
)

period_min = widgets.BoundedFloatText(
    value=10.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Period min",
    layout=widgets.Layout(width="220px"),
)

period_max = widgets.BoundedFloatText(
    value=1000.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Period max",
    layout=widgets.Layout(width="220px"),
)

zoom_min = widgets.BoundedFloatText(
    value=200.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Zoom min",
    layout=widgets.Layout(width="220px"),
)

zoom_max = widgets.BoundedFloatText(
    value=350.0,
    min=2.0,
    max=MAX_AGE,
    step=1.0,
    description="Zoom max",
    layout=widgets.Layout(width="220px"),
)

full_button = widgets.Button(description="Full 250 Myr", button_style="warning")
ten_myr_button = widgets.Button(description="First 10 Myr")
one_myr_button = widgets.Button(description="First 1 Myr")
run_button = widgets.Button(description="Run FFT", button_style="info")

status = widgets.HTML(
    value=(
        f"<b>La2010a:</b> {len(df):,} rows | "
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

def make_window(n: int) -> np.ndarray:
    if window_dd.value == "Hann":
        return np.hanning(n)
    if window_dd.value == "Blackman":
        return np.blackman(n)
    return np.ones(n)

def detrend_series(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    centered = x - np.mean(x)
    scale = max(np.ptp(centered), 1.0)
    u = centered / scale

    if np.iscomplexobj(y):
        real_fit = np.polyval(np.polyfit(u, y.real, degree), u)
        imag_fit = np.polyval(np.polyfit(u, y.imag, degree), u)
        return y - (real_fit + 1j * imag_fit)

    fit = np.polyval(np.polyfit(u, y, degree), u)
    return y - fit

def fft_extract(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    dt = float(np.median(np.diff(x)))
    grid = np.arange(x.min(), x.max() + dt, dt)

    if np.iscomplexobj(y):
        yr = np.interp(grid, x, y.real)
        yi = np.interp(grid, x, y.imag)
        values = yr + 1j * yi
    else:
        values = np.interp(grid, x, y)

    values = detrend_series(grid, values, int(detrend_dd.value))
    w = make_window(len(values))
    weighted = values * w

    if np.iscomplexobj(weighted):
        spectrum = np.fft.fft(weighted)
        freq = np.fft.fftfreq(len(weighted), d=dt)
        positive = freq > 0
        freq = freq[positive]
        spectrum = spectrum[positive]
        amplitude = np.abs(spectrum) / max(np.sum(w), 1.0)
    else:
        spectrum = np.fft.rfft(weighted)
        freq = np.fft.rfftfreq(len(weighted), d=dt)
        positive = freq > 0
        freq = freq[positive]
        spectrum = spectrum[positive]
        amplitude = 2.0 * np.abs(spectrum) / max(np.sum(w), 1.0)

    period = 1.0 / freq
    phase = np.angle(spectrum)

    return pd.DataFrame({
        "frequency_cycles_per_kyr": freq,
        "period_kyr": period,
        "amplitude": amplitude,
        "phase_rad": phase,
        "real_component": spectrum.real,
        "imag_component": spectrum.imag,
    })

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

def run_fft(_=None):
    with output:
        clear_output(wait=True)

        lo, hi = sorted((int(age_min.value), int(age_max.value)))
        sub = df[(df["age_kyr"] >= lo) & (df["age_kyr"] <= hi)].copy()
        sub = sub.sort_values("age_kyr")

        if len(sub) < 64:
            print("Selected interval is too short.")
            return

        x = sub["age_kyr"].to_numpy(dtype=float)

        key = signal_dd.value
        if key == "z_complex":
            y = (
                sub["p"].to_numpy(dtype=float)
                + 1j * sub["q"].to_numpy(dtype=float)
            )
            label = "complex z(t)=p+iq"
            color = C["green"]
        else:
            y = sub[key].to_numpy(dtype=float)
            label = {
                "p": "p(t)",
                "q": "q(t)",
                "z_abs": "|z(t)|",
            }[key]
            color = {
                "p": C["cyan"],
                "q": C["violet"],
                "z_abs": C["gold"],
            }[key]

        fft_df = fft_extract(x, y)

        p_lo, p_hi = sorted((float(period_min.value), float(period_max.value)))
        band = fft_df[
            (fft_df["period_kyr"] >= p_lo)
            & (fft_df["period_kyr"] <= p_hi)
        ].copy()

        if band.empty:
            print("No FFT bins fall inside the requested period range.")
            return

        band = band.sort_values("period_kyr").reset_index(drop=True)

        peak_idx, _ = find_peaks(band["amplitude"].to_numpy())
        if len(peak_idx) == 0:
            ranked = band.nlargest(int(peak_count.value), "amplitude").copy()
        else:
            peaks = band.iloc[peak_idx].copy()
            ranked = peaks.nlargest(int(peak_count.value), "amplitude").copy()

        ranked = ranked.reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked["period_myr"] = ranked["period_kyr"] / 1000.0
        ranked["frequency_cycles_per_year"] = (
            ranked["frequency_cycles_per_kyr"] / 1000.0
        )

        csv_path = OUT / f"EMBOPSA_LA2010_FFT_PEAKS_{key}_{VERSION}.csv"
        ranked.to_csv(csv_path, index=False)

        display(
            ranked[
                [
                    "rank",
                    "period_kyr",
                    "period_myr",
                    "frequency_cycles_per_kyr",
                    "frequency_cycles_per_year",
                    "amplitude",
                    "phase_rad",
                ]
            ].style.format({
                "period_kyr": "{:,.6f}",
                "period_myr": "{:,.9f}",
                "frequency_cycles_per_kyr": "{:.12e}",
                "frequency_cycles_per_year": "{:.12e}",
                "amplitude": "{:.12e}",
                "phase_rad": "{:.9f}",
            })
        )

        print(csv_path)

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

        fig, ax = plt.subplots(figsize=(13, 5.6))
        ax.plot(
            band["period_kyr"],
            band["amplitude"],
            color=color,
            linewidth=1.05,
        )
        ax.set_xscale("log")
        style(
            ax,
            f"La2010a FFT — {label}",
            "Period (kyr, log scale)",
            "Amplitude",
        )
        fig.tight_layout()
        display(fig)
        plt.close(fig)

        z_lo, z_hi = sorted((float(zoom_min.value), float(zoom_max.value)))
        zoom = fft_df[
            (fft_df["period_kyr"] >= z_lo)
            & (fft_df["period_kyr"] <= z_hi)
        ].sort_values("period_kyr")

        if not zoom.empty:
            fig, ax = plt.subplots(figsize=(13, 5.6))
            ax.plot(
                zoom["period_kyr"],
                zoom["amplitude"],
                color=color,
                linewidth=1.15,
            )
            ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
            ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")

            zoom_peak_idx, _ = find_peaks(zoom["amplitude"].to_numpy())
            if len(zoom_peak_idx):
                ax.scatter(
                    zoom.iloc[zoom_peak_idx]["period_kyr"],
                    zoom.iloc[zoom_peak_idx]["amplitude"],
                    color=C["magenta"],
                    edgecolor=C["text"],
                    s=34,
                    zorder=5,
                )

            style(
                ax,
                f"La2010a FFT zoom — {label}",
                "Period (kyr)",
                "Amplitude",
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
run_button.on_click(run_fft)

display(
    widgets.VBox([
        status,
        widgets.HBox([signal_dd, detrend_dd, window_dd, run_button]),
        widgets.HBox([age_min, age_max, one_myr_button, ten_myr_button, full_button]),
        widgets.HBox([period_min, period_max, zoom_min, zoom_max]),
        peak_count,
        output,
    ])
)

run_fft()
# V0001Q
