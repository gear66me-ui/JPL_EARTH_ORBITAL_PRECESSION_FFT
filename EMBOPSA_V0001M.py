# V0001M
# NO AI-GENERATED IMAGES.
# AUTOMATIC La2010a DOWNLOAD + INTERACTIVE LONG-CYCLE ANALYSIS.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

DATA_URL = (
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
    "green": "#35E0A1",
    "red": "#FF5C7A",
    "magenta": "#FF6EC7",
}

status = widgets.HTML()
output = widgets.Output()

series = widgets.SelectMultiple(
    options=[("p(t)", "p"), ("q(t)", "q"), ("|z(t)|", "z")],
    value=("p",),
    description="Series",
    rows=3,
)

window = widgets.FloatRangeSlider(
    value=[-500, 0],
    min=-1000,
    max=0,
    step=10,
    description="Window kyr",
    continuous_update=False,
    layout=widgets.Layout(width="700px"),
)

period_band = widgets.FloatRangeSlider(
    value=[100, 500],
    min=20,
    max=1000,
    step=10,
    description="Period kyr",
    continuous_update=False,
    layout=widgets.Layout(width="700px"),
)

detrend_degree = widgets.Dropdown(
    options=[("Mean", 0), ("Linear", 1), ("Quadratic", 2), ("Cubic", 3)],
    value=1,
    description="Detrend",
)

view_mode = widgets.Dropdown(
    options=["Time series", "Spectrum", "Both"],
    value="Both",
    description="View",
)

render_button = widgets.Button(
    description="Render",
    button_style="info",
)

STATE = {"df": None}

def load_data():
    status.value = "<b>Downloading La2010a…</b>"
    r = requests.get(DATA_URL, timeout=300)
    r.raise_for_status()

    text = r.text
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t_kyr", "a", "l", "k", "h", "q", "p"],
        engine="python",
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("t_kyr").reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError("La2010a parsing failed: too few numeric rows.")

    df["z"] = np.hypot(df["p"], df["q"])
    STATE["df"] = df

    status.value = (
        f"<b style='color:#35E0A1'>READY:</b> "
        f"{len(df):,} rows | "
        f"{df.t_kyr.min():,.0f} to {df.t_kyr.max():,.0f} kyr | "
        f"cadence ≈ {np.median(np.diff(df.t_kyr)):,.3f} kyr"
    )

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

def detrend(y):
    x = np.arange(len(y), dtype=float)
    degree = int(detrend_degree.value)
    fit = np.polyval(np.polyfit(x, y, degree), x)
    return y - fit

def compute_spectrum(t_kyr, y):
    order = np.argsort(t_kyr)
    t = t_kyr[order]
    y = y[order]

    dt = float(np.median(np.diff(t)))
    grid = np.arange(t.min(), t.max() + dt, dt)
    values = np.interp(grid, t, y)

    values = detrend(values)
    win = np.hanning(len(values))
    spec = np.fft.rfft(values * win)
    freq_per_kyr = np.fft.rfftfreq(len(values), d=dt)
    amp = 2.0 * np.abs(spec) / max(np.sum(win), 1.0)

    good = freq_per_kyr > 0
    periods_kyr = 1.0 / freq_per_kyr[good]
    return periods_kyr, amp[good]

def render(_=None):
    with output:
        clear_output(wait=True)

        if STATE["df"] is None:
            try:
                load_data()
            except Exception as exc:
                print(f"REJECTED: {exc}")
                return

        df = STATE["df"]
        lo, hi = window.value
        subset = df[(df.t_kyr >= lo) & (df.t_kyr <= hi)].copy()

        if len(subset) < 20:
            print("Selected time window contains too few points.")
            return

        selected = list(series.value) or ["p"]

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

        colors = {"p": C["cyan"], "q": C["violet"], "z": C["gold"]}
        labels = {"p": "p(t)", "q": "q(t)", "z": "|z(t)|"}

        for key in selected:
            y = subset[key].to_numpy(float)
            t = subset.t_kyr.to_numpy(float)

            if view_mode.value in {"Time series", "Both"}:
                fig, ax = plt.subplots(figsize=(12.5, 5.2))
                ax.plot(t, y, color=colors[key], linewidth=1.0)
                ax.axvline(0, color=C["red"], linestyle="--", linewidth=0.9)
                style(
                    ax,
                    f"La2010a {labels[key]}",
                    "Time from J2000 (kyr)",
                    labels[key],
                )
                fig.tight_layout()
                display(fig)
                plt.close(fig)

            if view_mode.value in {"Spectrum", "Both"}:
                periods, amplitudes = compute_spectrum(t, y)
                p_lo, p_hi = period_band.value
                mask = (periods >= p_lo) & (periods <= p_hi)

                pp = periods[mask]
                aa = amplitudes[mask]
                order = np.argsort(pp)
                pp, aa = pp[order], aa[order]

                fig, ax = plt.subplots(figsize=(12.5, 5.2))
                ax.plot(pp, aa, color=colors[key], linewidth=1.15)
                ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
                ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")
                style(
                    ax,
                    f"La2010a {labels[key]} spectrum",
                    "Period (kyr)",
                    "Amplitude",
                )
                leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
                for text in leg.get_texts():
                    text.set_color(C["text"])
                fig.tight_layout()
                display(fig)
                plt.close(fig)

                if len(aa):
                    idx = np.argsort(aa)[::-1][:12]
                    peaks = pd.DataFrame({
                        "rank": np.arange(1, len(idx) + 1),
                        "period_kyr": pp[idx],
                        "amplitude": aa[idx],
                    })
                    display(
                        peaks.style.format({
                            "period_kyr": "{:,.3f}",
                            "amplitude": "{:.9e}",
                        })
                    )

render_button.on_click(render)

display(
    widgets.VBox([
        status,
        widgets.HBox([
            series,
            widgets.VBox([view_mode, detrend_degree, render_button]),
        ]),
        window,
        period_band,
        output,
    ])
)

render()
# V0001M
