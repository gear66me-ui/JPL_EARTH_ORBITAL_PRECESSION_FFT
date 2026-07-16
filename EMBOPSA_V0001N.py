# V0001N
# NO AI IMAGES — PYTHON / PANDAS / MATPLOTLIB / IPYWIDGETS ONLY

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"

C = {
    "bg":"#000000","grid":"#29364D","text":"#F4F7FB","muted":"#AAB4C3",
    "cyan":"#52D6FF","violet":"#B388FF","gold":"#FFD166","red":"#FF5C7A"
}

r = requests.get(URL, timeout=300)
r.raise_for_status()

df = pd.read_csv(
    io.StringIO(r.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr","a","l","k","h","q","p"],
    engine="python"
)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().sort_values("t_kyr").reset_index(drop=True)
df["z"] = np.hypot(df["p"], df["q"])

tmin = float(df["t_kyr"].min())
tmax = float(df["t_kyr"].max())
step = float(np.median(np.diff(df["t_kyr"])))

series = widgets.SelectMultiple(
    options=[("p(t)","p"),("q(t)","q"),("|z(t)|","z")],
    value=("p",),
    description="Series",
    rows=3
)

window = widgets.FloatRangeSlider(
    value=[max(tmin, -1000.0), min(tmax, 0.0)],
    min=tmin,
    max=tmax,
    step=max(step, 1.0),
    description="Window kyr",
    continuous_update=False,
    readout_format=".0f",
    layout=widgets.Layout(width="760px")
)

view = widgets.Dropdown(
    options=["Time series","Spectrum","Both"],
    value="Time series",
    description="View"
)

period_band = widgets.FloatRangeSlider(
    value=[100,500],
    min=10,
    max=1000,
    step=10,
    description="Period kyr",
    continuous_update=False,
    layout=widgets.Layout(width="760px")
)

render_button = widgets.Button(description="Render", button_style="info")
status = widgets.HTML(
    value=(
        f"<b>La2010a:</b> {len(df):,} rows | "
        f"{tmin:,.0f} to {tmax:,.0f} kyr | cadence {step:,.0f} kyr"
    )
)
out = widgets.Output()

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=.6, alpha=.75)
    for s in ax.spines.values():
        s.set_color(C["grid"])

def spectrum(t, y):
    dt = float(np.median(np.diff(t)))
    yd = y - np.polyval(np.polyfit(t - t.mean(), y, 1), t - t.mean())
    w = np.hanning(len(yd))
    spec = np.fft.rfft(yd * w)
    f = np.fft.rfftfreq(len(yd), d=dt)
    amp = 2.0 * np.abs(spec) / max(np.sum(w), 1.0)
    good = f > 0
    return 1.0 / f[good], amp[good]

def render(_=None):
    with out:
        clear_output(wait=True)

        lo, hi = window.value
        sub = df[(df["t_kyr"] >= lo) & (df["t_kyr"] <= hi)].copy()
        if len(sub) < 8:
            print("Selected interval contains too few rows.")
            return

        selected = list(series.value) or ["p"]
        colors = {"p":C["cyan"],"q":C["violet"],"z":C["gold"]}
        labels = {"p":"p(t)","q":"q(t)","z":"|z(t)|"}

        stats = []
        for key in selected:
            y = sub[key].to_numpy(float)
            stats.append({
                "series": labels[key],
                "minimum": float(np.min(y)),
                "maximum": float(np.max(y)),
                "absolute_maximum": float(np.max(np.abs(y))),
                "peak_to_peak": float(np.ptp(y)),
                "rms": float(np.sqrt(np.mean(y*y))),
                "time_of_min_kyr": float(sub.iloc[int(np.argmin(y))]["t_kyr"]),
                "time_of_max_kyr": float(sub.iloc[int(np.argmax(y))]["t_kyr"]),
            })

        display(pd.DataFrame(stats).style.format({
            "minimum":"{:.9f}",
            "maximum":"{:.9f}",
            "absolute_maximum":"{:.9f}",
            "peak_to_peak":"{:.9f}",
            "rms":"{:.9f}",
            "time_of_min_kyr":"{:,.0f}",
            "time_of_max_kyr":"{:,.0f}",
        }))

        plt.close("all")
        plt.ioff()
        plt.rcParams.update({
            "figure.facecolor":C["bg"],"axes.facecolor":C["bg"],
            "savefig.facecolor":C["bg"],"text.color":C["text"],
            "axes.labelcolor":C["text"],"xtick.color":C["text"],
            "ytick.color":C["text"],"font.family":"DejaVu Sans"
        })

        for key in selected:
            t = sub["t_kyr"].to_numpy(float)
            y = sub[key].to_numpy(float)

            if view.value in {"Time series","Both"}:
                fig, ax = plt.subplots(figsize=(13,5.4))
                ax.plot(t, y, color=colors[key], linewidth=1.0)
                ax.axvline(0, color=C["red"], linestyle="--", linewidth=.9)
                style(ax, f"La2010a {labels[key]}", "Time from J2000 (kyr)", labels[key])
                fig.tight_layout()
                display(fig)
                plt.close(fig)

            if view.value in {"Spectrum","Both"}:
                period, amp = spectrum(t, y)
                plo, phi = period_band.value
                m = (period >= plo) & (period <= phi)
                pp, aa = period[m], amp[m]
                order = np.argsort(pp)
                pp, aa = pp[order], aa[order]

                fig, ax = plt.subplots(figsize=(13,5.4))
                ax.plot(pp, aa, color=colors[key], linewidth=1.15)
                ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
                ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")
                style(ax, f"La2010a {labels[key]} spectrum", "Period (kyr)", "Amplitude")
                leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
                for txt in leg.get_texts():
                    txt.set_color(C["text"])
                fig.tight_layout()
                display(fig)
                plt.close(fig)

render_button.on_click(render)

display(widgets.VBox([
    status,
    widgets.HBox([series, widgets.VBox([view, render_button])]),
    window,
    period_band,
    out
]))

render()
