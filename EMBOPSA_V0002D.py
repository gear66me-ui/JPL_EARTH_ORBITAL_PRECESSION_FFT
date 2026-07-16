# EMBOPSA_V0002D
# Lightweight La2010a spectral back-test and future explorer.
# Model values are evaluated only for the visible window.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0002D"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)
TRAIN_END_KYR = -50_000.0
N_HARMONICS = 120
FULL_MIN = -250.0
FULL_MAX = 250.0

progress = widgets.HTML("<b>Loading La2010a data…</b>")
display(progress)

response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()
raw = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)
for c in raw.columns:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)
raw["time_myr"] = raw["t_kyr"] / 1000.0
raw["time_years"] = raw["t_kyr"] * 1000.0
raw["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(np.hypot(raw["p"], raw["q"]), 0.0, 1.0))
)

train = raw[raw["t_kyr"] <= TRAIN_END_KYR].copy()
t_train = train["t_kyr"].to_numpy(float)
dt_kyr = float(np.median(np.diff(t_train)))
t_origin = float(t_train[0])
n_train = len(train)

progress.value = "<b>Fitting 120-harmonic p/q model…</b>"

def fit_spectral(y):
    mean = float(np.mean(y))
    coeff = np.fft.rfft(y - mean)
    freq = np.fft.rfftfreq(len(y), d=dt_kyr)
    candidates = np.arange(1, len(coeff))
    strongest = candidates[np.argsort(np.abs(coeff[candidates]))[-N_HARMONICS:]]
    strongest = strongest[np.argsort(freq[strongest])]
    return mean, freq[strongest], coeff[strongest]

p_mean, p_freq, p_coeff = fit_spectral(train["p"].to_numpy(float))
q_mean, q_freq, q_coeff = fit_spectral(train["q"].to_numpy(float))


def evaluate(t_kyr, mean, freq, coeff):
    t_kyr = np.asarray(t_kyr, dtype=float)
    tau = t_kyr - t_origin
    out = np.empty(len(tau), dtype=float)
    chunk = 3000
    for start in range(0, len(tau), chunk):
        stop = min(len(tau), start + chunk)
        phase = np.exp(2j * np.pi * np.outer(tau[start:stop], freq))
        out[start:stop] = mean + (2.0 / n_train) * np.real(phase @ coeff)
    return out


def model_frame(t_kyr):
    p = evaluate(t_kyr, p_mean, p_freq, p_coeff)
    q = evaluate(t_kyr, q_mean, q_freq, q_coeff)
    inc = np.degrees(2.0 * np.arcsin(np.clip(np.hypot(p, q), 0.0, 1.0)))
    return p, q, inc

progress.value = "<b>Ready.</b> Model is computed only for the visible window."

window_size = widgets.Dropdown(
    options=[
        ("100 thousand years", 0.1),
        ("250 thousand years", 0.25),
        ("500 thousand years", 0.5),
        ("1 million years", 1.0),
        ("2 million years", 2.0),
        ("5 million years", 5.0),
        ("10 million years", 10.0),
        ("25 million years", 25.0),
        ("50 million years", 50.0),
        ("100 million years", 100.0),
        ("250 million years", 250.0),
        ("Full 500 million years", 500.0),
    ],
    value=0.1,
    description="Window:",
    layout=widgets.Layout(width="440px"),
)
position = widgets.FloatSlider(
    value=-0.1,
    min=FULL_MIN,
    max=FULL_MAX - 0.1,
    step=0.01,
    description="Start:",
    continuous_update=False,
    readout_format=".2f",
    layout=widgets.Layout(width="800px"),
)
show_actual = widgets.Checkbox(value=True, description="Actual La2010a", indent=False)
show_model = widgets.Checkbox(value=True, description="Spectral model", indent=False)
show_error = widgets.Checkbox(value=False, description="Model − actual error", indent=False)

training_button = widgets.Button(description="Training")
validation_button = widgets.Button(description="Validation")
today_button = widgets.Button(description="Today")
future_button = widgets.Button(description="Future")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
full_button = widgets.Button(description="Full ±250 Myr")
status = widgets.HTML()
plot_output = widgets.Output()


def sync_position_limit():
    width = float(window_size.value)
    position.max = FULL_MAX - width
    if position.value > position.max:
        position.value = position.max


def visible_times(start, end):
    width = end - start
    if width <= 1:
        step_kyr = 1.0
    elif width <= 10:
        step_kyr = 2.0
    elif width <= 50:
        step_kyr = 10.0
    elif width <= 100:
        step_kyr = 20.0
    else:
        step_kyr = 100.0
    return np.arange(start * 1000.0, end * 1000.0 + step_kyr, step_kyr)


def render(*_):
    sync_position_limit()
    start = float(position.value)
    end = min(FULL_MAX, start + float(window_size.value))
    status.value = f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000"
    fig = go.Figure()

    observed = raw[(raw["time_myr"] >= start) & (raw["time_myr"] <= min(end, 0.0))].copy()
    if len(observed) > 6000:
        observed = observed.iloc[::max(1, len(observed)//6000)].copy()

    if show_actual.value and not observed.empty:
        fig.add_trace(go.Scattergl(
            x=observed["time_myr"], y=observed["inclination_deg"],
            customdata=np.column_stack([observed["time_years"], observed["p"], observed["q"]]),
            mode="lines", name="Actual La2010a",
            line=dict(color="#52D6FF", width=2.0),
            hovertemplate=("<b>Actual La2010a</b><br>Time: %{x:.6f} Myr"
                           "<br>Years: %{customdata[0]:,.0f}"
                           "<br>Inclination: %{y:.6f}°"
                           "<br>p: %{customdata[1]:.9f}<br>q: %{customdata[2]:.9f}<extra></extra>"),
        ))

    if show_model.value or show_error.value:
        t_model = visible_times(start, end)
        p_m, q_m, i_m = model_frame(t_model)
        x_m = t_model / 1000.0
        if show_model.value:
            fig.add_trace(go.Scattergl(
                x=x_m, y=i_m,
                customdata=np.column_stack([t_model*1000.0, p_m, q_m]),
                mode="lines", name="Spectral model",
                line=dict(color="#FF6EC7", width=1.8, dash="dash"),
                hovertemplate=("<b>Spectral model</b><br>Time: %{x:.6f} Myr"
                               "<br>Years: %{customdata[0]:,.0f}"
                               "<br>Inclination: %{y:.6f}°<extra></extra>"),
            ))
        if show_error.value and end <= 0.0 and not observed.empty:
            i_interp = np.interp(observed["t_kyr"].to_numpy(float), t_model, i_m)
            fig.add_trace(go.Scattergl(
                x=observed["time_myr"],
                y=i_interp - observed["inclination_deg"].to_numpy(float),
                mode="lines", name="Model − actual error",
                line=dict(color="#FF9F43", width=1.5), yaxis="y2",
                hovertemplate="<b>Error</b><br>Time: %{x:.6f} Myr<br>Error: %{y:+.6f}°<extra></extra>",
            ))

    fig.add_vline(x=-50.0, line=dict(color="#FF9F43", width=1.1, dash="dot"), annotation_text="Training ends")
    fig.add_vline(x=0.0, line=dict(color="#35E0A1", width=1.8), annotation_text="J2000")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000",
        title=dict(text="La2010a vs Spectral Model — Lightweight Explorer", x=0.5),
        xaxis=dict(title="Millions of years relative to J2000", range=[start, end], gridcolor="#28364A",
                   showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#1F7A4D", spikethickness=2),
        yaxis=dict(title="Inclination (degrees)", gridcolor="#28364A",
                   showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#1F7A4D", spikethickness=2),
        yaxis2=dict(title="Error (degrees)", overlaying="y", side="right", visible=bool(show_error.value), showgrid=False),
        hovermode="closest", hoverdistance=100, spikedistance=-1, height=720,
        margin=dict(l=75, r=75, t=80, b=70),
        legend=dict(bgcolor="rgba(0,0,0,0.78)", bordercolor="#28364A", borderwidth=1),
    )
    with plot_output:
        clear_output(wait=True)
        display(fig)


def set_center(center):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, center - width/2.0))

training_button.on_click(lambda _: set_center(-150.0))
validation_button.on_click(lambda _: set_center(-25.0))
today_button.on_click(lambda _: set_center(0.0))
future_button.on_click(lambda _: set_center(25.0))
previous_button.on_click(lambda _: setattr(position, "value", max(FULL_MIN, position.value-window_size.value)))
next_button.on_click(lambda _: setattr(position, "value", min(position.max, position.value+window_size.value)))
full_button.on_click(lambda _: (setattr(window_size, "value", 500.0), setattr(position, "value", FULL_MIN)))

for w in (window_size, position, show_actual, show_model, show_error):
    w.observe(render, names="value")

controls = widgets.VBox([
    status,
    widgets.HTML("<b style='color:#ff9f43'>Future curve is an illustrative spectral continuation, not a precise astronomical forecast.</b>"),
    window_size,
    position,
    widgets.HBox([show_actual, show_model, show_error]),
    widgets.HBox([training_button, validation_button, previous_button, today_button, next_button, future_button, full_button]),
])
display(widgets.VBox([controls, plot_output]))
render()
