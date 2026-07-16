# EMBOPSA_V0002A
# La2010a past + illustrative spectral continuation to +250 Myr.
# NO AI-GENERATED IMAGES.
# The future curve is a mathematical Fourier continuation, not a reliable prediction.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0002A"

DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

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

t_past_kyr = df["t_kyr"].to_numpy(dtype=float)
p_past = df["p"].to_numpy(dtype=float)
q_past = df["q"].to_numpy(dtype=float)

pole_radius = np.hypot(p_past, q_past)
i_past = np.degrees(2.0 * np.arcsin(np.clip(pole_radius, 0.0, 1.0)))

TODAY_I = float(i_past[-1])

# -------------------------------------------------------------------------
# Illustrative spectral continuation
# -------------------------------------------------------------------------
# Retain the strongest Fourier components of p(t) and q(t), then evaluate
# those harmonics forward. This preserves the broad quasiperiodic texture,
# but it cannot preserve the true chaotic phase evolution for 250 Myr.
N_HARMONICS = 160
N = len(t_past_kyr)
dt_kyr = float(np.median(np.diff(t_past_kyr)))
t0_kyr = float(t_past_kyr[0])

def spectral_model(y: np.ndarray, n_harmonics: int):
    mean = float(np.mean(y))
    centered = y - mean
    coeff = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(y), d=dt_kyr)

    usable = np.arange(1, len(coeff))
    strongest = usable[np.argsort(np.abs(coeff[usable]))[-n_harmonics:]]
    strongest = strongest[np.argsort(freq[strongest])]

    return mean, freq[strongest], coeff[strongest]

p_mean, p_freq, p_coeff = spectral_model(p_past, N_HARMONICS)
q_mean, q_freq, q_coeff = spectral_model(q_past, N_HARMONICS)

def evaluate_spectral(
    t_kyr: np.ndarray,
    mean: float,
    freq: np.ndarray,
    coeff: np.ndarray,
    chunk_size: int = 10000,
) -> np.ndarray:
    out = np.empty(len(t_kyr), dtype=float)
    tau = t_kyr - t0_kyr

    for start in range(0, len(t_kyr), chunk_size):
        stop = min(len(t_kyr), start + chunk_size)
        phase = np.exp(
            2j * np.pi * np.outer(tau[start:stop], freq)
        )
        out[start:stop] = (
            mean
            + (2.0 / N) * np.real(phase @ coeff)
        )

    return out

# Future displayed at 10-kyr cadence: enough for a 250-Myr overview while
# retaining the main inclination oscillations.
t_future_kyr = np.arange(0.0, 250000.0 + 10.0, 10.0)
p_future = evaluate_spectral(t_future_kyr, p_mean, p_freq, p_coeff)
q_future = evaluate_spectral(t_future_kyr, q_mean, q_freq, q_coeff)

future_radius = np.hypot(p_future, q_future)
i_future = np.degrees(
    2.0 * np.arcsin(np.clip(future_radius, 0.0, 1.0))
)

past = pd.DataFrame({
    "time_myr": t_past_kyr / 1000.0,
    "time_years": t_past_kyr * 1000.0,
    "inclination_deg": i_past,
    "p": p_past,
    "q": q_past,
})

future = pd.DataFrame({
    "time_myr": t_future_kyr / 1000.0,
    "time_years": t_future_kyr * 1000.0,
    "inclination_deg": i_future,
    "p": p_future,
    "q": q_future,
})

FULL_MIN = -250.0
FULL_MAX = 250.0

window_size = widgets.Dropdown(
    options=[
        ("0.5 million years", 0.5),
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
    value=10.0,
    description="Window:",
    layout=widgets.Layout(width="430px"),
)

position = widgets.FloatSlider(
    value=-10.0,
    min=FULL_MIN,
    max=FULL_MAX - 10.0,
    step=0.1,
    description="Start:",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="790px"),
)

past_button = widgets.Button(description="−250 Myr")
today_button = widgets.Button(description="Today")
future_button = widgets.Button(description="+250 Myr")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
full_button = widgets.Button(description="Full ±250 Myr")

status = widgets.HTML()
warning = widgets.HTML(
    value=(
        "<b style='color:#ff9f43'>Future curve:</b> illustrative spectral "
        "continuation only — not an 80–90% accurate astronomical prediction."
    )
)
plot_output = widgets.Output()

def sync_position_limit():
    width = float(window_size.value)
    position.max = FULL_MAX - width
    if position.value > position.max:
        position.value = position.max

def past_stride(width_myr: float) -> int:
    if width_myr <= 2:
        return 1
    if width_myr <= 10:
        return 2
    if width_myr <= 25:
        return 5
    if width_myr <= 50:
        return 10
    if width_myr <= 100:
        return 20
    return 100

def render(*_):
    sync_position_limit()

    start = float(position.value)
    width = float(window_size.value)
    end = min(FULL_MAX, start + width)

    past_subset = past[
        (past["time_myr"] >= start)
        & (past["time_myr"] <= min(end, 0.0))
    ].copy()

    future_subset = future[
        (future["time_myr"] >= max(start, 0.0))
        & (future["time_myr"] <= end)
    ].copy()

    if not past_subset.empty:
        past_subset = past_subset.iloc[::past_stride(width)].copy()

    status.value = (
        f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000 "
        f"&nbsp; | &nbsp; <b>Negative = past; positive = modeled future.</b>"
    )

    fig = go.Figure()

    if not past_subset.empty:
        fig.add_trace(
            go.Scattergl(
                x=past_subset["time_myr"],
                y=past_subset["inclination_deg"],
                customdata=np.column_stack([
                    past_subset["time_years"],
                    past_subset["p"],
                    past_subset["q"],
                ]),
                mode="lines",
                name="La2010a past",
                line=dict(color="#52D6FF", width=1.6),
                hovertemplate=(
                    "<b>La2010a past</b>"
                    "<br>Time from J2000: %{x:.6f} Myr"
                    "<br>Years from J2000: %{customdata[0]:,.0f}"
                    "<br>Inclination: %{y:.6f}°"
                    "<br>p: %{customdata[1]:.9f}"
                    "<br>q: %{customdata[2]:.9f}"
                    "<extra></extra>"
                ),
            )
        )

    if not future_subset.empty:
        fig.add_trace(
            go.Scattergl(
                x=future_subset["time_myr"],
                y=future_subset["inclination_deg"],
                customdata=np.column_stack([
                    future_subset["time_years"],
                    future_subset["p"],
                    future_subset["q"],
                ]),
                mode="lines",
                name="Illustrative spectral future",
                line=dict(color="#FF6EC7", width=1.6, dash="dash"),
                hovertemplate=(
                    "<b>Illustrative future model</b>"
                    "<br>Time from J2000: +%{x:.6f} Myr"
                    "<br>Years after J2000: %{customdata[0]:,.0f}"
                    "<br>Modeled inclination: %{y:.6f}°"
                    "<br>modeled p: %{customdata[1]:.9f}"
                    "<br>modeled q: %{customdata[2]:.9f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=TODAY_I,
        line=dict(color="#FFD166", width=1.0, dash="dash"),
        annotation_text=f"J2000 = {TODAY_I:.6f}°",
        annotation_position="top left",
    )

    fig.add_vline(
        x=0.0,
        line=dict(color="#35E0A1", width=2),
        annotation_text="J2000",
        annotation_position="top",
    )

    all_y = []
    if not past_subset.empty:
        all_y.extend(past_subset["inclination_deg"].tolist())
    if not future_subset.empty:
        all_y.extend(future_subset["inclination_deg"].tolist())
    if not all_y:
        all_y = [TODAY_I]

    y_min = max(0.0, min(all_y) - 0.15)
    y_max = max(3.15, max(all_y) + 0.15)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(
            text="Earth Orbital Inclination: La2010a Past + Spectral Future",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Millions of years relative to J2000",
            range=[start, end],
            gridcolor="#28364A",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#1F7A4D",
            spikethickness=2,
        ),
        yaxis=dict(
            title="Inclination (degrees)",
            range=[y_min, y_max],
            gridcolor="#28364A",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#1F7A4D",
            spikethickness=2,
        ),
        hovermode="closest",
        hoverdistance=100,
        spikedistance=-1,
        height=700,
        margin=dict(l=70, r=25, t=75, b=70),
        legend=dict(
            bgcolor="rgba(0,0,0,0.75)",
            bordercolor="#28364A",
            borderwidth=1,
        ),
    )

    with plot_output:
        clear_output(wait=True)
        display(fig)

def go_past(_):
    position.value = FULL_MIN

def go_today(_):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, -width / 2.0))

def go_future(_):
    width = float(window_size.value)
    position.value = min(position.max, FULL_MAX - width)

def go_previous(_):
    position.value = max(
        FULL_MIN,
        float(position.value) - float(window_size.value),
    )

def go_next(_):
    position.value = min(
        float(position.max),
        float(position.value) + float(window_size.value),
    )

def go_full(_):
    window_size.value = 500.0
    position.value = FULL_MIN

window_size.observe(render, names="value")
position.observe(render, names="value")

past_button.on_click(go_past)
today_button.on_click(go_today)
future_button.on_click(go_future)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
full_button.on_click(go_full)

controls = widgets.VBox([
    status,
    warning,
    window_size,
    position,
    widgets.HBox([
        past_button,
        previous_button,
        today_button,
        next_button,
        future_button,
        full_button,
    ]),
])

display(widgets.VBox([controls, plot_output]))
render()
