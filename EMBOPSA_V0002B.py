# EMBOPSA_V0002B
# La2010a spectral-model back-test and ±250 Myr explorer.
# NO AI-GENERATED IMAGES.
#
# Model:
#   p(t) = p_bar + 2/N * Re[sum C_p,k exp(i 2π f_k (t-t0))]
#   q(t) = q_bar + 2/N * Re[sum C_q,k exp(i 2π f_k (t-t0))]
#   i(t) = 2 asin(sqrt(p(t)^2 + q(t)^2))
#
# Training interval : -250 to -50 Myr
# Validation interval:  -50 to   0 Myr (not used during fitting)
# Forecast interval  :    0 to +250 Myr
#
# The far-future curve is illustrative only. Chaotic phase divergence makes
# a unique 250-Myr astronomical prediction impossible from this model.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0002B"

DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

TRAIN_END_KYR = -50_000.0
N_HARMONICS = 180
FUTURE_STEP_KYR = 1.0

response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()

raw = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)

for column in raw.columns:
    raw[column] = pd.to_numeric(raw[column], errors="coerce")

raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)

raw["time_myr"] = raw["t_kyr"] / 1000.0
raw["time_years"] = raw["t_kyr"] * 1000.0
raw["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(
        np.clip(np.hypot(raw["p"], raw["q"]), 0.0, 1.0)
    )
)

train = raw[raw["t_kyr"] <= TRAIN_END_KYR].copy()
validation = raw[raw["t_kyr"] > TRAIN_END_KYR].copy()

t_train = train["t_kyr"].to_numpy(dtype=float)
dt_kyr = float(np.median(np.diff(t_train)))
t_origin = float(t_train[0])
n_train = len(train)

def fit_spectral(y: np.ndarray, n_harmonics: int):
    mean = float(np.mean(y))
    centered = y - mean
    coeff = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(centered), d=dt_kyr)

    candidates = np.arange(1, len(coeff))
    strongest = candidates[
        np.argsort(np.abs(coeff[candidates]))[-n_harmonics:]
    ]
    strongest = strongest[np.argsort(freq[strongest])]

    return mean, freq[strongest], coeff[strongest]

p_mean, p_freq, p_coeff = fit_spectral(
    train["p"].to_numpy(dtype=float),
    N_HARMONICS,
)
q_mean, q_freq, q_coeff = fit_spectral(
    train["q"].to_numpy(dtype=float),
    N_HARMONICS,
)

def evaluate_spectral(
    t_kyr: np.ndarray,
    mean: float,
    freq: np.ndarray,
    coeff: np.ndarray,
    chunk_size: int = 6000,
) -> np.ndarray:
    result = np.empty(len(t_kyr), dtype=float)
    tau = np.asarray(t_kyr, dtype=float) - t_origin

    for start in range(0, len(tau), chunk_size):
        stop = min(len(tau), start + chunk_size)
        exponential = np.exp(
            2j * np.pi * np.outer(tau[start:stop], freq)
        )
        result[start:stop] = (
            mean
            + (2.0 / n_train) * np.real(exponential @ coeff)
        )

    return result

# Evaluate model over all observed times for training and validation plots.
t_actual = raw["t_kyr"].to_numpy(dtype=float)
p_model_actual = evaluate_spectral(t_actual, p_mean, p_freq, p_coeff)
q_model_actual = evaluate_spectral(t_actual, q_mean, q_freq, q_coeff)
i_model_actual = np.degrees(
    2.0 * np.arcsin(
        np.clip(np.hypot(p_model_actual, q_model_actual), 0.0, 1.0)
    )
)

model_actual = pd.DataFrame({
    "t_kyr": t_actual,
    "time_myr": t_actual / 1000.0,
    "time_years": t_actual * 1000.0,
    "p_model": p_model_actual,
    "q_model": q_model_actual,
    "inclination_model_deg": i_model_actual,
})

comparison = raw.merge(model_actual, on=["t_kyr", "time_myr", "time_years"])
comparison["error_deg"] = (
    comparison["inclination_model_deg"]
    - comparison["inclination_deg"]
)

validation_cmp = comparison[
    comparison["t_kyr"] > TRAIN_END_KYR
].copy()

error = validation_cmp["error_deg"].to_numpy(dtype=float)
actual_v = validation_cmp["inclination_deg"].to_numpy(dtype=float)
model_v = validation_cmp["inclination_model_deg"].to_numpy(dtype=float)

RMS_DEG = float(np.sqrt(np.mean(error**2)))
MAE_DEG = float(np.mean(np.abs(error)))
MAX_ABS_DEG = float(np.max(np.abs(error)))
BIAS_DEG = float(np.mean(error))
CORRELATION = float(np.corrcoef(actual_v, model_v)[0, 1])
ss_res = float(np.sum((actual_v - model_v) ** 2))
ss_tot = float(np.sum((actual_v - np.mean(actual_v)) ** 2))
R2 = float(1.0 - ss_res / ss_tot)

# Future model, 0 to +250 Myr.
t_future = np.arange(
    0.0,
    250_000.0 + FUTURE_STEP_KYR,
    FUTURE_STEP_KYR,
)
p_future = evaluate_spectral(t_future, p_mean, p_freq, p_coeff)
q_future = evaluate_spectral(t_future, q_mean, q_freq, q_coeff)
i_future = np.degrees(
    2.0 * np.arcsin(
        np.clip(np.hypot(p_future, q_future), 0.0, 1.0)
    )
)

future = pd.DataFrame({
    "t_kyr": t_future,
    "time_myr": t_future / 1000.0,
    "time_years": t_future * 1000.0,
    "p_model": p_future,
    "q_model": q_future,
    "inclination_model_deg": i_future,
})

FULL_MIN = -250.0
FULL_MAX = 250.0
TODAY_I = float(raw.loc[raw["t_kyr"].abs().idxmin(), "inclination_deg"])

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

show_actual = widgets.Checkbox(
    value=True,
    description="Actual La2010a",
    indent=False,
)
show_model = widgets.Checkbox(
    value=True,
    description="Spectral model",
    indent=False,
)
show_error = widgets.Checkbox(
    value=False,
    description="Model − actual error",
    indent=False,
)

past_button = widgets.Button(description="Training")
validation_button = widgets.Button(description="Validation")
today_button = widgets.Button(description="Today")
future_button = widgets.Button(description="Future")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
full_button = widgets.Button(description="Full ±250 Myr")

status = widgets.HTML()
metrics = widgets.HTML(
    value=(
        "<b>Withheld validation: −50 to 0 Myr</b><br>"
        f"RMS error: {RMS_DEG:.6f}° &nbsp; | &nbsp; "
        f"MAE: {MAE_DEG:.6f}° &nbsp; | &nbsp; "
        f"Maximum |error|: {MAX_ABS_DEG:.6f}°<br>"
        f"Bias: {BIAS_DEG:+.6f}° &nbsp; | &nbsp; "
        f"Correlation: {CORRELATION:.6f} &nbsp; | &nbsp; "
        f"R²: {R2:.6f}"
    )
)
warning = widgets.HTML(
    value=(
        "<b style='color:#ff9f43'>Interpretation:</b> "
        "The model is trained only on −250 to −50 Myr. "
        "The cyan curve from −50 to 0 Myr is real withheld data; "
        "the magenta curve there is a genuine out-of-sample prediction. "
        "The dashed future beyond J2000 is illustrative."
    )
)
plot_output = widgets.Output()

def sync_position_limit():
    width = float(window_size.value)
    position.max = FULL_MAX - width
    if position.value > position.max:
        position.value = position.max

def data_stride(width_myr: float) -> int:
    if width_myr <= 1:
        return 1
    if width_myr <= 5:
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

    observed = comparison[
        (comparison["time_myr"] >= start)
        & (comparison["time_myr"] <= min(end, 0.0))
    ].copy()

    forecast = future[
        (future["time_myr"] >= max(start, 0.0))
        & (future["time_myr"] <= end)
    ].copy()

    stride = data_stride(width)
    observed = observed.iloc[::stride].copy()
    forecast = forecast.iloc[::stride].copy()

    zone = (
        "training"
        if end <= -50.0
        else "validation"
        if start >= -50.0 and end <= 0.0
        else "future"
        if start >= 0.0
        else "mixed"
    )

    status.value = (
        f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000 "
        f"&nbsp; | &nbsp; <b>Region:</b> {zone}"
    )

    fig = go.Figure()

    if show_actual.value and not observed.empty:
        fig.add_trace(
            go.Scattergl(
                x=observed["time_myr"],
                y=observed["inclination_deg"],
                customdata=np.column_stack([
                    observed["time_years"],
                    observed["p"],
                    observed["q"],
                ]),
                mode="lines",
                name="Actual La2010a",
                line=dict(color="#52D6FF", width=2.0),
                hovertemplate=(
                    "<b>Actual La2010a</b>"
                    "<br>Time from J2000: %{x:.6f} Myr"
                    "<br>Years from J2000: %{customdata[0]:,.0f}"
                    "<br>Inclination: %{y:.6f}°"
                    "<br>p: %{customdata[1]:.9f}"
                    "<br>q: %{customdata[2]:.9f}"
                    "<extra></extra>"
                ),
            )
        )

    if show_model.value and not observed.empty:
        is_validation = observed["time_myr"].max() > -50.0
        fig.add_trace(
            go.Scattergl(
                x=observed["time_myr"],
                y=observed["inclination_model_deg"],
                customdata=np.column_stack([
                    observed["time_years"],
                    observed["p_model"],
                    observed["q_model"],
                ]),
                mode="lines",
                name=(
                    "Model prediction"
                    if is_validation
                    else "Model fit"
                ),
                line=dict(
                    color="#FF6EC7",
                    width=1.8,
                    dash="dash",
                ),
                hovertemplate=(
                    "<b>Spectral model</b>"
                    "<br>Time from J2000: %{x:.6f} Myr"
                    "<br>Years from J2000: %{customdata[0]:,.0f}"
                    "<br>Modeled inclination: %{y:.6f}°"
                    "<br>modeled p: %{customdata[1]:.9f}"
                    "<br>modeled q: %{customdata[2]:.9f}"
                    "<extra></extra>"
                ),
            )
        )

    if show_model.value and not forecast.empty:
        fig.add_trace(
            go.Scattergl(
                x=forecast["time_myr"],
                y=forecast["inclination_model_deg"],
                customdata=np.column_stack([
                    forecast["time_years"],
                    forecast["p_model"],
                    forecast["q_model"],
                ]),
                mode="lines",
                name="Illustrative future model",
                line=dict(
                    color="#FF6EC7",
                    width=1.8,
                    dash="dash",
                ),
                hovertemplate=(
                    "<b>Illustrative future model</b>"
                    "<br>Time from J2000: +%{x:.6f} Myr"
                    "<br>Years after J2000: %{customdata[0]:,.0f}"
                    "<br>Modeled inclination: %{y:.6f}°"
                    "<extra></extra>"
                ),
            )
        )

    if show_error.value and not observed.empty:
        fig.add_trace(
            go.Scattergl(
                x=observed["time_myr"],
                y=observed["error_deg"],
                mode="lines",
                name="Model − actual error",
                line=dict(color="#FF9F43", width=1.6),
                hovertemplate=(
                    "<b>Prediction error</b>"
                    "<br>Time from J2000: %{x:.6f} Myr"
                    "<br>Error: %{y:+.6f}°"
                    "<extra></extra>"
                ),
                yaxis="y2",
            )
        )

    fig.add_vrect(
        x0=-250,
        x1=-50,
        fillcolor="rgba(53,224,161,0.045)",
        line_width=0,
        layer="below",
    )
    fig.add_vrect(
        x0=-50,
        x1=0,
        fillcolor="rgba(255,159,67,0.055)",
        line_width=0,
        layer="below",
    )
    fig.add_vrect(
        x0=0,
        x1=250,
        fillcolor="rgba(255,110,199,0.04)",
        line_width=0,
        layer="below",
    )

    fig.add_vline(
        x=-50.0,
        line=dict(color="#FF9F43", width=1.2, dash="dot"),
        annotation_text="Training ends",
        annotation_position="top left",
    )
    fig.add_vline(
        x=0.0,
        line=dict(color="#35E0A1", width=1.8),
        annotation_text="J2000",
        annotation_position="top",
    )

    y_candidates = [TODAY_I]
    if not observed.empty:
        if show_actual.value:
            y_candidates.extend(observed["inclination_deg"].tolist())
        if show_model.value:
            y_candidates.extend(observed["inclination_model_deg"].tolist())
    if show_model.value and not forecast.empty:
        y_candidates.extend(forecast["inclination_model_deg"].tolist())

    y_min = max(0.0, min(y_candidates) - 0.15)
    y_max = max(3.15, max(y_candidates) + 0.15)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(
            text="La2010a vs Spectral Prediction — Back-Test and Future",
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
        yaxis2=dict(
            title="Error (degrees)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinecolor="#FF9F43",
            visible=bool(show_error.value),
        ),
        hovermode="closest",
        hoverdistance=100,
        spikedistance=-1,
        height=720,
        margin=dict(l=75, r=75, t=80, b=70),
        legend=dict(
            bgcolor="rgba(0,0,0,0.78)",
            bordercolor="#28364A",
            borderwidth=1,
        ),
    )

    with plot_output:
        clear_output(wait=True)
        display(fig)

def go_training(_):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, -150.0 - width / 2.0))

def go_validation(_):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, -25.0 - width / 2.0))

def go_today(_):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, -width / 2.0))

def go_future(_):
    width = float(window_size.value)
    position.value = max(FULL_MIN, min(position.max, 25.0 - width / 2.0))

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
show_actual.observe(render, names="value")
show_model.observe(render, names="value")
show_error.observe(render, names="value")

past_button.on_click(go_training)
validation_button.on_click(go_validation)
today_button.on_click(go_today)
future_button.on_click(go_future)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
full_button.on_click(go_full)

controls = widgets.VBox([
    status,
    metrics,
    warning,
    window_size,
    position,
    widgets.HBox([show_actual, show_model, show_error]),
    widgets.HBox([
        past_button,
        validation_button,
        previous_button,
        today_button,
        next_button,
        future_button,
        full_button,
    ]),
])

display(widgets.VBox([controls, plot_output]))
render()
