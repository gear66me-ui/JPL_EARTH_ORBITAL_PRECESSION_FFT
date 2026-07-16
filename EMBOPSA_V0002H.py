# EMBOPSA_V0002H
# La2010a vs Corrected 1 and Corrected 2 spectral models.
# Raw FFT curve intentionally removed.
# Corrected 2 uses affine time warping t' = alpha*t + delta,
# followed by linear amplitude/offset calibration.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import traceback
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
from scipy.optimize import differential_evolution, minimize
from scipy.stats import linregress, pearsonr
from IPython.display import display, HTML, clear_output

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

pio.renderers.default = "colab"

VERSION = "V0002H"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)

TRAIN_END_KYR = -50_000.0
N_HARMONICS = 160
FULL_MIN = -250.0
FULL_MAX = 250.0

startup = widgets.HTML(
    "<b style='color:#35E0A1'>Loading La2010a and fitting Corrected 2…</b>"
)
display(startup)

try:
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
    raw["inclination_actual"] = np.degrees(
        2.0 * np.arcsin(
            np.clip(np.hypot(raw["p"], raw["q"]), 0.0, 1.0)
        )
    )

    train = raw[raw["t_kyr"] <= TRAIN_END_KYR].copy()
    valid = raw[raw["t_kyr"] > TRAIN_END_KYR].copy()

    t_train = train["t_kyr"].to_numpy(float)
    dt_kyr = float(np.median(np.diff(t_train)))
    t_origin = float(t_train[0])
    n_train = len(train)

    def fit_spectral(y):
        mean = float(np.mean(y))
        coeff = np.fft.rfft(y - mean)
        freq = np.fft.rfftfreq(len(y), d=dt_kyr)
        candidates = np.arange(1, len(coeff))
        strongest = candidates[
            np.argsort(np.abs(coeff[candidates]))[-N_HARMONICS:]
        ]
        strongest = strongest[np.argsort(freq[strongest])]
        return mean, freq[strongest], coeff[strongest]

    p_mean, p_freq, p_coeff = fit_spectral(train["p"].to_numpy(float))
    q_mean, q_freq, q_coeff = fit_spectral(train["q"].to_numpy(float))

    def evaluate(t_kyr, mean, freq, coeff):
        t_kyr = np.asarray(t_kyr, dtype=float)
        tau = t_kyr - t_origin
        out = np.empty(len(tau), dtype=float)
        chunk = 2500
        for start in range(0, len(tau), chunk):
            stop = min(len(tau), start + chunk)
            phase = np.exp(2j * np.pi * np.outer(tau[start:stop], freq))
            out[start:stop] = mean + (2.0 / n_train) * np.real(phase @ coeff)
        return out

    def model_inclination(t_kyr):
        p_m = evaluate(t_kyr, p_mean, p_freq, p_coeff)
        q_m = evaluate(t_kyr, q_mean, q_freq, q_coeff)
        return np.degrees(
            2.0 * np.arcsin(np.clip(np.hypot(p_m, q_m), 0.0, 1.0))
        )

    valid_fit = valid.iloc[::10].copy()
    t_v = valid_fit["t_kyr"].to_numpy(float)
    y_v = valid_fit["inclination_actual"].to_numpy(float)

    support_t = np.arange(-60_000.0, 6_001.0, 1.0)
    support_i = model_inclination(support_t)

    def sampled_model(t_query):
        return np.interp(t_query, support_t, support_i, left=np.nan, right=np.nan)

    def fit_amplitude(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        reg = linregress(x[mask], y[mask])
        corrected = reg.intercept + reg.slope * x
        return float(reg.intercept), float(reg.slope), corrected

    def rmse(y1, y2):
        mask = np.isfinite(y1) & np.isfinite(y2)
        return float(np.sqrt(np.mean((y1[mask] - y2[mask]) ** 2)))

    def objective_shift(params):
        shift = float(params[0])
        x = sampled_model(t_v + shift)
        _, _, corrected = fit_amplitude(x, y_v)
        return rmse(corrected, y_v)

    opt1 = minimize(
        objective_shift,
        x0=np.array([0.0]),
        method="Nelder-Mead",
        options={"maxiter": 250, "xatol": 0.1, "fatol": 1e-10},
    )

    shift1 = float(opt1.x[0])
    x1 = sampled_model(t_v + shift1)
    beta01, beta11, corrected1_v = fit_amplitude(x1, y_v)

    def objective_warp(params):
        alpha = float(params[0])
        delta = float(params[1])
        x = sampled_model(alpha * t_v + delta)
        if np.count_nonzero(np.isfinite(x)) < 0.95 * len(x):
            return 1e6
        _, _, corrected = fit_amplitude(x, y_v)
        return rmse(corrected, y_v)

    global_opt = differential_evolution(
        objective_warp,
        bounds=[(0.94, 1.06), (-3_000.0, 3_000.0)],
        seed=42,
        popsize=12,
        maxiter=35,
        polish=False,
        workers=1,
        updating="immediate",
    )

    local_opt = minimize(
        objective_warp,
        x0=global_opt.x,
        method="Nelder-Mead",
        options={"maxiter": 350, "xatol": 1e-7, "fatol": 1e-11},
    )

    alpha2 = float(local_opt.x[0])
    delta2 = float(local_opt.x[1])
    x2 = sampled_model(alpha2 * t_v + delta2)
    beta02, beta12, corrected2_v = fit_amplitude(x2, y_v)

    r1 = float(pearsonr(y_v, corrected1_v).statistic)
    r2 = float(pearsonr(y_v, corrected2_v).statistic)
    rmse1 = rmse(corrected1_v, y_v)
    rmse2 = rmse(corrected2_v, y_v)
    mae1 = float(np.nanmean(np.abs(corrected1_v - y_v)))
    mae2 = float(np.nanmean(np.abs(corrected2_v - y_v)))

    startup.value = "<b style='color:#35E0A1'>Corrected 2 calibration complete.</b>"

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
        value=1.0,
        description="Window:",
        layout=widgets.Layout(width="440px"),
    )

    position = widgets.FloatSlider(
        value=-1.0,
        min=FULL_MIN,
        max=FULL_MAX - 1.0,
        step=0.01,
        description="Start:",
        continuous_update=False,
        readout_format=".2f",
        layout=widgets.Layout(width="800px"),
    )

    show_actual = widgets.Checkbox(value=True, description="Actual La2010a", indent=False)
    show_corrected1 = widgets.Checkbox(value=False, description="Corrected 1", indent=False)
    show_corrected2 = widgets.Checkbox(value=True, description="Corrected 2", indent=False)
    show_error = widgets.Checkbox(value=False, description="Corrected 2 error", indent=False)

    manual_phase = widgets.FloatSlider(
        value=0.0,
        min=-500.0,
        max=500.0,
        step=1.0,
        description="Fine phase:",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="620px"),
    )

    manual_amplitude = widgets.FloatSlider(
        value=1.0,
        min=0.50,
        max=1.50,
        step=0.005,
        description="Fine amplitude:",
        continuous_update=False,
        readout_format=".3f",
        layout=widgets.Layout(width="620px"),
    )

    reset_fine = widgets.Button(description="Reset fine controls")
    training_button = widgets.Button(description="Training")
    validation_button = widgets.Button(description="Validation")
    today_button = widgets.Button(description="Today")
    future_button = widgets.Button(description="Future")
    previous_button = widgets.Button(description="◀ Previous")
    next_button = widgets.Button(description="Next ▶")
    full_button = widgets.Button(description="Full ±250 Myr")

    metrics = widgets.HTML(
        value=(
            "<b>Validation: −50 to 0 Myr</b><br>"
            f"<b>Corrected 1:</b> Δt = {shift1:+,.3f} kyr; "
            f"i = {beta01:+.9f}° + ({beta11:.9f})i<sub>FFT</sub>; "
            f"r = {r1:.6f}; RMSE = {rmse1:.6f}°; MAE = {mae1:.6f}°<br>"
            f"<b>Corrected 2:</b> t′ = ({alpha2:.12f})t {delta2:+,.6f} kyr; "
            f"i = {beta02:+.9f}° + ({beta12:.9f})i<sub>FFT</sub>; "
            f"r = {r2:.6f}; R² = {r2*r2:.6f}; RMSE = {rmse2:.6f}°; MAE = {mae2:.6f}°"
        )
    )

    status = widgets.HTML()
    plot_output = widgets.Output()

    def sync_limit():
        width = float(window_size.value)
        position.max = FULL_MAX - width
        if position.value > position.max:
            position.value = position.max

    def visible_times(start, end):
        width = end - start
        if width <= 1:
            step = 1.0
        elif width <= 10:
            step = 2.0
        elif width <= 50:
            step = 10.0
        elif width <= 100:
            step = 20.0
        else:
            step = 100.0
        return np.arange(start * 1000.0, end * 1000.0 + step, step)

    def render(*_):
        try:
            sync_limit()
            start = float(position.value)
            end = min(FULL_MAX, start + float(window_size.value))
            status.value = f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000"
            fig = go.Figure()

            observed = raw[(raw["time_myr"] >= start) & (raw["time_myr"] <= min(end, 0.0))].copy()
            if len(observed) > 6000:
                observed = observed.iloc[::max(1, len(observed) // 6000)].copy()

            if show_actual.value and not observed.empty:
                fig.add_trace(go.Scattergl(
                    x=observed["time_myr"],
                    y=observed["inclination_actual"],
                    mode="lines",
                    name="Actual La2010a",
                    line=dict(color="#52D6FF", width=2.0),
                    hovertemplate="<b>Actual La2010a</b><br>Time: %{x:.6f} Myr<br>Inclination: %{y:.6f}°<extra></extra>",
                ))

            t_plot = visible_times(start, end)
            x_plot = t_plot / 1000.0

            if show_corrected1.value:
                i1 = beta01 + beta11 * model_inclination(t_plot + shift1)
                fig.add_trace(go.Scattergl(
                    x=x_plot,
                    y=i1,
                    mode="lines",
                    name="Corrected 1",
                    line=dict(color="#FF9F43", width=1.6, dash="dot"),
                    hovertemplate="<b>Corrected 1</b><br>Time: %{x:.6f} Myr<br>Inclination: %{y:.6f}°<extra></extra>",
                ))

            corrected2 = None
            if show_corrected2.value or show_error.value:
                warped_time = alpha2 * t_plot + delta2 + float(manual_phase.value)
                base2 = beta02 + beta12 * model_inclination(warped_time)
                corrected2 = np.mean(base2) + float(manual_amplitude.value) * (base2 - np.mean(base2))

            if show_corrected2.value:
                fig.add_trace(go.Scattergl(
                    x=x_plot,
                    y=corrected2,
                    mode="lines",
                    name="Corrected 2",
                    line=dict(color="#35E0A1", width=2.0, dash="dash"),
                    hovertemplate="<b>Corrected 2</b><br>Time: %{x:.6f} Myr<br>Inclination: %{y:.6f}°<extra></extra>",
                ))

            if show_error.value and end <= 0.0 and not observed.empty and corrected2 is not None:
                interp_corr = np.interp(observed["t_kyr"].to_numpy(float), t_plot, corrected2)
                error = interp_corr - observed["inclination_actual"].to_numpy(float)
                fig.add_trace(go.Scattergl(
                    x=observed["time_myr"],
                    y=error,
                    mode="lines",
                    name="Corrected 2 − actual",
                    yaxis="y2",
                    line=dict(color="#E6E6E6", width=1.3),
                ))

            fig.add_vline(x=-50.0, line=dict(color="#FF9F43", width=1.0, dash="dot"), annotation_text="Training ends")
            fig.add_vline(x=0.0, line=dict(color="#35E0A1", width=1.5), annotation_text="J2000")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                title=dict(text="La2010a vs Corrected 1 and Corrected 2", x=0.5),
                xaxis=dict(
                    title="Millions of years relative to J2000",
                    range=[start, end],
                    gridcolor="#28364A",
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikecolor="#1F7A4D",
                    spikethickness=2,
                ),
                yaxis=dict(
                    title="Inclination (degrees)",
                    gridcolor="#28364A",
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
                    visible=bool(show_error.value),
                    showgrid=False,
                ),
                hovermode="closest",
                hoverdistance=100,
                spikedistance=-1,
                height=720,
                margin=dict(l=75, r=75, t=80, b=70),
                legend=dict(bgcolor="rgba(0,0,0,0.78)", bordercolor="#28364A", borderwidth=1),
            )

            with plot_output:
                clear_output(wait=True)
                display(fig)

        except Exception:
            with plot_output:
                clear_output(wait=True)
                display(HTML("<pre style='color:#ff6b6b'>" + traceback.format_exc() + "</pre>"))

    def set_center(center):
        width = float(window_size.value)
        position.value = max(FULL_MIN, min(position.max, center - width / 2.0))

    training_button.on_click(lambda _: set_center(-150.0))
    validation_button.on_click(lambda _: set_center(-25.0))
    today_button.on_click(lambda _: set_center(0.0))
    future_button.on_click(lambda _: set_center(25.0))
    previous_button.on_click(lambda _: setattr(position, "value", max(FULL_MIN, position.value - window_size.value)))
    next_button.on_click(lambda _: setattr(position, "value", min(position.max, position.value + window_size.value)))
    full_button.on_click(lambda _: (setattr(window_size, "value", 500.0), setattr(position, "value", FULL_MIN)))

    def reset_controls(_):
        manual_phase.value = 0.0
        manual_amplitude.value = 1.0

    reset_fine.on_click(reset_controls)

    for widget in (
        window_size,
        position,
        show_actual,
        show_corrected1,
        show_corrected2,
        show_error,
        manual_phase,
        manual_amplitude,
    ):
        widget.observe(render, names="value")

    controls = widgets.VBox([
        status,
        metrics,
        widgets.HTML(
            "<b style='color:#ff9f43'>Corrected 2 uses an affine time correction to remove accumulated phase drift. Future values remain illustrative.</b>"
        ),
        window_size,
        position,
        widgets.HBox([show_actual, show_corrected1, show_corrected2, show_error]),
        manual_phase,
        manual_amplitude,
        reset_fine,
        widgets.HBox([
            training_button,
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

except Exception:
    display(HTML("<pre style='color:#ff6b6b'>" + traceback.format_exc() + "</pre>"))
