# EMBOPSA_V0002G
# La2010a vs raw FFT vs phase-branch/variance-corrected FFT.
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
from scipy.stats import pearsonr
from IPython.display import display, HTML, clear_output

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

pio.renderers.default = "colab"

VERSION = "V0002G"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)
TRAIN_END_KYR = -50_000.0
N_HARMONICS = 120
MAX_LAG_KYR = 500.0
FULL_MIN = -250.0
FULL_MAX = 250.0

startup = widgets.HTML(
    "<b style='color:#35E0A1'>Loading La2010a and solving phase branch…</b>"
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
    raw["inclination_actual"] = np.degrees(
        2.0 * np.arcsin(np.clip(np.hypot(raw["p"], raw["q"]), 0.0, 1.0))
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
            out[start:stop] = mean + (2.0 / n_train) * np.real(
                phase @ coeff
            )
        return out

    def raw_model_inclination(t_kyr):
        p_m = evaluate(t_kyr, p_mean, p_freq, p_coeff)
        q_m = evaluate(t_kyr, q_mean, q_freq, q_coeff)
        i_m = np.degrees(
            2.0 * np.arcsin(np.clip(np.hypot(p_m, q_m), 0.0, 1.0))
        )
        return i_m

    # Validation interval and one-time extended model grid.
    t_valid = valid["t_kyr"].to_numpy(float)
    y_valid = valid["inclination_actual"].to_numpy(float)
    t_ext = np.arange(
        t_valid.min() - MAX_LAG_KYR,
        t_valid.max() + MAX_LAG_KYR + 1.0,
        1.0,
    )
    x_ext = raw_model_inclination(t_ext)
    x_valid_raw = np.interp(t_valid, t_ext, x_ext)

    # Search all phase branches. For each lag, match mean and standard
    # deviation rather than ordinary regression slope; this avoids the
    # artificial amplitude compression caused by multiplying by r.
    def calibrate_for_lag(lag_kyr):
        shifted = np.interp(t_valid + lag_kyr, t_ext, x_ext)
        sx = float(np.std(shifted))
        sy = float(np.std(y_valid))
        amp = sy / sx if sx > 0 else 1.0
        offset = float(np.mean(y_valid) - amp * np.mean(shifted))
        corrected = offset + amp * shifted
        r = float(pearsonr(y_valid, corrected).statistic)
        rmse = float(np.sqrt(np.mean((corrected - y_valid) ** 2)))
        return rmse, r, offset, amp, corrected

    coarse_lags = np.arange(-MAX_LAG_KYR, MAX_LAG_KYR + 5.0, 5.0)
    coarse = []
    for lag in coarse_lags:
        rmse, r, offset, amp, _ = calibrate_for_lag(float(lag))
        if r > 0:
            coarse.append((rmse, -r, float(lag), offset, amp))
    coarse.sort()
    coarse_best = coarse[0][2]

    fine_lags = np.arange(coarse_best - 10.0, coarse_best + 10.0 + 1.0, 1.0)
    fine = []
    for lag in fine_lags:
        rmse, r, offset, amp, corrected = calibrate_for_lag(float(lag))
        if r > 0:
            fine.append((rmse, -r, float(lag), offset, amp, corrected))
    fine.sort()
    best_rmse, neg_r, best_shift_kyr, beta0, beta1, corrected_valid = fine[0]
    best_r = -neg_r

    raw_r = float(pearsonr(y_valid, x_valid_raw).statistic)
    raw_rmse = float(np.sqrt(np.mean((x_valid_raw - y_valid) ** 2)))
    corrected_mae = float(np.mean(np.abs(corrected_valid - y_valid)))

    startup.value = "<b style='color:#35E0A1'>Phase branch corrected.</b>"

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
    phase_trim = widgets.FloatSlider(
        value=0.0,
        min=-250.0,
        max=250.0,
        step=1.0,
        description="Phase trim kyr:",
        continuous_update=False,
        layout=widgets.Layout(width="800px"),
    )
    amplitude_trim = widgets.FloatSlider(
        value=1.0,
        min=0.50,
        max=1.50,
        step=0.01,
        description="Amplitude trim:",
        continuous_update=False,
        readout_format=".2f",
        layout=widgets.Layout(width="800px"),
    )

    show_actual = widgets.Checkbox(True, description="Actual La2010a", indent=False)
    show_raw = widgets.Checkbox(True, description="Raw FFT", indent=False)
    show_corrected = widgets.Checkbox(True, description="Corrected FFT", indent=False)
    show_error = widgets.Checkbox(False, description="Corrected error", indent=False)

    training_button = widgets.Button(description="Training")
    validation_button = widgets.Button(description="Validation")
    today_button = widgets.Button(description="Today")
    future_button = widgets.Button(description="Future")
    previous_button = widgets.Button(description="◀ Previous")
    next_button = widgets.Button(description="Next ▶")
    reset_button = widgets.Button(description="Reset correction")
    full_button = widgets.Button(description="Full ±250 Myr")

    metrics = widgets.HTML()
    status = widgets.HTML()
    plot_output = widgets.Output()

    def update_metrics():
        total_shift = best_shift_kyr + float(phase_trim.value)
        total_amp = beta1 * float(amplitude_trim.value)
        metrics.value = (
            "<b>Validation calibration: −50 to 0 Myr</b><br>"
            f"Automatic phase shift = {best_shift_kyr:+.0f} kyr; "
            f"manual trim = {phase_trim.value:+.0f} kyr; "
            f"total = {total_shift:+.0f} kyr<br>"
            f"Variance-preserving correction: i<sub>corr</sub> = "
            f"{beta0:+.9f}° + ({total_amp:.9f}) "
            f"[i<sub>FFT shifted</sub> − mean] with mean restored<br>"
            f"Raw: r = {raw_r:.6f}, RMSE = {raw_rmse:.6f}°<br>"
            f"Automatic corrected: r = {best_r:.6f}, "
            f"R² = {best_r**2:.6f}, RMSE = {best_rmse:.6f}°, "
            f"MAE = {corrected_mae:.6f}°"
        )

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

    def corrected_curve(t_plot):
        total_shift = best_shift_kyr + float(phase_trim.value)
        shifted = raw_model_inclination(t_plot + total_shift)
        centered = shifted - np.mean(shifted)
        target_mean = np.mean(y_valid)
        amp = beta1 * float(amplitude_trim.value)
        return target_mean + amp * centered

    def render(*_):
        try:
            sync_limit()
            update_metrics()
            start = float(position.value)
            end = min(FULL_MAX, start + float(window_size.value))
            status.value = (
                f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000"
            )

            fig = go.Figure()
            observed = raw[
                (raw["time_myr"] >= start)
                & (raw["time_myr"] <= min(end, 0.0))
            ].copy()
            if len(observed) > 6000:
                observed = observed.iloc[::max(1, len(observed)//6000)].copy()

            if show_actual.value and not observed.empty:
                fig.add_trace(go.Scattergl(
                    x=observed["time_myr"],
                    y=observed["inclination_actual"],
                    mode="lines",
                    name="Actual La2010a",
                    line=dict(color="#52D6FF", width=2.0),
                    hovertemplate=(
                        "<b>Actual La2010a</b><br>Time: %{x:.6f} Myr"
                        "<br>Inclination: %{y:.6f}°<extra></extra>"
                    ),
                ))

            t_plot = visible_times(start, end)
            x_plot = t_plot / 1000.0
            raw_i = raw_model_inclination(t_plot)
            corr_i = corrected_curve(t_plot)

            if show_raw.value:
                fig.add_trace(go.Scattergl(
                    x=x_plot,
                    y=raw_i,
                    mode="lines",
                    name="Raw FFT",
                    line=dict(color="#FF6EC7", width=1.4, dash="dot"),
                ))

            if show_corrected.value:
                fig.add_trace(go.Scattergl(
                    x=x_plot,
                    y=corr_i,
                    mode="lines",
                    name="Corrected FFT",
                    line=dict(color="#35E0A1", width=1.9, dash="dash"),
                ))

            if show_error.value and end <= 0.0 and not observed.empty:
                interp_corr = np.interp(
                    observed["t_kyr"].to_numpy(float), t_plot, corr_i
                )
                err = interp_corr - observed["inclination_actual"].to_numpy(float)
                fig.add_trace(go.Scattergl(
                    x=observed["time_myr"],
                    y=err,
                    mode="lines",
                    name="Corrected − actual",
                    yaxis="y2",
                    line=dict(color="#FF9F43", width=1.4),
                ))

            fig.add_vline(
                x=-50.0,
                line=dict(color="#FF9F43", width=1.0, dash="dot"),
                annotation_text="Training ends",
            )
            fig.add_vline(
                x=0.0,
                line=dict(color="#35E0A1", width=1.5),
                annotation_text="J2000",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                title=dict(
                    text="La2010a vs FFT — Phase-Branch and Variance Correction",
                    x=0.5,
                ),
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
                legend=dict(
                    bgcolor="rgba(0,0,0,0.78)",
                    bordercolor="#28364A",
                    borderwidth=1,
                ),
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
        position.value = max(FULL_MIN, min(position.max, center - width/2.0))

    training_button.on_click(lambda _: set_center(-150.0))
    validation_button.on_click(lambda _: set_center(-25.0))
    today_button.on_click(lambda _: set_center(0.0))
    future_button.on_click(lambda _: set_center(25.0))
    previous_button.on_click(lambda _: setattr(
        position, "value", max(FULL_MIN, position.value-window_size.value)
    ))
    next_button.on_click(lambda _: setattr(
        position, "value", min(position.max, position.value+window_size.value)
    ))
    reset_button.on_click(lambda _: (
        setattr(phase_trim, "value", 0.0),
        setattr(amplitude_trim, "value", 1.0),
    ))
    full_button.on_click(lambda _: (
        setattr(window_size, "value", 500.0),
        setattr(position, "value", FULL_MIN),
    ))

    for w in (
        window_size,
        position,
        phase_trim,
        amplitude_trim,
        show_actual,
        show_raw,
        show_corrected,
        show_error,
    ):
        w.observe(render, names="value")

    controls = widgets.VBox([
        metrics,
        status,
        widgets.HTML(
            "<b style='color:#ff9f43'>The future curve remains an illustrative spectral continuation.</b>"
        ),
        window_size,
        position,
        phase_trim,
        amplitude_trim,
        widgets.HBox([show_actual, show_raw, show_corrected, show_error]),
        widgets.HBox([
            training_button,
            validation_button,
            previous_button,
            today_button,
            next_button,
            future_button,
            reset_button,
            full_button,
        ]),
    ])

    display(widgets.VBox([controls, plot_output]))
    render()

except Exception:
    display(HTML("<pre style='color:#ff6b6b'>" + traceback.format_exc() + "</pre>"))
