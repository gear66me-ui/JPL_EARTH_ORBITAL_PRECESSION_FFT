# EMBOPSA_V0002N
# La2010a 253-kyr spectral-cycle explorer.
# Deterministic scientific plots only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import traceback
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

pio.renderers.default = "colab"

VERSION = "V0002N"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)

status = widgets.HTML(
    "<b style='color:#35E0A1'>Loading La2010a and locating the 253-kyr FFT cycle…</b>"
)
display(status)

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

    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)

    t_kyr = raw["t_kyr"].to_numpy(np.float64)
    time_myr = t_kyr / 1000.0
    p = raw["p"].to_numpy(np.float64)
    q = raw["q"].to_numpy(np.float64)

    n = len(raw)
    dt_kyr = float(np.median(np.diff(t_kyr)))
    record_kyr = n * dt_kyr

    P = np.fft.rfft(p)
    Q = np.fft.rfft(q)
    freq = np.fft.rfftfreq(n, d=dt_kyr)

    period_kyr = np.full_like(freq, np.inf, dtype=float)
    period_kyr[1:] = 1.0 / freq[1:]

    joint_power = np.abs(P) ** 2 + np.abs(Q) ** 2
    total_non_dc_power = float(np.sum(joint_power[1:]))

    i_actual = np.degrees(
        2.0 * np.arcsin(np.clip(np.hypot(p, q), 0.0, 1.0))
    )

    target_period = widgets.FloatText(
        value=253.0,
        description="Target period:",
        layout=widgets.Layout(width="260px"),
    )

    period_half_width = widgets.FloatSlider(
        value=40.0,
        min=5.0,
        max=150.0,
        step=5.0,
        description="Spectrum ±:",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="520px"),
    )

    time_window = widgets.Dropdown(
        options=[
            ("1 Myr", 1.0),
            ("2 Myr", 2.0),
            ("5 Myr", 5.0),
            ("10 Myr", 10.0),
            ("25 Myr", 25.0),
            ("50 Myr", 50.0),
            ("250 Myr", 250.0),
        ],
        value=5.0,
        description="Time window:",
        layout=widgets.Layout(width="320px"),
    )

    time_start = widgets.FloatSlider(
        value=-5.0,
        min=-250.0,
        max=-5.0,
        step=0.1,
        description="Start:",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="760px"),
    )

    display_mode = widgets.ToggleButtons(
        options=[
            ("Spectrum + isolated cycle", "both"),
            ("Spectrum only", "spectrum"),
            ("Isolated cycle only", "cycle"),
            ("p–q phase portrait", "phase"),
        ],
        value="both",
        description="View:",
    )

    show_full = widgets.Checkbox(
        value=True,
        description="Show full La2010a inclination",
        indent=False,
    )

    normalize_cycle = widgets.Checkbox(
        value=False,
        description="Mean-center isolated inclination",
        indent=False,
    )

    previous_button = widgets.Button(description="◀ Previous")
    next_button = widgets.Button(description="Next ▶")
    j2000_button = widgets.Button(description="J2000")
    full_button = widgets.Button(description="Full −250 to 0 Myr")

    metrics = widgets.HTML()
    plot_output = widgets.Output()

    def sync_time_limits():
        width = float(time_window.value)
        time_start.max = -width
        if time_start.value > time_start.max:
            time_start.value = time_start.max

    def selected_bin():
        target = max(1e-9, float(target_period.value))
        valid = np.arange(1, len(freq))
        return int(valid[np.argmin(np.abs(period_kyr[valid] - target))])

    def single_bin_reconstruction(k):
        P_one = np.zeros_like(P)
        Q_one = np.zeros_like(Q)
        P_one[0] = P[0]
        Q_one[0] = Q[0]
        P_one[k] = P[k]
        Q_one[k] = Q[k]
        p_one = np.fft.irfft(P_one, n=n)
        q_one = np.fft.irfft(Q_one, n=n)
        i_one = np.degrees(
            2.0 * np.arcsin(
                np.clip(np.hypot(p_one, q_one), 0.0, 1.0)
            )
        )
        return p_one, q_one, i_one

    def render(*_):
        try:
            sync_time_limits()
            k = selected_bin()
            realized_period = float(period_kyr[k])
            realized_freq = float(freq[k])

            p_amp = 2.0 * abs(P[k]) / n
            q_amp = 2.0 * abs(Q[k]) / n
            p_phase = float(np.angle(P[k]))
            q_phase = float(np.angle(Q[k]))
            energy_fraction = float(joint_power[k] / total_non_dc_power)

            p_one, q_one, i_one = single_bin_reconstruction(k)

            if normalize_cycle.value:
                i_cycle_plot = i_one - np.mean(i_one)
                cycle_label = "Isolated cycle, mean-centered"
                cycle_axis = "Inclination contribution (degrees)"
            else:
                i_cycle_plot = i_one
                cycle_label = "Isolated 253-kyr harmonic"
                cycle_axis = "Inclination (degrees)"

            metrics.value = (
                f"<b>Target period:</b> {float(target_period.value):,.6f} kyr<br>"
                f"<b>Nearest FFT bin:</b> k = {k:,}<br>"
                f"<b>Realized FFT period:</b> {realized_period:,.9f} kyr "
                f"({realized_period/1000.0:.9f} Myr)<br>"
                f"<b>Frequency:</b> {realized_freq:.15e} kyr⁻¹<br>"
                f"<b>Angular frequency:</b> "
                f"{2.0*np.pi*realized_freq:.15e} rad kyr⁻¹<br>"
                f"<b>p amplitude:</b> {p_amp:.15e}; "
                f"<b>phase:</b> {p_phase:+.12f} rad<br>"
                f"<b>q amplitude:</b> {q_amp:.15e}; "
                f"<b>phase:</b> {q_phase:+.12f} rad<br>"
                f"<b>Joint p,q spectral-energy fraction:</b> "
                f"{100.0*energy_fraction:.12f}%<br>"
                f"<b>Record length:</b> {record_kyr:,.3f} kyr; "
                f"<b>frequency resolution:</b> {1.0/record_kyr:.15e} kyr⁻¹"
            )

            left = float(time_start.value)
            right = min(0.0, left + float(time_window.value))
            time_mask = (time_myr >= left) & (time_myr <= right)
            idx = np.flatnonzero(time_mask)

            if len(idx) > 15000:
                idx = idx[::max(1, len(idx)//15000)]

            x_time = time_myr[idx]
            mode = display_mode.value

            if mode == "both":
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    vertical_spacing=0.12,
                    row_heights=[0.43, 0.57],
                    subplot_titles=(
                        "FFT spectrum near the target period",
                        "Isolated target harmonic in the time domain",
                    ),
                )
                spectrum_row = 1
                cycle_row = 2
            else:
                fig = go.Figure()
                spectrum_row = None
                cycle_row = None

            if mode in ("both", "spectrum"):
                half = float(period_half_width.value)
                low = max(1e-6, float(target_period.value) - half)
                high = float(target_period.value) + half
                spectral_mask = (
                    np.isfinite(period_kyr)
                    & (period_kyr >= low)
                    & (period_kyr <= high)
                )
                sidx = np.flatnonzero(spectral_mask)
                order = np.argsort(period_kyr[sidx])
                sidx = sidx[order]
                spectral_amplitude = 2.0 * np.sqrt(joint_power[sidx]) / n

                spectrum_trace = go.Scatter(
                    x=period_kyr[sidx],
                    y=spectral_amplitude,
                    mode="lines+markers",
                    name="Joint p,q amplitude",
                    marker=dict(size=5),
                    hovertemplate=(
                        "Period: %{x:.9f} kyr"
                        "<br>Joint amplitude: %{y:.12e}"
                        "<extra></extra>"
                    ),
                )

                selected_trace = go.Scatter(
                    x=[realized_period],
                    y=[2.0*np.sqrt(joint_power[k])/n],
                    mode="markers",
                    name="Selected bin",
                    marker=dict(size=12, symbol="diamond"),
                    hovertemplate=(
                        f"k = {k}<br>"
                        "Period: %{x:.9f} kyr"
                        "<br>Joint amplitude: %{y:.12e}"
                        "<extra></extra>"
                    ),
                )

                if mode == "both":
                    fig.add_trace(spectrum_trace, row=spectrum_row, col=1)
                    fig.add_trace(selected_trace, row=spectrum_row, col=1)
                    fig.update_xaxes(title_text="Period (kyr)", row=spectrum_row, col=1)
                    fig.update_yaxes(title_text="Joint amplitude", row=spectrum_row, col=1)
                else:
                    fig.add_trace(spectrum_trace)
                    fig.add_trace(selected_trace)
                    fig.update_xaxes(title_text="Period (kyr)")
                    fig.update_yaxes(title_text="Joint p,q amplitude")
                    fig.update_layout(title="FFT spectrum near the 253-kyr cycle")

            if mode in ("both", "cycle"):
                if show_full.value:
                    full_trace = go.Scattergl(
                        x=x_time,
                        y=i_actual[idx],
                        mode="lines",
                        name="Full La2010a inclination",
                        line=dict(color="#52D6FF", width=1.8),
                        hovertemplate=(
                            "Time: %{x:.6f} Myr"
                            "<br>Inclination: %{y:.9f}°"
                            "<extra></extra>"
                        ),
                    )
                    if mode == "both":
                        fig.add_trace(full_trace, row=cycle_row, col=1)
                    else:
                        fig.add_trace(full_trace)

                cycle_trace = go.Scattergl(
                    x=x_time,
                    y=i_cycle_plot[idx],
                    mode="lines",
                    name=cycle_label,
                    line=dict(color="#35E0A1", width=2.2),
                    hovertemplate=(
                        "Time: %{x:.6f} Myr"
                        "<br>Value: %{y:.12f}°"
                        "<extra></extra>"
                    ),
                )

                if mode == "both":
                    fig.add_trace(cycle_trace, row=cycle_row, col=1)
                    fig.update_xaxes(title_text="Millions of years relative to J2000", row=cycle_row, col=1)
                    fig.update_yaxes(title_text=cycle_axis, row=cycle_row, col=1)
                else:
                    fig.add_trace(cycle_trace)
                    fig.update_xaxes(title_text="Millions of years relative to J2000")
                    fig.update_yaxes(title_text=cycle_axis)
                    fig.update_layout(title="Isolated 253-kyr Fourier cycle")

            if mode == "phase":
                fig.add_trace(
                    go.Scattergl(
                        x=p_one[idx],
                        y=q_one[idx],
                        mode="lines",
                        name="Selected harmonic trajectory",
                        line=dict(color="#35E0A1", width=2.0),
                        hovertemplate=(
                            "p: %{x:.12e}"
                            "<br>q: %{y:.12e}"
                            "<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(title="p–q phase portrait of the selected 253-kyr harmonic")
                fig.update_xaxes(title_text="p")
                fig.update_yaxes(title_text="q", scaleanchor="x", scaleratio=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                height=860 if mode == "both" else 720,
                margin=dict(l=80, r=40, t=90, b=80),
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(0,0,0,0.78)",
                    bordercolor="#28364A",
                    borderwidth=1,
                ),
            )

            fig.update_xaxes(
                gridcolor="#28364A",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#1F7A4D",
                spikethickness=2,
            )
            fig.update_yaxes(
                gridcolor="#28364A",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#1F7A4D",
                spikethickness=2,
            )

            with plot_output:
                clear_output(wait=True)
                display(fig)

            status.value = "<b style='color:#35E0A1'>253-kyr spectral-cycle explorer ready.</b>"

        except Exception:
            with plot_output:
                clear_output(wait=True)
                display(HTML("<pre style='color:#ff6b6b'>" + traceback.format_exc() + "</pre>"))

    def move_previous(_):
        time_start.value = max(-250.0, time_start.value - float(time_window.value))

    def move_next(_):
        time_start.value = min(time_start.max, time_start.value + float(time_window.value))

    def go_j2000(_):
        time_start.value = -float(time_window.value)

    def go_full(_):
        time_window.value = 250.0
        time_start.value = -250.0

    previous_button.on_click(move_previous)
    next_button.on_click(move_next)
    j2000_button.on_click(go_j2000)
    full_button.on_click(go_full)

    for control in (
        target_period,
        period_half_width,
        time_window,
        time_start,
        display_mode,
        show_full,
        normalize_cycle,
    ):
        control.observe(render, names="value")

    controls = widgets.VBox([
        metrics,
        display_mode,
        widgets.HBox([target_period, period_half_width]),
        widgets.HBox([time_window, show_full, normalize_cycle]),
        time_start,
        widgets.HBox([
            previous_button,
            next_button,
            j2000_button,
            full_button,
        ]),
    ])

    display(widgets.VBox([controls, plot_output]))
    render()

except Exception:
    display(HTML("<pre style='color:#ff6b6b'>" + traceback.format_exc() + "</pre>"))
