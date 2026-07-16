# EMBOPSA_V0002K
# Exact La2010a FFT equation and coefficient explorer.
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
from IPython.display import display, HTML, Math, clear_output

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

pio.renderers.default = "colab"

VERSION = "V0002K"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)

startup = widgets.HTML(
    "<b style='color:#35E0A1'>Loading La2010a and computing exact FFT coefficients…</b>"
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

    for column in raw.columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)

    t_kyr = raw["t_kyr"].to_numpy(float)
    p = raw["p"].to_numpy(float)
    q = raw["q"].to_numpy(float)

    N = len(raw)
    dt_kyr = float(np.median(np.diff(t_kyr)))
    T_kyr = N * dt_kyr

    P = np.fft.rfft(p)
    Q = np.fft.rfft(q)
    freq = np.fft.rfftfreq(N, d=dt_kyr)

    p_exact = np.fft.irfft(P, n=N)
    q_exact = np.fft.irfft(Q, n=N)

    i_actual = np.degrees(
        2.0 * np.arcsin(np.clip(np.hypot(p, q), 0.0, 1.0))
    )
    i_exact = np.degrees(
        2.0 * np.arcsin(np.clip(np.hypot(p_exact, q_exact), 0.0, 1.0))
    )

    def coefficient_table(coeff, component):
        amplitude = np.abs(coeff)
        phase = np.angle(coeff)
        period = np.full_like(freq, np.inf, dtype=float)
        period[1:] = 1.0 / freq[1:]

        return pd.DataFrame({
            "component": component,
            "k": np.arange(len(coeff)),
            "frequency_per_kyr": freq,
            "period_kyr": period,
            "real": coeff.real,
            "imag": coeff.imag,
            "magnitude": amplitude,
            "phase_rad": phase,
            "phase_deg": np.degrees(phase),
            "cosine_amplitude": np.where(
                np.arange(len(coeff)) == 0,
                amplitude / N,
                2.0 * amplitude / N,
            ),
        })

    table_p = coefficient_table(P, "p")
    table_q = coefficient_table(Q, "q")

    startup.value = (
        "<b style='color:#35E0A1'>Exact FFT coefficient system ready.</b>"
    )

    equation = widgets.HTMLMath(
        value=(
            r"<b>Exact discrete reconstruction</b><br>"
            r"\(P_k=\sum_{n=0}^{N-1}p_n e^{-i2\pi kn/N}\), "
            r"\(Q_k=\sum_{n=0}^{N-1}q_n e^{-i2\pi kn/N}\)<br>"
            r"\(p_n=\frac{1}{N}\sum_{k=0}^{N-1}P_k e^{i2\pi kn/N}\), "
            r"\(q_n=\frac{1}{N}\sum_{k=0}^{N-1}Q_k e^{i2\pi kn/N}\)<br>"
            r"\(i_n=2\arcsin\sqrt{p_n^2+q_n^2}\)"
        )
    )

    summary = widgets.HTML(
        value=(
            f"<b>N = {N:,} samples</b><br>"
            f"Sampling interval Δt = {dt_kyr:.6f} kyr<br>"
            f"Record length T = {T_kyr:,.6f} kyr "
            f"({T_kyr/1000.0:,.6f} Myr)<br>"
            f"Nonredundant coefficients per component = {len(P):,}<br>"
            f"Fundamental frequency = {1.0/T_kyr:.15e} kyr⁻¹<br>"
            f"Maximum reconstruction error in inclination = "
            f"{np.max(np.abs(i_exact-i_actual)):.15e}°"
        )
    )

    component = widgets.ToggleButtons(
        options=[("p coefficients", "p"), ("q coefficients", "q")],
        value="p",
        description="Component:",
    )

    ranking = widgets.Dropdown(
        options=[
            ("Strongest magnitude", "magnitude"),
            ("Lowest frequency", "frequency"),
            ("Shortest period", "shortest"),
            ("Harmonic number", "index"),
        ],
        value="magnitude",
        description="Order:",
        layout=widgets.Layout(width="350px"),
    )

    top_n = widgets.IntSlider(
        value=30,
        min=5,
        max=200,
        step=5,
        description="Terms:",
        continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )

    harmonic_index = widgets.BoundedIntText(
        value=1,
        min=0,
        max=len(P)-1,
        step=1,
        description="Harmonic k:",
        layout=widgets.Layout(width="260px"),
    )

    reconstruction_terms = widgets.IntSlider(
        value=50,
        min=1,
        max=1000,
        step=1,
        description="Top harmonics:",
        continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )

    window_myr = widgets.Dropdown(
        options=[
            ("0.1 Myr", 0.1),
            ("0.5 Myr", 0.5),
            ("1 Myr", 1.0),
            ("5 Myr", 5.0),
            ("10 Myr", 10.0),
            ("25 Myr", 25.0),
            ("50 Myr", 50.0),
            ("250 Myr", 250.0),
        ],
        value=1.0,
        description="Window:",
    )

    center_myr = widgets.FloatSlider(
        value=-0.5,
        min=-250.0,
        max=0.0,
        step=0.1,
        description="Center:",
        continuous_update=False,
        layout=widgets.Layout(width="700px"),
    )

    view = widgets.ToggleButtons(
        options=[
            ("Coefficient spectrum", "spectrum"),
            ("Single coefficient", "single"),
            ("Partial reconstruction", "partial"),
        ],
        value="spectrum",
        description="View:",
    )

    details = widgets.HTML()
    plot_output = widgets.Output()

    def active_table():
        return table_p if component.value == "p" else table_q

    def ordered_table():
        table = active_table().copy()

        if ranking.value == "magnitude":
            table = table.sort_values("magnitude", ascending=False)
        elif ranking.value == "frequency":
            table = table.sort_values("frequency_per_kyr", ascending=True)
        elif ranking.value == "shortest":
            table = table.sort_values("period_kyr", ascending=True)
        else:
            table = table.sort_values("k", ascending=True)

        return table.head(int(top_n.value))

    def selected_coefficients(coeff, count):
        idx = np.argsort(np.abs(coeff))[::-1][:count]
        mask = np.zeros_like(coeff, dtype=bool)
        mask[idx] = True
        kept = np.where(mask, coeff, 0.0 + 0.0j)
        return kept, idx

    def render(*_):
        try:
            fig = go.Figure()

            if view.value == "spectrum":
                table = ordered_table()

                fig.add_trace(
                    go.Bar(
                        x=table["k"],
                        y=table["cosine_amplitude"],
                        customdata=np.column_stack([
                            table["frequency_per_kyr"],
                            table["period_kyr"],
                            table["phase_deg"],
                            table["real"],
                            table["imag"],
                        ]),
                        name=f"{component.value} coefficients",
                        hovertemplate=(
                            "k = %{x}"
                            "<br>Amplitude = %{y:.12e}"
                            "<br>Frequency = %{customdata[0]:.12e} kyr⁻¹"
                            "<br>Period = %{customdata[1]:.6f} kyr"
                            "<br>Phase = %{customdata[2]:.6f}°"
                            "<br>Real = %{customdata[3]:.12e}"
                            "<br>Imag = %{customdata[4]:.12e}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    title=f"FFT coefficient spectrum — component {component.value}",
                    xaxis_title="Harmonic index k",
                    yaxis_title="Cosine-equivalent amplitude",
                )

            elif view.value == "single":
                table = active_table()
                k = int(harmonic_index.value)
                row = table.iloc[k]

                details.value = (
                    f"<b>Component {component.value}, harmonic k = {k}</b><br>"
                    f"Coefficient = {row['real']:+.15e} "
                    f"{row['imag']:+.15e}i<br>"
                    f"Magnitude = {row['magnitude']:.15e}<br>"
                    f"Cosine-equivalent amplitude = "
                    f"{row['cosine_amplitude']:.15e}<br>"
                    f"Phase = {row['phase_rad']:+.15e} rad "
                    f"({row['phase_deg']:+.9f}°)<br>"
                    f"Frequency = {row['frequency_per_kyr']:.15e} kyr⁻¹<br>"
                    f"Period = {row['period_kyr']:.9f} kyr"
                )

                phase = np.linspace(0.0, 2.0*np.pi, 1000)
                amplitude = float(row["cosine_amplitude"])
                phi = float(row["phase_rad"])
                term = amplitude * np.cos(phase + phi)

                fig.add_trace(
                    go.Scatter(
                        x=phase,
                        y=term,
                        mode="lines",
                        name=f"k={k}",
                    )
                )

                fig.update_layout(
                    title=f"Single Fourier term — {component.value}, k={k}",
                    xaxis_title="Phase angle (radians)",
                    yaxis_title=f"{component.value} contribution",
                )

            else:
                count = int(reconstruction_terms.value)
                P_kept, idx_p = selected_coefficients(P, count)
                Q_kept, idx_q = selected_coefficients(Q, count)

                p_partial = np.fft.irfft(P_kept, n=N)
                q_partial = np.fft.irfft(Q_kept, n=N)
                i_partial = np.degrees(
                    2.0 * np.arcsin(
                        np.clip(np.hypot(p_partial, q_partial), 0.0, 1.0)
                    )
                )

                half = float(window_myr.value) / 2.0
                center = float(center_myr.value)
                left = max(-250.0, center-half)
                right = min(0.0, center+half)

                mask = (
                    (raw["t_kyr"].to_numpy(float)/1000.0 >= left)
                    & (raw["t_kyr"].to_numpy(float)/1000.0 <= right)
                )

                x = raw.loc[mask, "t_kyr"].to_numpy(float)/1000.0
                actual = i_actual[mask]
                partial = i_partial[mask]

                stride = max(1, len(x)//10000)
                x = x[::stride]
                actual = actual[::stride]
                partial = partial[::stride]

                residual = partial - actual
                rmse = float(np.sqrt(np.mean(residual**2)))
                r = float(np.corrcoef(actual, partial)[0, 1])

                details.value = (
                    f"<b>Partial reconstruction</b><br>"
                    f"Strongest coefficients retained per component = {count}<br>"
                    f"Pearson r in visible window = {r:.12f}<br>"
                    f"RMSE in visible window = {rmse:.12e}°"
                )

                fig.add_trace(
                    go.Scattergl(
                        x=x,
                        y=actual,
                        mode="lines",
                        name="Actual La2010a",
                        line=dict(width=2.0),
                    )
                )

                fig.add_trace(
                    go.Scattergl(
                        x=x,
                        y=partial,
                        mode="lines",
                        name=f"Top-{count} FFT reconstruction",
                        line=dict(width=1.6, dash="dash"),
                    )
                )

                fig.update_layout(
                    title="La2010a vs partial Fourier reconstruction",
                    xaxis_title="Millions of years relative to J2000",
                    yaxis_title="Inclination (degrees)",
                )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
                height=720,
                margin=dict(l=80, r=40, t=80, b=80),
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

        except Exception:
            with plot_output:
                clear_output(wait=True)
                display(
                    HTML(
                        "<pre style='color:#ff6b6b'>"
                        + traceback.format_exc()
                        + "</pre>"
                    )
                )

    for control in (
        component,
        ranking,
        top_n,
        harmonic_index,
        reconstruction_terms,
        window_myr,
        center_myr,
        view,
    ):
        control.observe(render, names="value")

    controls = widgets.VBox([
        equation,
        summary,
        view,
        widgets.HBox([component, ranking]),
        top_n,
        harmonic_index,
        reconstruction_terms,
        window_myr,
        center_myr,
        details,
        plot_output,
    ])

    display(controls)
    render()

except Exception:
    display(
        HTML(
            "<pre style='color:#ff6b6b'>"
            + traceback.format_exc()
            + "</pre>"
        )
    )
