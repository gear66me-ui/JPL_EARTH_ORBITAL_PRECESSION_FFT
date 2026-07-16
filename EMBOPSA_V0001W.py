# EMBOPSA_V0001W
# Colab/tablet-compatible Earth orbital inclination explorer.
# NO AI-GENERATED IMAGES.
# Navigation uses ipywidgets; touch readout uses native Plotly hover/tap.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0001W"

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

df["age_myr"] = -df["t_kyr"] / 1000.0
df["age_years"] = df["age_myr"] * 1_000_000.0
df["pole_radius"] = np.hypot(df["p"], df["q"])
df["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(df["pole_radius"], 0.0, 1.0))
)

MAX_AGE = float(df["age_myr"].max())
TODAY_I = float(df.loc[df["t_kyr"].abs().idxmin(), "inclination_deg"])

global_max_row = df.loc[df["inclination_deg"].idxmax()]
GLOBAL_MAX_AGE = float(global_max_row["age_myr"])
GLOBAL_MAX_I = float(global_max_row["inclination_deg"])

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
        ("Full 250 million years", MAX_AGE),
    ],
    value=5.0,
    description="Window:",
    layout=widgets.Layout(width="420px"),
)

position = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=MAX_AGE - 5.0,
    step=0.1,
    description="Position:",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="760px"),
)

today_button = widgets.Button(description="Today")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
maximum_button = widgets.Button(description="Go to maximum")
full_button = widgets.Button(description="Full record")
status = widgets.HTML()
plot_output = widgets.Output()

def sync_position_limit():
    width = float(window_size.value)
    position.max = max(0.0, MAX_AGE - width)
    if position.value > position.max:
        position.value = position.max

def choose_stride(width_myr: float) -> int:
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
    end = min(MAX_AGE, start + width)

    subset = df[
        (df["age_myr"] >= start)
        & (df["age_myr"] <= end)
    ].copy()

    stride = choose_stride(width)
    plotted = subset.iloc[::stride].copy()

    local_max_row = subset.loc[subset["inclination_deg"].idxmax()]
    local_max_age = float(local_max_row["age_myr"])
    local_max_i = float(local_max_row["inclination_deg"])
    local_min_i = float(subset["inclination_deg"].min())

    status.value = (
        f"<b>Window:</b> {start:,.3f}–{end:,.3f} Myr before J2000 "
        f"&nbsp; | &nbsp; <b>Touch any cyan point for the exact time and angle.</b>"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=plotted["age_myr"],
            y=plotted["inclination_deg"],
            customdata=np.column_stack([
                plotted["age_years"],
                plotted["p"],
                plotted["q"],
            ]),
            mode="lines+markers",
            name="Earth orbital inclination",
            line=dict(color="#52D6FF", width=1.5),
            marker=dict(color="#52D6FF", size=5),
            hovertemplate=(
                "<b>Earth orbital inclination</b>"
                "<br>Age before J2000: %{x:.6f} Myr"
                "<br>Years before J2000: %{customdata[0]:,.0f}"
                "<br>Inclination: %{y:.6f}°"
                "<br>p: %{customdata[1]:.9f}"
                "<br>q: %{customdata[2]:.9f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[start, end],
            y=[TODAY_I, TODAY_I],
            mode="lines",
            name=f"J2000 inclination = {TODAY_I:.6f}°",
            line=dict(color="#FFD166", width=1.2, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[start, end],
            y=[3.0, 3.0],
            mode="lines",
            name="3° reference",
            line=dict(color="#FF5C7A", width=1.1, dash="dot"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[local_max_age],
            y=[local_max_i],
            mode="markers",
            name="Window maximum",
            marker=dict(color="#35E0A1", size=11),
            hovertemplate=(
                "<b>Window maximum</b>"
                "<br>Age before J2000: %{x:.6f} Myr"
                "<br>Inclination: %{y:.6f}°"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(
            text="Earth Orbital-Plane Inclination Window Explorer",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Millions of years before J2000",
            range=[start, end],
            gridcolor="#28364A",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#FFD166",
            spikethickness=1,
        ),
        yaxis=dict(
            title="Inclination (degrees)",
            range=[max(0.0, local_min_i - 0.15), max(3.15, local_max_i + 0.15)],
            gridcolor="#28364A",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#FFD166",
            spikethickness=1,
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

    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    with plot_output:
        clear_output(wait=True)
        display(fig)

def go_today(_):
    position.value = 0.0

def go_previous(_):
    position.value = max(
        0.0,
        float(position.value) - float(window_size.value),
    )

def go_next(_):
    position.value = min(
        float(position.max),
        float(position.value) + float(window_size.value),
    )

def go_maximum(_):
    width = float(window_size.value)
    target = max(0.0, GLOBAL_MAX_AGE - width / 2.0)
    position.value = min(float(position.max), target)

def go_full(_):
    window_size.value = MAX_AGE
    position.value = 0.0

window_size.observe(render, names="value")
position.observe(render, names="value")

today_button.on_click(go_today)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
maximum_button.on_click(go_maximum)
full_button.on_click(go_full)

controls = widgets.VBox([
    status,
    window_size,
    position,
    widgets.HBox([
        today_button,
        previous_button,
        next_button,
        maximum_button,
        full_button,
    ]),
])

display(widgets.VBox([controls, plot_output]))
render()
