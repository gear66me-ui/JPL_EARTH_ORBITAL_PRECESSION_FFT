# EMBOPSA_V0001U
# Colab-compatible touch/click cycle measurement using Plotly FigureWidget.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import sys
import subprocess
import importlib.util

for package in ("plotly", "anywidget"):
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

VERSION = "V0001U"

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
df["pole_radius"] = np.hypot(df["p"], df["q"])
df["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(df["pole_radius"], 0.0, 1.0))
)

MAX_AGE = float(df["age_myr"].max())
TODAY_I = float(df.loc[df["t_kyr"].abs().idxmin(), "inclination_deg"])

global_max_row = df.loc[df["inclination_deg"].idxmax()]
GLOBAL_MAX_AGE = float(global_max_row["age_myr"])
GLOBAL_MAX_I = float(global_max_row["inclination_deg"])

window_width = widgets.SelectionSlider(
    options=[
        ("0.5 Myr", 0.5),
        ("1 Myr", 1.0),
        ("2 Myr", 2.0),
        ("5 Myr", 5.0),
        ("10 Myr", 10.0),
        ("25 Myr", 25.0),
        ("50 Myr", 50.0),
        ("100 Myr", 100.0),
        ("250 Myr", MAX_AGE),
    ],
    value=10.0,
    description="Window",
    continuous_update=False,
    layout=widgets.Layout(width="760px"),
)

window_start = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=max(0.0, MAX_AGE - 10.0),
    step=0.1,
    description="Position",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="760px"),
)

sampling = widgets.SelectionSlider(
    options=[
        ("1 kyr", 1),
        ("5 kyr", 5),
        ("10 kyr", 10),
        ("25 kyr", 25),
        ("50 kyr", 50),
        ("100 kyr", 100),
    ],
    value=5,
    description="Sampling",
    continuous_update=False,
    layout=widgets.Layout(width="760px"),
)

today_button = widgets.Button(description="Today")
maximum_button = widgets.Button(description="Go to maximum")
previous_button = widgets.Button(description="◀ Previous")
next_button = widgets.Button(description="Next ▶")
clear_button = widgets.Button(description="Clear A–B", button_style="danger")

status = widgets.HTML()
measurement = widgets.HTML(
    "<b>Touch measurement:</b> Tap one curve point for A, then another for B."
)

fig = go.FigureWidget()
selected = []
current = pd.DataFrame()

def sync_limits():
    width = float(window_width.value)
    window_start.max = max(0.0, MAX_AGE - width)
    if window_start.value > window_start.max:
        window_start.value = window_start.max

def clear_measurement(_=None):
    selected.clear()
    measurement.value = (
        "<b>Touch measurement:</b> Tap one curve point for A, then another for B."
    )
    with fig.batch_update():
        fig.layout.shapes = ()
        fig.layout.annotations = ()

def marker_shape(x, color):
    return dict(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=2, dash="dash"),
    )

def marker_annotation(x, label, color):
    return dict(
        x=x,
        y=1,
        xref="x",
        yref="paper",
        text=f"<b>{label}</b><br>{x:.6f} Myr",
        showarrow=False,
        textangle=-90,
        xanchor="right",
        yanchor="top",
        font=dict(color=color, size=12),
        bgcolor="rgba(0,0,0,0.75)",
        bordercolor=color,
    )

def on_curve_click(trace, points, state):
    if not points.point_inds:
        return

    point_index = int(points.point_inds[0])
    x = float(trace.x[point_index])
    y = float(trace.y[point_index])

    if len(selected) >= 2:
        clear_measurement()

    selected.append((x, y))

    shapes = []
    annotations = []

    if len(selected) >= 1:
        a_x, a_y = selected[0]
        shapes.append(marker_shape(a_x, "#FFD166"))
        annotations.append(marker_annotation(a_x, "A", "#FFD166"))
        measurement.value = (
            f"<b>A:</b> {a_x:,.6f} Myr before J2000 "
            f"({a_y:.6f}°). Tap B."
        )

    if len(selected) == 2:
        b_x, b_y = selected[1]
        shapes.append(marker_shape(b_x, "#B388FF"))
        annotations.append(marker_annotation(b_x, "B", "#B388FF"))

        a_x = selected[0][0]
        delta_myr = abs(b_x - a_x)
        delta_kyr = delta_myr * 1000.0
        delta_years = delta_myr * 1_000_000.0

        measurement.value = (
            f"<b>A:</b> {a_x:,.6f} Myr &nbsp; | &nbsp; "
            f"<b>B:</b> {b_x:,.6f} Myr &nbsp; | &nbsp; "
            f"<b>Separation:</b> {delta_years:,.0f} years "
            f"= {delta_kyr:,.3f} kyr "
            f"= {delta_myr:,.6f} Myr"
        )

    with fig.batch_update():
        fig.layout.shapes = tuple(shapes)
        fig.layout.annotations = tuple(annotations)

def render(*_):
    global current

    sync_limits()
    clear_measurement()

    start = float(window_start.value)
    width = float(window_width.value)
    end = min(MAX_AGE, start + width)

    subset = df[
        (df["age_myr"] >= start)
        & (df["age_myr"] <= end)
    ].copy()

    stride = max(1, int(sampling.value))
    current = subset.iloc[::stride].copy()

    local_max_row = subset.loc[subset["inclination_deg"].idxmax()]
    local_max_age = float(local_max_row["age_myr"])
    local_max_i = float(local_max_row["inclination_deg"])
    local_min_i = float(subset["inclination_deg"].min())

    status.value = (
        f"<b>Window:</b> {start:,.3f}–{end:,.3f} Myr before J2000 "
        f"&nbsp; | &nbsp; <b>Tap two points on the cyan curve.</b>"
    )

    with fig.batch_update():
        fig.data = []

        fig.add_scatter(
            x=current["age_myr"],
            y=current["inclination_deg"],
            mode="lines+markers",
            name="Earth orbital inclination",
            line=dict(color="#52D6FF", width=1.4),
            marker=dict(color="#52D6FF", size=5, opacity=0.75),
            hovertemplate=(
                "Age: %{x:.6f} Myr before J2000"
                "<br>Inclination: %{y:.6f}°"
                "<extra></extra>"
            ),
        )

        fig.add_scatter(
            x=[start, end],
            y=[TODAY_I, TODAY_I],
            mode="lines",
            name=f"J2000 = {TODAY_I:.6f}°",
            line=dict(color="#FFD166", width=1.2, dash="dash"),
            hoverinfo="skip",
        )

        fig.add_scatter(
            x=[start, end],
            y=[3.0, 3.0],
            mode="lines",
            name="3° reference",
            line=dict(color="#FF5C7A", width=1.1, dash="dot"),
            hoverinfo="skip",
        )

        fig.add_scatter(
            x=[local_max_age],
            y=[local_max_i],
            mode="markers",
            name="Window maximum",
            marker=dict(color="#35E0A1", size=10),
            hovertemplate=(
                "Window maximum"
                "<br>Age: %{x:.6f} Myr"
                "<br>Inclination: %{y:.6f}°"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            title=dict(
                text="Earth Orbital Inclination — Touch Two Points to Measure a Cycle",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Millions of years before J2000",
                range=[start, end],
                gridcolor="#28364A",
                zeroline=False,
            ),
            yaxis=dict(
                title="Inclination (degrees)",
                range=[max(0.0, local_min_i - 0.2), max(3.15, local_max_i + 0.2)],
                gridcolor="#28364A",
                zeroline=False,
            ),
            height=680,
            margin=dict(l=70, r=30, t=85, b=70),
            hovermode="closest",
            clickmode="event+select",
            legend=dict(
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="#28364A",
                borderwidth=1,
            ),
        )

    fig.data[0].on_click(on_curve_click)

def go_today(_):
    window_start.value = 0.0

def go_maximum(_):
    width = float(window_width.value)
    window_start.value = min(
        max(0.0, GLOBAL_MAX_AGE - width / 2.0),
        window_start.max,
    )

def go_previous(_):
    window_start.value = max(
        0.0,
        window_start.value - float(window_width.value),
    )

def go_next(_):
    window_start.value = min(
        window_start.max,
        window_start.value + float(window_width.value),
    )

window_width.observe(render, names="value")
window_start.observe(render, names="value")
sampling.observe(render, names="value")

today_button.on_click(go_today)
maximum_button.on_click(go_maximum)
previous_button.on_click(go_previous)
next_button.on_click(go_next)
clear_button.on_click(clear_measurement)

controls = widgets.VBox([
    status,
    measurement,
    window_width,
    window_start,
    sampling,
    widgets.HBox([
        today_button,
        maximum_button,
        previous_button,
        next_button,
        clear_button,
    ]),
])

display(widgets.VBox([controls, fig]))
render()
