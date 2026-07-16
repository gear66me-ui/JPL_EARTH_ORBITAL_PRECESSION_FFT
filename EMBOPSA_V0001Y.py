# EMBOPSA_V0001Y
# Earth orbital inclination explorer, -250 Myr to +250 Myr axis.
# Past side uses La2010a data. Future side is explicitly marked unavailable.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = "V0001Y"

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

# Signed time: negative = past, zero = J2000, positive = future.
df["time_myr"] = df["t_kyr"] / 1000.0
df["time_years"] = df["time_myr"] * 1_000_000.0
df["pole_radius"] = np.hypot(df["p"], df["q"])
df["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(df["pole_radius"], 0.0, 1.0))
)

TODAY_I = float(df.loc[df["t_kyr"].abs().idxmin(), "inclination_deg"])
PAST_MIN = float(df["time_myr"].min())
PAST_MAX = 0.0
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
    layout=widgets.Layout(width="420px"),
)

position = widgets.FloatSlider(
    value=-10.0,
    min=FULL_MIN,
    max=FULL_MAX - 10.0,
    step=0.1,
    description="Start:",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="780px"),
)

past_button = widgets.Button(description="−250 Myr")
today_button = widgets.Button(description="Today")
future_button = widgets.Button(description="+250 Myr")
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
    end = min(FULL_MAX, start + width)

    subset = df[
        (df["time_myr"] >= start)
        & (df["time_myr"] <= min(end, 0.0))
    ].copy()

    stride = choose_stride(width)
    plotted = subset.iloc[::stride].copy()

    status.value = (
        f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000 "
        f"&nbsp; | &nbsp; <b>Negative = past, positive = future.</b>"
    )

    fig = go.Figure()

    if not plotted.empty:
        fig.add_trace(
            go.Scattergl(
                x=plotted["time_myr"],
                y=plotted["inclination_deg"],
                customdata=np.column_stack([
                    plotted["time_years"],
                    plotted["p"],
                    plotted["q"],
                ]),
                mode="lines",
                name="La2010a Earth orbital inclination",
                line=dict(color="#52D6FF", width=1.6),
                hovertemplate=(
                    "<b>Earth orbital inclination</b>"
                    "<br>Time from J2000: %{x:.6f} Myr"
                    "<br>Years from J2000: %{customdata[0]:,.0f}"
                    "<br>Inclination: %{y:.6f}°"
                    "<br>p: %{customdata[1]:.9f}"
                    "<br>q: %{customdata[2]:.9f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=TODAY_I,
        line=dict(color="#FFD166", width=1.1, dash="dash"),
        annotation_text=f"J2000 = {TODAY_I:.6f}°",
        annotation_position="top left",
    )

    fig.add_vline(
        x=0.0,
        line=dict(color="#35E0A1", width=2),
        annotation_text="J2000",
        annotation_position="top",
    )

    if end > 0:
        future_left = max(start, 0.0)
        fig.add_vrect(
            x0=future_left,
            x1=end,
            fillcolor="rgba(120,120,120,0.18)",
            line_width=0,
            annotation_text="Future: no La2010a data",
            annotation_position="top left",
        )

    if start < PAST_MIN:
        fig.add_vrect(
            x0=start,
            x1=min(end, PAST_MIN),
            fillcolor="rgba(120,120,120,0.18)",
            line_width=0,
            annotation_text="Outside La2010a range",
            annotation_position="top left",
        )

    y_values = plotted["inclination_deg"] if not plotted.empty else pd.Series([TODAY_I])
    y_min = max(0.0, float(y_values.min()) - 0.15)
    y_max = max(3.15, float(y_values.max()) + 0.15)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        title=dict(
            text="Earth Orbital-Plane Inclination: −250 Myr to +250 Myr",
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
    position.value = max(FULL_MIN, min(position.max, FULL_MAX - width))

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
