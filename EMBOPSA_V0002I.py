# EMBOPSA_V0002I
# Full-record La2010a FFT reconstruction audit and periodic continuation.
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
VERSION = "V0002I"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)
FULL_MIN, FULL_MAX = -250.0, 250.0

startup = widgets.HTML("<b style='color:#35E0A1'>Loading La2010a and auditing the full-record FFT…</b>")
display(startup)

try:
    response = requests.get(DATA_URL, timeout=300)
    response.raise_for_status()
    raw = pd.read_csv(
        io.StringIO(response.text), sep=r"\s+", header=None,
        names=["t_kyr", "a", "l", "k", "h", "q", "p"], engine="python"
    )
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)

    t = raw["t_kyr"].to_numpy(float)
    p = raw["p"].to_numpy(float)
    q = raw["q"].to_numpy(float)
    n = len(raw)
    dt = float(np.median(np.diff(t)))

    p_fft = np.fft.rfft(p)
    q_fft = np.fft.rfft(q)
    p_rec = np.fft.irfft(p_fft, n=n)
    q_rec = np.fft.irfft(q_fft, n=n)

    i_actual = np.degrees(2.0*np.arcsin(np.clip(np.hypot(p, q), 0.0, 1.0)))
    i_fft = np.degrees(2.0*np.arcsin(np.clip(np.hypot(p_rec, q_rec), 0.0, 1.0)))

    residual = i_fft - i_actual
    r_i = float(pearsonr(i_actual, i_fft).statistic)
    rmse_i = float(np.sqrt(np.mean(residual**2)))
    mae_i = float(np.mean(np.abs(residual)))
    max_i = float(np.max(np.abs(residual)))
    rmse_p = float(np.sqrt(np.mean((p_rec-p)**2)))
    rmse_q = float(np.sqrt(np.mean((q_rec-q)**2)))

    raw["time_myr"] = t/1000.0
    raw["inclination_actual"] = i_actual
    raw["inclination_fft"] = i_fft

    startup.value = "<b style='color:#35E0A1'>Full-record FFT audit complete.</b>"

    metrics = widgets.HTML(value=(
        "<b>Corrected 3 — complete La2010a FFT reconstruction</b><br>"
        f"Samples = {n:,}; cadence = {dt:.0f} kyr; retained FFT bins = {len(p_fft):,}<br>"
        f"Inclination: Pearson r = {r_i:.12f}; R² = {r_i*r_i:.12f}; "
        f"RMSE = {rmse_i:.12e}°; MAE = {mae_i:.12e}°; max |error| = {max_i:.12e}°<br>"
        f"p RMSE = {rmse_p:.12e}; q RMSE = {rmse_q:.12e}<br>"
        "<b style='color:#ff9f43'>Historical agreement is an in-sample reconstruction. "
        "The future is only the mathematical periodic continuation of that spectrum.</b>"
    ))

    window = widgets.Dropdown(
        options=[("100 thousand years",0.1),("250 thousand years",0.25),
                 ("500 thousand years",0.5),("1 million years",1.0),
                 ("2 million years",2.0),("5 million years",5.0),
                 ("10 million years",10.0),("25 million years",25.0),
                 ("50 million years",50.0),("100 million years",100.0),
                 ("250 million years",250.0),("Full 500 million years",500.0)],
        value=1.0, description="Window:", layout=widgets.Layout(width="440px")
    )
    position = widgets.FloatSlider(
        value=-1.0, min=FULL_MIN, max=FULL_MAX-1.0, step=0.01,
        description="Start:", continuous_update=False, readout_format=".2f",
        layout=widgets.Layout(width="800px")
    )
    show_actual = widgets.Checkbox(value=True, description="Actual La2010a", indent=False)
    show_fft = widgets.Checkbox(value=True, description="Corrected 3 — full FFT", indent=False)
    show_error = widgets.Checkbox(value=False, description="Residual", indent=False)

    btn_past = widgets.Button(description="−250 Myr")
    btn_today = widgets.Button(description="Today")
    btn_future = widgets.Button(description="+250 Myr")
    btn_prev = widgets.Button(description="◀ Previous")
    btn_next = widgets.Button(description="Next ▶")
    btn_full = widgets.Button(description="Full ±250 Myr")
    status = widgets.HTML()
    out = widgets.Output()

    def sync():
        width=float(window.value)
        position.max=FULL_MAX-width
        if position.value>position.max:
            position.value=position.max

    def sample_times(start,end):
        width=end-start
        step=1.0 if width<=2 else 2.0 if width<=10 else 10.0 if width<=50 else 20.0 if width<=100 else 100.0
        return np.arange(start*1000.0,end*1000.0+step,step)

    def periodic_fft_values(t_query):
        idx=np.rint((np.asarray(t_query)-t[0])/dt).astype(np.int64)%n
        return i_fft[idx]

    def render(*_):
        try:
            sync()
            start=float(position.value)
            end=min(FULL_MAX,start+float(window.value))
            status.value=f"<b>Window:</b> {start:,.3f} to {end:,.3f} Myr relative to J2000"
            fig=go.Figure()

            observed=raw[(raw.time_myr>=start)&(raw.time_myr<=min(end,0.0))].copy()
            if len(observed)>7000:
                observed=observed.iloc[::max(1,len(observed)//7000)]

            if show_actual.value and not observed.empty:
                fig.add_trace(go.Scattergl(
                    x=observed.time_myr,y=observed.inclination_actual,mode="lines",
                    name="Actual La2010a",line=dict(color="#52D6FF",width=2.0),
                    hovertemplate="<b>Actual La2010a</b><br>Time: %{x:.6f} Myr<br>Inclination: %{y:.9f}°<extra></extra>"
                ))

            tq=sample_times(start,end)
            xq=tq/1000.0
            iq=periodic_fft_values(tq)
            if show_fft.value:
                fig.add_trace(go.Scattergl(
                    x=xq,y=iq,mode="lines",name="Corrected 3 — full FFT",
                    line=dict(color="#35E0A1",width=1.8,dash="dash"),
                    hovertemplate="<b>Corrected 3</b><br>Time: %{x:.6f} Myr<br>Inclination: %{y:.9f}°<extra></extra>"
                ))

            if show_error.value and end<=0.0 and not observed.empty:
                model_obs=periodic_fft_values(observed.t_kyr.to_numpy(float))
                err=model_obs-observed.inclination_actual.to_numpy(float)
                fig.add_trace(go.Scattergl(
                    x=observed.time_myr,y=err,mode="lines",name="FFT − actual",
                    yaxis="y2",line=dict(color="#FF9F43",width=1.4)
                ))

            fig.add_vline(x=0.0,line=dict(color="#35E0A1",width=1.4),annotation_text="J2000")
            fig.update_layout(
                template="plotly_dark",paper_bgcolor="#000000",plot_bgcolor="#000000",
                title=dict(text="La2010a vs Corrected 3 — Full-Record FFT Audit",x=0.5),
                xaxis=dict(title="Millions of years relative to J2000",range=[start,end],gridcolor="#28364A",
                           showspikes=True,spikemode="across",spikesnap="cursor",spikecolor="#1F7A4D",spikethickness=2),
                yaxis=dict(title="Inclination (degrees)",gridcolor="#28364A",
                           showspikes=True,spikemode="across",spikesnap="cursor",spikecolor="#1F7A4D",spikethickness=2),
                yaxis2=dict(title="Residual (degrees)",overlaying="y",side="right",visible=bool(show_error.value),showgrid=False),
                hovermode="closest",hoverdistance=100,spikedistance=-1,height=720,
                margin=dict(l=75,r=75,t=80,b=70),
                legend=dict(bgcolor="rgba(0,0,0,0.78)",bordercolor="#28364A",borderwidth=1)
            )
            with out:
                clear_output(wait=True)
                display(fig)
        except Exception:
            with out:
                clear_output(wait=True)
                display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))

    def set_start(v): position.value=max(FULL_MIN,min(position.max,v))
    btn_past.on_click(lambda _:set_start(FULL_MIN))
    btn_today.on_click(lambda _:set_start(-float(window.value)/2.0))
    btn_future.on_click(lambda _:set_start(FULL_MAX-float(window.value)))
    btn_prev.on_click(lambda _:set_start(position.value-float(window.value)))
    btn_next.on_click(lambda _:set_start(position.value+float(window.value)))
    btn_full.on_click(lambda _:(setattr(window,"value",500.0),set_start(FULL_MIN)))

    for w in (window,position,show_actual,show_fft,show_error):
        w.observe(render,names="value")

    controls=widgets.VBox([
        status,metrics,window,position,
        widgets.HBox([show_actual,show_fft,show_error]),
        widgets.HBox([btn_past,btn_prev,btn_today,btn_next,btn_future,btn_full])
    ])
    display(widgets.VBox([controls,out]))
    render()

except Exception:
    display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))
