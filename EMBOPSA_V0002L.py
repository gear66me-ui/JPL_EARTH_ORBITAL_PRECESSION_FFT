# EMBOPSA_V0002L
# FFT-3 exact reconstruction with a separate, correctly scaled residual panel.
# NO AI-GENERATED IMAGES.

import io, traceback
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
DATA_URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"

startup = widgets.HTML("<b style='color:#35E0A1'>Loading La2010a and rebuilding FFT-3…</b>")
display(startup)

try:
    r = requests.get(DATA_URL, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None,
                     names=["t_kyr","a","l","k","h","q","p"], engine="python")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("t_kyr").reset_index(drop=True)

    t_kyr = df.t_kyr.to_numpy(float)
    t_myr = t_kyr / 1000.0
    p = df.p.to_numpy(float)
    q = df.q.to_numpy(float)
    n = len(df)

    # FFT-3: all bins retained, so inverse FFT reconstructs the sampled record.
    p3 = np.fft.irfft(np.fft.rfft(p), n=n)
    q3 = np.fft.irfft(np.fft.rfft(q), n=n)
    actual = np.degrees(2*np.arcsin(np.clip(np.hypot(p, q), 0, 1)))
    fft3 = np.degrees(2*np.arcsin(np.clip(np.hypot(p3, q3), 0, 1)))
    error = fft3 - actual

    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    maxerr = float(np.max(np.abs(error)))

    startup.value = "<b style='color:#35E0A1'>FFT-3 residual audit ready.</b>"
    metrics = widgets.HTML(value=(
        "<b>FFT-3 true error function</b><br>"
        "ε(t) = i<sub>FFT-3</sub>(t) − i<sub>La2010a</sub>(t)<br>"
        f"Samples = {n:,}<br>"
        f"RMSE = {rmse:.15e}° &nbsp; | &nbsp; MAE = {mae:.15e}°<br>"
        f"Bias = {bias:+.15e}° &nbsp; | &nbsp; Max |ε| = {maxerr:.15e}°<br>"
        "<b style='color:#ff9f43'>Residuals are shown in a separate panel; they are no longer autoscaled over the inclination curve.</b>"
    ))

    window = widgets.Dropdown(options=[("0.1 Myr",0.1),("0.25 Myr",0.25),("0.5 Myr",0.5),
        ("1 Myr",1.0),("2 Myr",2.0),("5 Myr",5.0),("10 Myr",10.0),
        ("25 Myr",25.0),("50 Myr",50.0),("100 Myr",100.0),("250 Myr",250.0)],
        value=1.0, description="Window:", layout=widgets.Layout(width="330px"))
    start = widgets.FloatSlider(value=-1.0, min=-250.0, max=-1.0, step=0.01,
        description="Start:", continuous_update=False, readout_format=".2f",
        layout=widgets.Layout(width="800px"))
    show_actual = widgets.Checkbox(True, description="Actual La2010a", indent=False)
    show_fft3 = widgets.Checkbox(True, description="FFT-3", indent=False)
    show_error = widgets.Checkbox(True, description="FFT-3 error panel", indent=False)
    units = widgets.Dropdown(options=[("Degrees",1.0,"°"),("Nanodegrees",1e9,"n°"),
        ("Picodegrees",1e12,"p°"),("Femtodegrees",1e15,"f°")],
        value=("Femtodegrees",1e15,"f°"), description="Error units:",
        layout=widgets.Layout(width="360px"))
    prevb, nextb, todayb, fullb = [widgets.Button(description=x) for x in
        ("◀ Previous","Next ▶","J2000","Full −250 to 0 Myr")]
    status = widgets.HTML()
    out = widgets.Output()

    def sync():
        start.max = -float(window.value)
        if start.value > start.max:
            start.value = start.max

    def render(*_):
        try:
            sync()
            left = float(start.value)
            right = min(0.0, left + float(window.value))
            idx = np.flatnonzero((t_myr >= left) & (t_myr <= right))
            if len(idx) > 15000:
                idx = idx[::max(1, len(idx)//15000)]
            x, ya, yf, er = t_myr[idx], actual[idx], fft3[idx], error[idx]
            label, scale, unit = units.value
            ers = er * float(scale)
            status.value = f"<b>Window:</b> {left:,.3f} to {right:,.3f} Myr &nbsp; | &nbsp; residual units: {label}"

            if show_error.value:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.72,0.28], subplot_titles=("La2010a and FFT-3","FFT-3 − La2010a residual"))
            else:
                fig = make_subplots(rows=1, cols=1)

            if show_actual.value:
                fig.add_trace(go.Scattergl(x=x,y=ya,mode="lines",name="Actual La2010a",
                    line=dict(color="#52D6FF",width=2.1)), row=1,col=1)
            if show_fft3.value:
                fig.add_trace(go.Scattergl(x=x,y=yf,mode="lines",name="FFT-3",
                    line=dict(color="#35E0A1",width=1.7,dash="dash")), row=1,col=1)
            if show_error.value:
                fig.add_trace(go.Scattergl(x=x,y=ers,mode="lines",name=f"FFT-3 error ({unit})",
                    line=dict(color="#FF9F43",width=1.3),
                    hovertemplate=f"Time: %{{x:.6f}} Myr<br>Error: %{{y:+.9f}} {unit}<extra></extra>"), row=2,col=1)
                fig.add_hline(y=0,line_width=1,line_dash="dash",line_color="#E6E6E6",row=2,col=1)
                lim = max(float(np.max(np.abs(ers)))*1.10, np.finfo(float).eps*float(scale))
                fig.update_yaxes(title_text=f"Residual ({unit})", range=[-lim,lim],
                    exponentformat="e", showexponent="all", row=2,col=1)

            fig.update_yaxes(title_text="Inclination (degrees)", row=1,col=1)
            fig.update_xaxes(title_text="Millions of years relative to J2000",
                row=2 if show_error.value else 1,col=1)
            fig.update_layout(template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#000",
                title=dict(text="FFT-3 Exact Reconstruction and True Error Function",x=0.5),
                height=820 if show_error.value else 680,margin=dict(l=85,r=45,t=95,b=75),
                hovermode="closest",legend=dict(bgcolor="rgba(0,0,0,.78)",bordercolor="#28364A",borderwidth=1))
            fig.update_xaxes(gridcolor="#28364A")
            fig.update_yaxes(gridcolor="#28364A")
            with out:
                clear_output(wait=True)
                display(fig)
        except Exception:
            with out:
                clear_output(wait=True)
                display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))

    prevb.on_click(lambda _: setattr(start,"value",max(-250.0,start.value-window.value)))
    nextb.on_click(lambda _: setattr(start,"value",min(start.max,start.value+window.value)))
    todayb.on_click(lambda _: setattr(start,"value",-float(window.value)))
    fullb.on_click(lambda _: (setattr(window,"value",250.0),setattr(start,"value",-250.0)))
    for w in (window,start,show_actual,show_fft3,show_error,units):
        w.observe(render,names="value")

    controls = widgets.VBox([status,metrics,window,start,
        widgets.HBox([show_actual,show_fft3,show_error]),units,
        widgets.HBox([prevb,nextb,todayb,fullb])])
    display(widgets.VBox([controls,out]))
    render()
except Exception:
    display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))
