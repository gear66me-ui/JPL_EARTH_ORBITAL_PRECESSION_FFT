# EMBOPSA_V0002J
# La2010a vs full-record FFT correlation audit.
# NO AI-GENERATED IMAGES.

import io, traceback
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress
from IPython.display import display, HTML, clear_output

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

pio.renderers.default = "colab"
URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
status = widgets.HTML("<b style='color:#35E0A1'>Loading La2010a and calculating correlation audit…</b>")
display(status)

try:
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None,
                     names=["t_kyr","a","l","k","h","q","p"], engine="python")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("t_kyr").reset_index(drop=True)

    t = df.t_kyr.to_numpy(float)
    p = df.p.to_numpy(float)
    q = df.q.to_numpy(float)
    n = len(df)

    p_rec = np.fft.irfft(np.fft.rfft(p), n=n)
    q_rec = np.fft.irfft(np.fft.rfft(q), n=n)
    actual = np.degrees(2*np.arcsin(np.clip(np.hypot(p,q),0,1)))
    model = np.degrees(2*np.arcsin(np.clip(np.hypot(p_rec,q_rec),0,1)))
    residual = model - actual

    pr = pearsonr(actual, model)
    sr = spearmanr(actual, model)
    kt = kendalltau(actual, model)
    reg = linregress(model, actual)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    maxerr = float(np.max(np.abs(residual)))

    df["time_myr"] = t/1000.0
    df["actual"] = actual
    df["model"] = model
    df["residual"] = residual

    block_rows = []
    for left in np.arange(-250.0,0.0,25.0):
        right = left + 25.0
        b = df[(df.time_myr>=left)&(df.time_myr<right)]
        e = b.residual.to_numpy(float)
        block_rows.append({"interval":f"{left:.0f} to {right:.0f}",
                           "r":float(pearsonr(b.actual,b.model).statistic),
                           "rmse":float(np.sqrt(np.mean(e**2))),
                           "maxerr":float(np.max(np.abs(e)))})
    blocks = pd.DataFrame(block_rows)

    status.value = "<b style='color:#35E0A1'>Correlation audit complete.</b>"
    metrics = widgets.HTML(value=(
        "<b>La2010a vs Corrected 3 — complete FFT reconstruction</b><br>"
        f"Samples = {n:,}<br>"
        f"Pearson r = {pr.statistic:.15f}; p = {pr.pvalue:.3e}; R² = {pr.statistic**2:.15f}<br>"
        f"Spearman ρ = {sr.statistic:.15f}; Kendall τ = {kt.statistic:.15f}<br>"
        f"Regression: i<sub>actual</sub> = {reg.intercept:+.15e}° + ({reg.slope:.15f})i<sub>FFT</sub><br>"
        f"RMSE = {rmse:.15e}°; MAE = {mae:.15e}°; bias = {bias:+.15e}°; max |error| = {maxerr:.15e}°<br>"
        "<b style='color:#ff9f43'>All FFT bins are retained. This confirms numerical reconstruction, not independent future prediction.</b>"))

    view = widgets.ToggleButtons(options=[("Scatter","scatter"),("Residual","residual"),("25-Myr blocks","blocks"),("Overlay","overlay")], value="scatter", description="View:")
    out = widgets.Output()

    def render(*_):
        try:
            fig = go.Figure()
            stride = max(1,n//15000)
            s = df.iloc[::stride]
            if view.value == "scatter":
                fig.add_trace(go.Scattergl(x=s.model,y=s.actual,mode="markers",name="Samples",marker=dict(size=4,opacity=.35)))
                lo = float(min(s.model.min(),s.actual.min())); hi = float(max(s.model.max(),s.actual.max()))
                fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Identity",line=dict(dash="dash")))
                fig.update_layout(title="Actual La2010a vs Corrected 3 FFT",xaxis_title="FFT inclination (degrees)",yaxis_title="Actual inclination (degrees)")
            elif view.value == "residual":
                fig.add_trace(go.Scattergl(x=s.time_myr,y=s.residual,mode="lines",name="FFT − actual"))
                fig.add_hline(y=0,line_dash="dash")
                fig.update_layout(title="Numerical residual across La2010a",xaxis_title="Millions of years relative to J2000",yaxis_title="Residual (degrees)")
            elif view.value == "blocks":
                fig.add_trace(go.Bar(x=blocks.interval,y=blocks.rmse,name="RMSE",customdata=np.column_stack([blocks.r,blocks.maxerr]),hovertemplate="%{x} Myr<br>RMSE: %{y:.15e}°<br>r: %{customdata[0]:.15f}<br>Max: %{customdata[1]:.15e}°<extra></extra>"))
                fig.update_layout(title="Correlation stability by 25-Myr block",xaxis_title="Interval",yaxis_title="RMSE (degrees)")
            else:
                fig.add_trace(go.Scattergl(x=s.time_myr,y=s.actual,mode="lines",name="Actual La2010a",line=dict(width=2)))
                fig.add_trace(go.Scattergl(x=s.time_myr,y=s.model,mode="lines",name="Corrected 3 FFT",line=dict(width=1.5,dash="dash")))
                fig.update_layout(title="La2010a and Corrected 3 FFT overlay",xaxis_title="Millions of years relative to J2000",yaxis_title="Inclination (degrees)")
            fig.update_layout(template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#000",height=720,margin=dict(l=80,r=40,t=80,b=90),hovermode="closest")
            fig.update_xaxes(gridcolor="#28364A",showspikes=True,spikemode="across",spikecolor="#1F7A4D")
            fig.update_yaxes(gridcolor="#28364A",showspikes=True,spikemode="across",spikecolor="#1F7A4D")
            with out:
                clear_output(wait=True); display(fig)
        except Exception:
            with out:
                clear_output(wait=True); display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))

    view.observe(render,names="value")
    display(widgets.VBox([metrics,view,out]))
    render()
except Exception:
    display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))
