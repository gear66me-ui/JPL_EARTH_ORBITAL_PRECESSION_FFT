# EMBOPSA_V0002E
# Colab-safe La2010a spectral back-test and future explorer.
# NO AI-GENERATED IMAGES.

from __future__ import annotations
import io, traceback
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
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
TRAIN_END_KYR = -50000.0
N_HARMONICS = 120
FULL_MIN, FULL_MAX = -250.0, 250.0

display(HTML("<h3 style='color:#35E0A1'>EMBOPSA V0002E starting…</h3>"))
print("Loading La2010a data…", flush=True)

try:
    r = requests.get(DATA_URL, timeout=300)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None,
                      names=["t_kyr","a","l","k","h","q","p"], engine="python")
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)
    raw["time_myr"] = raw["t_kyr"] / 1000.0
    raw["time_years"] = raw["t_kyr"] * 1000.0
    raw["inclination_deg"] = np.degrees(2*np.arcsin(np.clip(np.hypot(raw["p"],raw["q"]),0,1)))
    print(f"Loaded {len(raw):,} rows.", flush=True)

    train = raw[raw["t_kyr"] <= TRAIN_END_KYR]
    t0 = float(train["t_kyr"].iloc[0])
    dt = float(np.median(np.diff(train["t_kyr"])))
    n = len(train)

    def fit(y):
        mean = float(np.mean(y))
        coeff = np.fft.rfft(y-mean)
        freq = np.fft.rfftfreq(len(y), d=dt)
        idx = np.arange(1, len(coeff))
        idx = idx[np.argsort(np.abs(coeff[idx]))[-N_HARMONICS:]]
        idx = idx[np.argsort(freq[idx])]
        return mean, freq[idx], coeff[idx]

    p_mean,p_freq,p_coeff = fit(train["p"].to_numpy(float))
    q_mean,q_freq,q_coeff = fit(train["q"].to_numpy(float))
    print("Spectral model fitted.", flush=True)

    def evaluate(t, mean, freq, coeff):
        t = np.asarray(t, float)
        tau = t-t0
        out = np.empty(len(tau))
        for s in range(0,len(tau),2500):
            e = min(len(tau), s+2500)
            phase = np.exp(2j*np.pi*np.outer(tau[s:e],freq))
            out[s:e] = mean + (2.0/n)*np.real(phase@coeff)
        return out

    def model_frame(t):
        p = evaluate(t,p_mean,p_freq,p_coeff)
        q = evaluate(t,q_mean,q_freq,q_coeff)
        inc = np.degrees(2*np.arcsin(np.clip(np.hypot(p,q),0,1)))
        return p,q,inc

    window = widgets.Dropdown(options=[("100 thousand years",0.1),("250 thousand years",0.25),("500 thousand years",0.5),("1 million years",1.0),("2 million years",2.0),("5 million years",5.0),("10 million years",10.0),("25 million years",25.0),("50 million years",50.0),("100 million years",100.0),("250 million years",250.0),("Full 500 million years",500.0)], value=0.1, description="Window:", layout=widgets.Layout(width="440px"))
    position = widgets.FloatSlider(value=-0.1,min=FULL_MIN,max=FULL_MAX-0.1,step=0.01,description="Start:",continuous_update=False,readout_format=".2f",layout=widgets.Layout(width="800px"))
    show_actual = widgets.Checkbox(value=True,description="Actual La2010a",indent=False)
    show_model = widgets.Checkbox(value=True,description="Spectral model",indent=False)
    show_error = widgets.Checkbox(value=False,description="Model − actual error",indent=False)
    status = widgets.HTML()
    out = widgets.Output()

    buttons = {name: widgets.Button(description=label) for name,label in [("training","Training"),("validation","Validation"),("previous","◀ Previous"),("today","Today"),("next","Next ▶"),("future","Future"),("full","Full ±250 Myr")]}

    def sync():
        position.max = FULL_MAX-float(window.value)
        if position.value>position.max: position.value=position.max

    def times(a,b):
        w=b-a
        step=1.0 if w<=1 else 2.0 if w<=10 else 10.0 if w<=50 else 20.0 if w<=100 else 100.0
        return np.arange(a*1000,b*1000+step,step)

    def render(*_):
        try:
            sync(); a=float(position.value); b=min(FULL_MAX,a+float(window.value))
            status.value=f"<b>Window:</b> {a:,.3f} to {b:,.3f} Myr relative to J2000"
            fig=go.Figure()
            obs=raw[(raw.time_myr>=a)&(raw.time_myr<=min(b,0.0))].copy()
            if len(obs)>6000: obs=obs.iloc[::max(1,len(obs)//6000)]
            if show_actual.value and not obs.empty:
                fig.add_trace(go.Scattergl(x=obs.time_myr,y=obs.inclination_deg,mode="lines",name="Actual La2010a",line=dict(color="#52D6FF",width=2),hovertemplate="<b>Actual</b><br>Time %{x:.6f} Myr<br>Inclination %{y:.6f}°<extra></extra>"))
            if show_model.value or show_error.value:
                t=times(a,b); p,q,inc=model_frame(t); x=t/1000.0
                if show_model.value:
                    fig.add_trace(go.Scattergl(x=x,y=inc,mode="lines",name="Spectral model",line=dict(color="#FF6EC7",width=1.8,dash="dash"),hovertemplate="<b>Model</b><br>Time %{x:.6f} Myr<br>Inclination %{y:.6f}°<extra></extra>"))
                if show_error.value and b<=0 and not obs.empty:
                    interp=np.interp(obs.t_kyr.to_numpy(float),t,inc)
                    fig.add_trace(go.Scattergl(x=obs.time_myr,y=interp-obs.inclination_deg.to_numpy(float),mode="lines",name="Model − actual error",yaxis="y2",line=dict(color="#FF9F43",width=1.5)))
            fig.add_vline(x=-50,line=dict(color="#FF9F43",dash="dot"),annotation_text="Training ends")
            fig.add_vline(x=0,line=dict(color="#35E0A1",width=2),annotation_text="J2000")
            fig.update_layout(template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#000",title=dict(text="La2010a vs Spectral Model — Colab-Safe Explorer",x=.5),xaxis=dict(title="Millions of years relative to J2000",range=[a,b],gridcolor="#28364A",showspikes=True,spikemode="across",spikecolor="#1F7A4D"),yaxis=dict(title="Inclination (degrees)",gridcolor="#28364A",showspikes=True,spikemode="across",spikecolor="#1F7A4D"),yaxis2=dict(title="Error (degrees)",overlaying="y",side="right",visible=bool(show_error.value),showgrid=False),hovermode="closest",height=720)
            with out:
                clear_output(wait=True)
                display(fig)
        except Exception:
            with out:
                clear_output(wait=True)
                display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))

    def center(c):
        position.value=max(FULL_MIN,min(position.max,c-float(window.value)/2))
    buttons["training"].on_click(lambda _:center(-150))
    buttons["validation"].on_click(lambda _:center(-25))
    buttons["today"].on_click(lambda _:center(0))
    buttons["future"].on_click(lambda _:center(25))
    buttons["previous"].on_click(lambda _:setattr(position,"value",max(FULL_MIN,position.value-window.value)))
    buttons["next"].on_click(lambda _:setattr(position,"value",min(position.max,position.value+window.value)))
    buttons["full"].on_click(lambda _:(setattr(window,"value",500.0),setattr(position,"value",FULL_MIN)))
    for w in (window,position,show_actual,show_model,show_error): w.observe(render,names="value")

    controls=widgets.VBox([status,widgets.HTML("<b style='color:#ff9f43'>Future curve is illustrative.</b>"),window,position,widgets.HBox([show_actual,show_model,show_error]),widgets.HBox(list(buttons.values()))])
    display(widgets.VBox([controls,out]))
    render()
    print("Widget displayed.",flush=True)
except Exception:
    display(HTML("<pre style='color:#ff6b6b'>"+traceback.format_exc()+"</pre>"))
