# V0001K
# NO AI-GENERATED IMAGES. PYTHON/PANDAS/MATPLOTLIB/IPYWIDGETS ONLY.

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

C = {
    "bg":"#000000","grid":"#29364D","text":"#F4F7FB","muted":"#AAB4C3",
    "cyan":"#52D6FF","violet":"#B388FF","gold":"#FFD166",
    "green":"#35E0A1","red":"#FF5C7A","magenta":"#FF6EC7"
}

STATE = {"df": None}

url_box = widgets.Text(
    value="",
    placeholder="Paste direct La2010 CSV/TXT data URL",
    description="URL",
    layout=widgets.Layout(width="760px"),
)
upload = widgets.FileUpload(
    accept=".csv,.txt,.dat",
    multiple=False,
    description="Upload data",
)
load_button = widgets.Button(description="Load", button_style="success")
status = widgets.HTML()

time_dd = widgets.Dropdown(description="Time")
signal_dd = widgets.Dropdown(description="Signal")
incl_dd = widgets.Dropdown(description="Inclination", options=["None"])
node_dd = widgets.Dropdown(description="Node", options=["None"])
units_dd = widgets.Dropdown(
    description="Time units",
    options=["years","kyr","Myr"],
    value="years",
)
mode_dd = widgets.Dropdown(
    description="View",
    options=["Time series","Spectrum","Both","Orbital pole p/q/|z|"],
    value="Both",
)
window = widgets.FloatRangeSlider(
    value=[-500,500], min=-1000, max=1000, step=10,
    description="Window (kyr)", continuous_update=False,
    layout=widgets.Layout(width="690px"),
)
period_band = widgets.FloatRangeSlider(
    value=[100,500], min=10, max=1000, step=10,
    description="Period (kyr)", continuous_update=False,
    layout=widgets.Layout(width="690px"),
)
detrend_dd = widgets.Dropdown(
    description="Detrend",
    options=[("Mean",0),("Linear",1),("Quadratic",2),("Cubic",3)],
    value=1,
)
render_button = widgets.Button(description="Render", button_style="info")
out = widgets.Output()

def parse_bytes(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    attempts = [
        dict(sep=None, engine="python", comment="#"),
        dict(delim_whitespace=True, comment="#"),
        dict(sep=",", comment="#"),
    ]
    for kw in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **kw)
            if df.shape[1] >= 2 and len(df) >= 8:
                return df
        except Exception:
            pass
    raise ValueError("Could not parse the uploaded file.")

def numeric_cols(df):
    converted = df.copy()
    for c in converted.columns:
        converted[c] = pd.to_numeric(converted[c], errors="ignore")
    nums = converted.select_dtypes(include=[np.number]).columns.tolist()
    return converted, nums

def load_data(_=None):
    try:
        raw = None
        if upload.value:
            item = next(iter(upload.value.values())) if isinstance(upload.value, dict) else upload.value[0]
            raw = bytes(item["content"])
        elif url_box.value.strip():
            r = requests.get(url_box.value.strip(), timeout=180)
            r.raise_for_status()
            raw = r.content
        else:
            raise ValueError("Upload a La2010 file or paste a direct data URL.")

        df = parse_bytes(raw)
        df, nums = numeric_cols(df)
        if len(nums) < 2:
            raise ValueError("At least two numeric columns are required.")

        STATE["df"] = df
        time_dd.options = nums
        signal_dd.options = nums
        incl_dd.options = ["None"] + nums
        node_dd.options = ["None"] + nums

        time_guess = next((c for c in nums if str(c).lower() in
                          {"t","time","year","years","kyr","myr","age"}), nums[0])
        time_dd.value = time_guess
        signal_dd.value = next((c for c in nums if c != time_guess), nums[1])

        status.value = (
            f"<b>Loaded:</b> {len(df):,} rows × {df.shape[1]} columns"
        )
    except Exception as exc:
        status.value = f"<b style='color:#FF5C7A'>REJECTED:</b> {exc}"

def years_scale():
    return {"years":1.0, "kyr":1000.0, "Myr":1_000_000.0}[units_dd.value]

def clean_xy(df, xcol, ycol):
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float) * years_scale()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    return x[order], y[order]

def detrend(x, y):
    degree = int(detrend_dd.value)
    xc = x - np.mean(x)
    scale = max(np.ptp(xc), 1.0)
    u = xc / scale
    fit = np.polyval(np.polyfit(u, y, degree), u)
    return y - fit

def regularize(x, y):
    dx = np.diff(x)
    step = np.median(dx[dx > 0])
    grid = np.arange(x.min(), x.max() + step, step)
    return grid, np.interp(grid, x, y), step

def spectrum(x, y):
    grid, values, dt = regularize(x, detrend(x, y))
    win = np.hanning(len(values))
    spec = np.fft.rfft((values - np.mean(values)) * win)
    freq = np.fft.rfftfreq(len(values), d=dt)
    amp = 2.0 * np.abs(spec) / max(np.sum(win), 1.0)
    good = freq > 0
    return 1.0 / freq[good], amp[good]

def pole_from_i_node(i, node):
    max_i = np.nanmax(np.abs(i))
    max_n = np.nanmax(np.abs(node))
    if max_i > 2*np.pi or max_n > 2*np.pi:
        i = np.deg2rad(i)
        node = np.deg2rad(node)
    p = np.sin(i/2.0) * np.sin(node)
    q = np.sin(i/2.0) * np.cos(node)
    return p, q, np.hypot(p, q)

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["text"], fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=.6, alpha=.75)
    for s in ax.spines.values():
        s.set_color(C["grid"])

def show_time(x, y, label, color):
    lo, hi = np.array(window.value) * 1000.0
    m = (x >= lo) & (x <= hi)
    fig, ax = plt.subplots(figsize=(12.5,5.2))
    ax.plot(x[m]/1000.0, y[m], color=color, linewidth=1.05)
    ax.axvline(0, color=C["red"], linestyle="--", linewidth=.9)
    style(ax, label, "Time (kyr)", label)
    fig.tight_layout()
    display(fig)
    plt.close(fig)

def show_spectrum(x, y, label, color):
    periods, amps = spectrum(x, y)
    lo, hi = np.array(period_band.value) * 1000.0
    m = (periods >= lo) & (periods <= hi)
    order = np.argsort(periods[m])
    pp, aa = periods[m][order], amps[m][order]

    fig, ax = plt.subplots(figsize=(12.5,5.2))
    ax.plot(pp/1000.0, aa, color=color, linewidth=1.1)
    ax.axvline(230, color=C["violet"], linestyle="--", label="230 kyr")
    ax.axvline(270, color=C["gold"], linestyle="--", label="270 kyr")
    style(ax, f"{label} spectrum", "Period (kyr)", "Amplitude")
    leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    if len(aa):
        idx = np.argsort(aa)[::-1][:10]
        peaks = pd.DataFrame({
            "rank": np.arange(1, len(idx)+1),
            "period_kyr": pp[idx]/1000.0,
            "amplitude": aa[idx],
        }).sort_values("rank")
        display(peaks.style.format({"period_kyr":"{:,.3f}", "amplitude":"{:.9e}"}))

def render(_=None):
    with out:
        clear_output(wait=True)
        df = STATE["df"]
        if df is None:
            print("Load a La2010 CSV/TXT file first.")
            return

        plt.close("all")
        plt.ioff()
        plt.rcParams.update({
            "figure.facecolor":C["bg"],"axes.facecolor":C["bg"],
            "savefig.facecolor":C["bg"],"text.color":C["text"],
            "axes.labelcolor":C["text"],"xtick.color":C["text"],
            "ytick.color":C["text"],"font.family":"DejaVu Sans"
        })

        if mode_dd.value == "Orbital pole p/q/|z|":
            if incl_dd.value == "None" or node_dd.value == "None":
                print("Select inclination and ascending-node columns.")
                return
            x, inc = clean_xy(df, time_dd.value, incl_dd.value)
            x2, node = clean_xy(df, time_dd.value, node_dd.value)
            if len(x) != len(x2) or not np.allclose(x, x2):
                node = np.interp(x, x2, node)
            p, q, z = pole_from_i_node(inc, node)
            for y, label, color in [
                (p,"p(t)",C["cyan"]), (q,"q(t)",C["violet"]), (z,"|z(t)|",C["gold"])
            ]:
                show_time(x, y, label, color)
                show_spectrum(x, y, label, color)
            return

        x, y = clean_xy(df, time_dd.value, signal_dd.value)
        label = str(signal_dd.value)

        if mode_dd.value in {"Time series","Both"}:
            show_time(x, y, label, C["cyan"])
        if mode_dd.value in {"Spectrum","Both"}:
            show_spectrum(x, y, label, C["gold"])

load_button.on_click(load_data)
render_button.on_click(render)

display(widgets.VBox([
    widgets.HBox([url_box, load_button]),
    upload,
    status,
    widgets.HBox([time_dd, signal_dd, units_dd]),
    widgets.HBox([incl_dd, node_dd, mode_dd]),
    window,
    period_band,
    widgets.HBox([detrend_dd, render_button]),
    out,
]))
# V0001K
