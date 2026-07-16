# EMBOPSA_V0001V
# Touch-enabled Earth orbital inclination explorer for Google Colab.
# NO AI-GENERATED IMAGES.
# Uses deterministic La2010a data and a browser-native Plotly interface.

from __future__ import annotations

import io
import json
import uuid
import numpy as np
import pandas as pd
import requests
from IPython.display import HTML, display

VERSION = "V0001V"

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

age = df["age_myr"].to_numpy(dtype=float)
inc = df["inclination_deg"].to_numpy(dtype=float)

today_i = float(df.loc[df["t_kyr"].abs().idxmin(), "inclination_deg"])
max_row = df.loc[df["inclination_deg"].idxmax()]
global_max_i = float(max_row["inclination_deg"])
global_max_age = float(max_row["age_myr"])
max_age = float(age.max())

widget_id = f"embopsa_{uuid.uuid4().hex}"

age_json = json.dumps(age.tolist(), separators=(",", ":"))
inc_json = json.dumps(inc.tolist(), separators=(",", ":"))

html = f"""
<div id="{widget_id}_root" style="
    background:#000;
    color:#f4f7fb;
    padding:14px;
    border:1px solid #28364a;
    border-radius:10px;
    font-family:Arial,sans-serif;
">
  <div style="font-size:18px;font-weight:700;margin-bottom:8px;">
    Earth Orbital Inclination — Touch Two Points to Measure a Cycle
  </div>

  <div id="{widget_id}_status" style="margin-bottom:8px;color:#d9e1ec;">
    Tap one point for A, then tap another point for B.
  </div>

  <div style="margin:8px 0;">
    <label style="display:inline-block;width:130px;">Window width</label>
    <input id="{widget_id}_width" type="range"
           min="0.5" max="{max_age:.3f}" step="0.5" value="10"
           style="width:62%;vertical-align:middle;">
    <span id="{widget_id}_width_value">10.0 Myr</span>
  </div>

  <div style="margin:8px 0;">
    <label style="display:inline-block;width:130px;">Window position</label>
    <input id="{widget_id}_position" type="range"
           min="0" max="{max_age - 10:.3f}" step="0.1" value="0"
           style="width:62%;vertical-align:middle;">
    <span id="{widget_id}_position_value">0.0 Myr</span>
  </div>

  <div style="margin:10px 0;">
    <button id="{widget_id}_today">Today</button>
    <button id="{widget_id}_previous">◀ Previous</button>
    <button id="{widget_id}_next">Next ▶</button>
    <button id="{widget_id}_maximum">Go to maximum</button>
    <button id="{widget_id}_clear">Clear A–B</button>
  </div>

  <div id="{widget_id}_measurement" style="
      margin:10px 0;
      padding:8px;
      background:#090d12;
      border:1px solid #28364a;
      border-radius:6px;
      font-weight:600;
  ">
    Tap one point for A, then another point for B.
  </div>

  <div id="{widget_id}_plot" style="width:100%;height:680px;"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function() {{
  const age = {age_json};
  const inc = {inc_json};

  const todayI = {today_i:.15f};
  const globalMaxI = {global_max_i:.15f};
  const globalMaxAge = {global_max_age:.15f};
  const maxAge = {max_age:.15f};

  const root = "{widget_id}";
  const plot = document.getElementById(root + "_plot");
  const widthSlider = document.getElementById(root + "_width");
  const positionSlider = document.getElementById(root + "_position");
  const widthValue = document.getElementById(root + "_width_value");
  const positionValue = document.getElementById(root + "_position_value");
  const measurement = document.getElementById(root + "_measurement");
  const status = document.getElementById(root + "_status");

  let selected = [];
  let shownX = [];
  let shownY = [];

  function nearestIndex(value) {{
    let lo = 0;
    let hi = age.length - 1;
    while (lo < hi) {{
      const mid = Math.floor((lo + hi) / 2);
      if (age[mid] < value) lo = mid + 1;
      else hi = mid;
    }}
    if (lo > 0 && Math.abs(age[lo - 1] - value) < Math.abs(age[lo] - value)) {{
      return lo - 1;
    }}
    return lo;
  }}

  function makeIndices(start, end) {{
    const i0 = nearestIndex(start);
    const i1 = nearestIndex(end);
    const count = Math.max(1, i1 - i0 + 1);
    const stride = Math.max(1, Math.ceil(count / 6000));
    const indices = [];
    for (let i = i0; i <= i1; i += stride) indices.push(i);
    if (indices[indices.length - 1] !== i1) indices.push(i1);
    return indices;
  }}

  function clearMeasurement() {{
    selected = [];
    measurement.innerHTML = "Tap one point for A, then another point for B.";
    Plotly.relayout(plot, {{shapes: [], annotations: []}});
  }}

  function render() {{
    const width = Math.min(parseFloat(widthSlider.value), maxAge);
    const maxPosition = Math.max(0, maxAge - width);

    positionSlider.max = maxPosition.toFixed(3);
    if (parseFloat(positionSlider.value) > maxPosition) {{
      positionSlider.value = maxPosition;
    }}

    const start = parseFloat(positionSlider.value);
    const end = Math.min(maxAge, start + width);

    widthValue.textContent = width.toFixed(1) + " Myr";
    positionValue.textContent = start.toFixed(1) + " Myr";

    const indices = makeIndices(start, end);
    shownX = indices.map(i => age[i]);
    shownY = indices.map(i => inc[i]);

    let localMaxI = -Infinity;
    let localMaxAge = start;
    let localMinI = Infinity;

    for (let i = 0; i < shownY.length; i++) {{
      if (shownY[i] > localMaxI) {{
        localMaxI = shownY[i];
        localMaxAge = shownX[i];
      }}
      if (shownY[i] < localMinI) localMinI = shownY[i];
    }}

    const traces = [
      {{
        x: shownX,
        y: shownY,
        mode: "lines+markers",
        name: "Earth orbital inclination",
        line: {{color:"#52d6ff", width:1.35}},
        marker: {{color:"#52d6ff", size:5, opacity:0.70}},
        hovertemplate:
          "Age: %{{x:.6f}} Myr before J2000" +
          "<br>Inclination: %{{y:.6f}}°<extra></extra>"
      }},
      {{
        x: [start, end],
        y: [todayI, todayI],
        mode: "lines",
        name: "J2000 = " + todayI.toFixed(6) + "°",
        line: {{color:"#ffd166", width:1.2, dash:"dash"}},
        hoverinfo: "skip"
      }},
      {{
        x: [start, end],
        y: [3.0, 3.0],
        mode: "lines",
        name: "3° reference",
        line: {{color:"#ff5c7a", width:1.1, dash:"dot"}},
        hoverinfo: "skip"
      }},
      {{
        x: [localMaxAge],
        y: [localMaxI],
        mode: "markers",
        name: "Window maximum",
        marker: {{color:"#35e0a1", size:10}},
        hovertemplate:
          "Window maximum" +
          "<br>Age: %{{x:.6f}} Myr" +
          "<br>Inclination: %{{y:.6f}}°<extra></extra>"
      }}
    ];

    const layout = {{
      paper_bgcolor:"#000000",
      plot_bgcolor:"#000000",
      font:{{color:"#f4f7fb"}},
      title:{{
        text:"Earth Orbital-Plane Inclination Window Explorer",
        x:0.5,
        xanchor:"center"
      }},
      xaxis:{{
        title:"Millions of years before J2000",
        range:[start, end],
        gridcolor:"#28364a",
        zeroline:false
      }},
      yaxis:{{
        title:"Inclination (degrees)",
        range:[Math.max(0, localMinI - 0.2), Math.max(3.15, localMaxI + 0.2)],
        gridcolor:"#28364a",
        zeroline:false
      }},
      margin:{{l:70,r:30,t:75,b:70}},
      hovermode:"closest",
      clickmode:"event+select",
      legend:{{
        bgcolor:"rgba(0,0,0,0.75)",
        bordercolor:"#28364a",
        borderwidth:1
      }},
      shapes:[],
      annotations:[]
    }};

    Plotly.react(plot, traces, layout, {{
      responsive:true,
      displaylogo:false,
      scrollZoom:true
    }});

    status.innerHTML =
      "<b>Window:</b> " + start.toFixed(3) + "–" + end.toFixed(3) +
      " Myr before J2000 &nbsp; | &nbsp; " +
      "<b>Tap two cyan points to measure.</b>";

    clearMeasurement();
  }}

  plot.on("plotly_click", function(data) {{
    if (!data || !data.points || data.points.length === 0) return;

    const point = data.points[0];
    if (point.curveNumber !== 0) return;

    if (selected.length >= 2) selected = [];

    const x = Number(point.x);
    const y = Number(point.y);
    selected.push({{x:x, y:y}});

    const shapes = [];
    const annotations = [];

    if (selected.length >= 1) {{
      const a = selected[0];
      shapes.push({{
        type:"line",
        x0:a.x, x1:a.x,
        y0:0, y1:1,
        xref:"x", yref:"paper",
        line:{{color:"#ffd166", width:2, dash:"dash"}}
      }});
      annotations.push({{
        x:a.x, y:1,
        xref:"x", yref:"paper",
        text:"<b>A</b><br>" + a.x.toFixed(6) + " Myr",
        showarrow:false,
        textangle:-90,
        xanchor:"right",
        yanchor:"top",
        font:{{color:"#ffd166", size:12}},
        bgcolor:"rgba(0,0,0,0.78)",
        bordercolor:"#ffd166"
      }});
      measurement.innerHTML =
        "<b>A:</b> " + a.x.toFixed(6) + " Myr before J2000 " +
        "(" + a.y.toFixed(6) + "°). Tap B.";
    }}

    if (selected.length === 2) {{
      const a = selected[0];
      const b = selected[1];
      const deltaMyr = Math.abs(b.x - a.x);
      const deltaKyr = deltaMyr * 1000.0;
      const deltaYears = deltaMyr * 1000000.0;

      shapes.push({{
        type:"line",
        x0:b.x, x1:b.x,
        y0:0, y1:1,
        xref:"x", yref:"paper",
        line:{{color:"#b388ff", width:2, dash:"dash"}}
      }});
      annotations.push({{
        x:b.x, y:1,
        xref:"x", yref:"paper",
        text:"<b>B</b><br>" + b.x.toFixed(6) + " Myr",
        showarrow:false,
        textangle:-90,
        xanchor:"right",
        yanchor:"top",
        font:{{color:"#b388ff", size:12}},
        bgcolor:"rgba(0,0,0,0.78)",
        bordercolor:"#b388ff"
      }});

      measurement.innerHTML =
        "<b>A:</b> " + a.x.toFixed(6) + " Myr &nbsp; | &nbsp; " +
        "<b>B:</b> " + b.x.toFixed(6) + " Myr &nbsp; | &nbsp; " +
        "<b>Separation:</b> " +
        Math.round(deltaYears).toLocaleString() + " years = " +
        deltaKyr.toFixed(3) + " kyr = " +
        deltaMyr.toFixed(6) + " Myr";
    }}

    Plotly.relayout(plot, {{
      shapes:shapes,
      annotations:annotations
    }});
  }});

  widthSlider.addEventListener("change", render);
  positionSlider.addEventListener("change", render);

  document.getElementById(root + "_today").onclick = function() {{
    positionSlider.value = 0;
    render();
  }};

  document.getElementById(root + "_previous").onclick = function() {{
    const width = parseFloat(widthSlider.value);
    positionSlider.value = Math.max(
      0,
      parseFloat(positionSlider.value) - width
    );
    render();
  }};

  document.getElementById(root + "_next").onclick = function() {{
    const width = parseFloat(widthSlider.value);
    const maxPosition = Math.max(0, maxAge - width);
    positionSlider.value = Math.min(
      maxPosition,
      parseFloat(positionSlider.value) + width
    );
    render();
  }};

  document.getElementById(root + "_maximum").onclick = function() {{
    const width = parseFloat(widthSlider.value);
    const maxPosition = Math.max(0, maxAge - width);
    positionSlider.value = Math.min(
      maxPosition,
      Math.max(0, globalMaxAge - width / 2)
    );
    render();
  }};

  document.getElementById(root + "_clear").onclick = clearMeasurement;

  render();
}})();
</script>
"""

display(HTML(html))
