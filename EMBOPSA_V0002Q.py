# EMBOPSA_V0002Q
# Runs the verified H.264 renderer and embeds the finished MP4 directly in Colab.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import requests
from IPython.display import HTML, display

VERSION = "V0002Q"
RENDERER_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "b5586155bc3a77ed035e2f23b8a02417deb4f8c6/EMBOPSA_V0002P.py"
)
OUT_MP4 = Path(
    "/content/embopsa_video/"
    "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_H264.mp4"
)

print(f"[{VERSION}] Running deterministic La2010a renderer…")
renderer = requests.get(RENDERER_URL, timeout=300)
renderer.raise_for_status()
exec(compile(renderer.text, "EMBOPSA_V0002P.py", "exec"), globals(), globals())

if not OUT_MP4.exists() or OUT_MP4.stat().st_size < 100_000:
    raise RuntimeError(f"Rendered MP4 is missing or too small: {OUT_MP4}")

probe_cmd = [
    "ffprobe", "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration,size",
    "-of", "json",
    str(OUT_MP4),
]
probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
metadata = json.loads(probe.stdout)
stream = metadata["streams"][0]
fmt = metadata["format"]

if stream.get("codec_name") != "h264":
    raise RuntimeError(f"Unexpected codec: {stream.get('codec_name')}")
if stream.get("pix_fmt") != "yuv420p":
    raise RuntimeError(f"Unexpected pixel format: {stream.get('pix_fmt')}")

encoded = base64.b64encode(OUT_MP4.read_bytes()).decode("ascii")
size_mb = OUT_MP4.stat().st_size / 1_000_000.0

print(
    f"Verified: H.264 | {stream['pix_fmt']} | "
    f"{stream['width']}×{stream['height']} | "
    f"{float(fmt['duration']):.3f} s | {size_mb:.3f} MB"
)

display(HTML(f"""
<div style="background:#000;padding:8px;max-width:1000px">
  <video width="960" controls autoplay muted playsinline preload="auto"
         style="display:block;width:100%;height:auto;background:#000">
    <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
    Your browser cannot play this embedded H.264 MP4.
  </video>
</div>
"""))
