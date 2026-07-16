# EMBOPSA_V0002R
# Runs the verified H.264 renderer and automatically downloads the MP4.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import subprocess
from pathlib import Path

import requests
from IPython.display import HTML, display

VERSION = "V0002R"
RENDERER_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "b5586155bc3a77ed035e2f23b8a02417deb4f8c6/EMBOPSA_V0002P.py"
)
OUT_MP4 = Path(
    "/content/embopsa_video/"
    "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_H264.mp4"
)

print(f"[{VERSION}] Running verified deterministic renderer…")
renderer = requests.get(RENDERER_URL, timeout=300)
renderer.raise_for_status()
exec(compile(renderer.text, "EMBOPSA_V0002P.py", "exec"), globals(), globals())

if not OUT_MP4.exists() or OUT_MP4.stat().st_size < 100_000:
    raise RuntimeError(f"Rendered MP4 is missing or too small: {OUT_MP4}")

probe = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height",
        "-of", "default=noprint_wrappers=1",
        str(OUT_MP4),
    ],
    check=True,
    capture_output=True,
    text=True,
)
print(probe.stdout.strip())
print(f"Video ready: {OUT_MP4} ({OUT_MP4.stat().st_size/1_000_000:.3f} MB)")

# Inline playback is unreliable in the Android Colab interface.
# Force a real file transfer to the device instead.
try:
    from google.colab import files
    display(HTML("<b>Starting MP4 download to your device…</b>"))
    files.download(str(OUT_MP4))
except Exception as exc:
    display(HTML(
        f"<b>Automatic download could not start:</b> {exc}<br>"
        f"Open the Colab Files panel and download:<br><code>{OUT_MP4}</code>"
    ))
