# EMBOPSA_V0002U
# Enlarged thin-line La2010a orbit video with bottom geological-time tracker.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002U"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "666dedb853f16eff3ba10bfe99e57dc863c1fb99/EMBOPSA_V0002T.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

source = source.replace('VERSION = "V0002T"', 'VERSION = "V0002U"')
source = source.replace(
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_TIME_TRACKER_H264.mp4"',
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_ENLARGED_TIME_TRACKER_H264.mp4"',
)
source = source.replace(
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES_TIME_TRACKER.avi"',
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_ENLARGED_TIME_TRACKER.avi"',
)

# Use substantially more of the 1280x720 frame for the orbit while retaining
# a compact right-side information panel and the bottom time tracker.
source = source.replace(
    'ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")',
    'ax = fig.add_axes([0.005, 0.075, 0.815, 0.885], projection="3d", facecolor="black")',
)
source = source.replace(
    'panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")',
    'panel = fig.add_axes([0.825, 0.10, 0.168, 0.82], facecolor="black")',
)

# Tighten the camera framing without clipping the full approximately 1-AU orbit.
source = source.replace('ax.set_xlim(-1.12, 1.12)', 'ax.set_xlim(-1.045, 1.045)')
source = source.replace('ax.set_ylim(-1.12, 1.12)', 'ax.set_ylim(-1.045, 1.045)')
source = source.replace('ax.set_zlim(-0.45, 0.45)', 'ax.set_zlim(-0.34, 0.34)')
source = source.replace('ax.set_box_aspect((1, 1, 0.42))', 'ax.set_box_aspect((1, 1, 0.34))')

# Enlarge the Sun and Earth markers slightly while preserving very thin lines.
source = source.replace(
    's=28, color="#52D6FF", edgecolor="white", linewidth=0.35',
    's=38, color="#52D6FF", edgecolor="white", linewidth=0.30',
)
source = source.replace(
    's=95, color="#FFD166", edgecolor="white", linewidth=0.35',
    's=125, color="#FFD166", edgecolor="white", linewidth=0.30',
)

# Move and widen the bottom time tracker under the enlarged orbit.
source = source.replace(
    'timeline = fig.add_axes([0.065, 0.012, 0.665, 0.060], facecolor="black")',
    'timeline = fig.add_axes([0.035, 0.008, 0.765, 0.064], facecolor="black")',
)

exec(compile(source, "EMBOPSA_V0002U_GENERATED.py", "exec"), globals(), globals())
