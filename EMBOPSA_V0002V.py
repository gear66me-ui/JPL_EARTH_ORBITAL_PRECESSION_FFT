# EMBOPSA_V0002V
# Genuine camera-zoomed La2010a Earth-orbit video.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002V"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "666dedb853f16eff3ba10bfe99e57dc863c1fb99/EMBOPSA_V0002T.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

source = source.replace('VERSION = "V0002T"', 'VERSION = "V0002V"')
source = source.replace(
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_TIME_TRACKER_H264.mp4"',
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_TRUE_ZOOM_1P9X_H264.mp4"',
)
source = source.replace(
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES_TIME_TRACKER.avi"',
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_TRUE_ZOOM_1P9X.avi"',
)

# Expand the 3-D drawing region across almost the full frame.
source = source.replace(
    'ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")',
    'ax = fig.add_axes([-0.055, 0.085, 0.885, 0.865], projection="3d", facecolor="black")',
)

# Keep the information panel compact at the right edge.
source = source.replace(
    'panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")',
    'panel = fig.add_axes([0.805, 0.105, 0.185, 0.82], facecolor="black")',
)

# This is the actual camera zoom. Matplotlib's 3-D projection is enlarged 1.9×
# while preserving the complete orbit limits and all orbital geometry.
source = source.replace(
    'ax.set_box_aspect((1, 1, 0.42))',
    'ax.set_box_aspect((1, 1, 0.42), zoom=1.90)',
)

# Tighten framing slightly without cropping the approximately 1-AU ellipse.
source = source.replace('ax.set_xlim(-1.12, 1.12)', 'ax.set_xlim(-1.045, 1.045)')
source = source.replace('ax.set_ylim(-1.12, 1.12)', 'ax.set_ylim(-1.045, 1.045)')
source = source.replace('ax.set_zlim(-0.45, 0.45)', 'ax.set_zlim(-0.34, 0.34)')

# Move title upward so the enlarged orbit owns the center of the frame.
source = source.replace(
    'color="white", fontsize=19, pad=16, fontweight="bold"',
    'color="white", fontsize=18, pad=4, fontweight="bold"',
)

# Make the bottom timeline use nearly the full width.
source = source.replace(
    'timeline = fig.add_axes([0.065, 0.012, 0.665, 0.060], facecolor="black")',
    'timeline = fig.add_axes([0.045, 0.010, 0.900, 0.062], facecolor="black")',
)

required = [
    'zoom=1.90',
    'TRUE_ZOOM_1P9X_H264.mp4',
    '[-0.055, 0.085, 0.885, 0.865]',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Zoom patch failed; missing token: {token}")

exec(compile(source, "EMBOPSA_V0002V_GENERATED.py", "exec"), globals(), globals())
