# EMBOPSA_V0002W
# Genuine 2.25x zoomed La2010a Earth-orbit video with bottom time tracker.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002W"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "969e2e11286724ca35de0c6a1bd8d4c55ccf01f5/EMBOPSA_V0002S.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

patches = [
    ('VERSION = "V0002S"', 'VERSION = "V0002W"'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_TRUE_ZOOM_2P25X.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_TRUE_ZOOM_2P25X_H264.mp4"'),
    ('ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")',
     'ax = fig.add_axes([-0.13, 0.055, 1.02, 0.91], projection="3d", facecolor="black")'),
    ('panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")',
     'panel = fig.add_axes([0.825, 0.115, 0.165, 0.79], facecolor="black")'),
    ('ax.set_box_aspect((1, 1, 0.42))',
     'ax.set_box_aspect((1, 1, 0.42), zoom=2.25)'),
    ('ax.set_xlim(-1.12, 1.12)', 'ax.set_xlim(-1.035, 1.035)'),
    ('ax.set_ylim(-1.12, 1.12)', 'ax.set_ylim(-1.035, 1.035)'),
    ('ax.set_zlim(-0.45, 0.45)', 'ax.set_zlim(-0.30, 0.30)'),
    ('color="white", fontsize=19, pad=16, fontweight="bold"',
     'color="white", fontsize=17, pad=2, fontweight="bold"'),
]

for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source line not found: {old}")
    source = source.replace(old, new, 1)

old_footer = '''        fig.text(
            0.035, 0.025,
            "Actual La2010a osculating orbit geometry • thin 10-kyr trail snapshots • accelerated Earth marker",
            color="#BFC7D5", fontsize=10
        )
'''

new_footer = '''        timeline = fig.add_axes([0.045, 0.008, 0.905, 0.066], facecolor="black")
        timeline.set_xlim(0.0, 1.0)
        timeline.set_ylim(0.0, 1.0)
        timeline.axis("off")
        progress = float(np.clip((epoch_kyr - START_KYR) / (END_KYR - START_KYR), 0.0, 1.0))
        timeline.plot([0.02, 0.98], [0.32, 0.32], color="#28364A", linewidth=0.8)
        timeline.plot([0.02, 0.02 + 0.96*progress], [0.32, 0.32], color="#35E0A1", linewidth=0.8)
        timeline.scatter([0.02 + 0.96*progress], [0.32], s=15,
                         color="#52D6FF", edgecolor="white", linewidth=0.20, zorder=5)
        for tick_kyr in (-500, -400, -300, -200, -100, 0):
            tx = 0.02 + 0.96*((tick_kyr - START_KYR)/(END_KYR - START_KYR))
            timeline.plot([tx, tx], [0.25, 0.39], color="#667085", linewidth=0.35)
            label = "Present" if tick_kyr == 0 else f"{abs(tick_kyr):.0f} kyr ago"
            timeline.text(tx, 0.00, label, color="#AAB4C3", fontsize=7,
                          ha="center", va="bottom")
        year_label = "Present — J2000" if epoch_kyr >= -0.0005 else (
            f"{abs(epoch_kyr)*1000.0:,.0f} years ago   |   epoch {epoch_kyr/1000.0:+.6f} Myr"
        )
        timeline.text(0.50, 0.74, year_label, color="white", fontsize=12,
                      fontweight="bold", ha="center", va="center")
'''

if old_footer not in source:
    raise RuntimeError("Original footer block not found in V0002S.")
source = source.replace(old_footer, new_footer, 1)

required = ["zoom=2.25", "TRUE_ZOOM_2P25X_H264.mp4", "timeline = fig.add_axes"]
for token in required:
    if token not in source:
        raise RuntimeError(f"Final verification failed: {token}")

exec(compile(source, "EMBOPSA_V0002W_GENERATED.py", "exec"), globals(), globals())
