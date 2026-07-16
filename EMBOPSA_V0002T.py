# EMBOPSA_V0002T
# Thin-line La2010a orbit video with an accurate bottom year tracker.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002T"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "969e2e11286724ca35de0c6a1bd8d4c55ccf01f5/EMBOPSA_V0002S.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

source = source.replace('VERSION = "V0002S"', 'VERSION = "V0002T"')
source = source.replace(
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES.avi"',
    'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES_TIME_TRACKER.avi"',
)
source = source.replace(
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_H264.mp4"',
    'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_TIME_TRACKER_H264.mp4"',
)

old_footer = '''        fig.text(
            0.035, 0.025,
            "Actual La2010a osculating orbit geometry • thin 10-kyr trail snapshots • accelerated Earth marker",
            color="#BFC7D5", fontsize=10
        )
'''

new_footer = '''        # Bottom geological-time tracker for this 500,000-year interval.
        timeline = fig.add_axes([0.065, 0.012, 0.665, 0.060], facecolor="black")
        timeline.set_xlim(0.0, 1.0)
        timeline.set_ylim(0.0, 1.0)
        timeline.axis("off")

        progress = (epoch_kyr - START_KYR) / (END_KYR - START_KYR)
        progress = float(np.clip(progress, 0.0, 1.0))

        timeline.plot([0.02, 0.98], [0.34, 0.34], color="#28364A", linewidth=1.1)
        timeline.plot([0.02, 0.02 + 0.96*progress], [0.34, 0.34],
                      color="#35E0A1", linewidth=1.1)
        timeline.scatter([0.02 + 0.96*progress], [0.34], s=18,
                         color="#52D6FF", edgecolor="white", linewidth=0.25, zorder=5)

        for tick_kyr in (-500, -400, -300, -200, -100, 0):
            tx = 0.02 + 0.96*((tick_kyr - START_KYR)/(END_KYR - START_KYR))
            timeline.plot([tx, tx], [0.27, 0.41], color="#667085", linewidth=0.45)
            tick_label = "Present" if tick_kyr == 0 else f"{abs(tick_kyr):.0f} kyr ago"
            timeline.text(tx, 0.02, tick_label, color="#AAB4C3", fontsize=7,
                          ha="center", va="bottom")

        if epoch_kyr < -0.0005:
            year_label = (
                f"{abs(epoch_kyr)*1000.0:,.0f} years ago"
                f"   |   epoch {epoch_kyr/1000.0:+.6f} Myr"
            )
        else:
            year_label = "Present — J2000"

        timeline.text(0.50, 0.72, year_label, color="white", fontsize=12,
                      fontweight="bold", ha="center", va="center")
'''

if old_footer not in source:
    raise RuntimeError("Could not locate the original footer block in V0002S.")

source = source.replace(old_footer, new_footer)
exec(compile(source, "EMBOPSA_V0002T_GENERATED.py", "exec"), globals(), globals())
