# EMBOPSA_V0002X
# True 3x full-frame La2010a Earth-orbit video, no side panel, year text only.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002X"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "969e2e11286724ca35de0c6a1bd8d4c55ccf01f5/EMBOPSA_V0002S.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

patches = [
    ('VERSION = "V0002S"', 'VERSION = "V0002X"'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_TRUE_ZOOM_3X_NO_PANEL.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_TRUE_ZOOM_3X_NO_PANEL_H264.mp4"'),
    ('ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")',
     'ax = fig.add_axes([-0.18, 0.075, 1.36, 0.88], projection="3d", facecolor="black")'),
    ('panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")\n        panel.axis("off")',
     'panel = fig.add_axes([0.0, 0.0, 0.001, 0.001], facecolor="black")\n        panel.axis("off")\n        panel.set_visible(False)'),
    ('ax.set_box_aspect((1, 1, 0.42))',
     'ax.set_box_aspect((1, 1, 0.42), zoom=3.00)'),
    ('ax.set_xlim(-1.12, 1.12)', 'ax.set_xlim(-1.025, 1.025)'),
    ('ax.set_ylim(-1.12, 1.12)', 'ax.set_ylim(-1.025, 1.025)'),
    ('ax.set_zlim(-0.45, 0.45)', 'ax.set_zlim(-0.25, 0.25)'),
    ('color="white", fontsize=19, pad=16, fontweight="bold"',
     'color="white", fontsize=18, pad=1, fontweight="bold"'),
]

for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source block not found: {old[:80]}")
    source = source.replace(old, new, 1)

old_footer = '''        fig.text(
            0.035, 0.025,
            "Actual La2010a osculating orbit geometry • thin 10-kyr trail snapshots • accelerated Earth marker",
            color="#BFC7D5", fontsize=10
        )
'''

new_footer = '''        # Bottom year text only; no side panel and no time/progress bar.
        if epoch_kyr < -0.0005:
            year_text = f"{abs(epoch_kyr)*1000.0:,.0f} years ago   |   epoch {epoch_kyr/1000.0:+.6f} Myr"
        else:
            year_text = "Present — J2000"
        fig.text(
            0.50, 0.028, year_text,
            color="white", fontsize=15, fontweight="bold",
            ha="center", va="center"
        )
'''

if old_footer not in source:
    raise RuntimeError("Original footer block not found.")
source = source.replace(old_footer, new_footer, 1)

old_write = '''        writer.write(frame)

        if j % 25 == 0 or j == N_FRAMES - 1:
'''

new_write = '''        # Verify the actual rendered orbit fills the frame before spending 20 minutes.
        if j == 0:
            b, g, r = cv2.split(frame)
            mask = (g > 115) & (b > 70) & (r < 190)
            mask[:70, :] = False
            mask[-80:, :] = False
            ys, xs = np.where(mask)
            if xs.size < 100:
                raise RuntimeError("Orbit verification failed: orbit pixels were not detected.")
            orbit_w = int(xs.max() - xs.min() + 1)
            orbit_h = int(ys.max() - ys.min() + 1)
            width_fraction = orbit_w / WIDTH
            height_fraction = orbit_h / HEIGHT
            print(
                f"\\nVerified first frame orbit footprint: "
                f"{orbit_w}×{orbit_h} px = "
                f"{100.0*width_fraction:.1f}% width × "
                f"{100.0*height_fraction:.1f}% height"
            )
            if width_fraction < 0.78 or height_fraction < 0.48:
                raise RuntimeError(
                    "3x zoom verification failed: orbit does not occupy enough of the frame."
                )

        writer.write(frame)

        if j % 25 == 0 or j == N_FRAMES - 1:
'''

if old_write not in source:
    raise RuntimeError("Writer block not found.")
source = source.replace(old_write, new_write, 1)

required = [
    'zoom=3.00',
    'panel.set_visible(False)',
    'TRUE_ZOOM_3X_NO_PANEL_H264.mp4',
    'Verified first frame orbit footprint',
    'width_fraction < 0.78',
    'Bottom year text only',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Final source verification failed; missing: {token}")

exec(compile(source, "EMBOPSA_V0002X_GENERATED.py", "exec"), globals(), globals())
