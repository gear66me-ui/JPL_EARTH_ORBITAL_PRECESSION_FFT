# EMBOPSA_V0002Z
# Standalone 3x-equivalent cropped La2010a Earth-orbit video.
# No side panel; bottom year text only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002Z"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "969e2e11286724ca35de0c6a1bd8d4c55ccf01f5/EMBOPSA_V0002S.py"
)

r = requests.get(BASE_URL, timeout=300)
r.raise_for_status()
source = r.text

patches = [
    ('VERSION = "V0002S"', 'VERSION = "V0002Z"'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_3X_CROP_NO_PANEL.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_3X_CROP_NO_PANEL_H264.mp4"'),
    ('ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")',
     'ax = fig.add_axes([-0.08, 0.075, 1.16, 0.88], projection="3d", facecolor="black")'),
    ('panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")\n        panel.axis("off")',
     'panel = fig.add_axes([0.0, 0.0, 0.001, 0.001], facecolor="black")\n        panel.axis("off")\n        panel.set_visible(False)'),
    ('ax.set_xlim(-1.12, 1.12)', 'ax.set_xlim(-1.03, 1.03)'),
    ('ax.set_ylim(-1.12, 1.12)', 'ax.set_ylim(-1.03, 1.03)'),
    ('ax.set_zlim(-0.45, 0.45)', 'ax.set_zlim(-0.27, 0.27)'),
]

for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source block not found: {old[:90]}")
    source = source.replace(old, new, 1)

old_footer = '''        fig.text(
            0.035, 0.025,
            "Actual La2010a osculating orbit geometry • thin 10-kyr trail snapshots • accelerated Earth marker",
            color="#BFC7D5", fontsize=10
        )
'''
new_footer = '''        if epoch_kyr < -0.0005:
            year_text = f"{abs(epoch_kyr)*1000.0:,.0f} years ago   |   epoch {epoch_kyr/1000.0:+.6f} Myr"
        else:
            year_text = "Present — J2000"
        fig.text(0.50, 0.028, year_text, color="white", fontsize=15,
                 fontweight="bold", ha="center", va="center")
'''
if old_footer not in source:
    raise RuntimeError("Footer block not found.")
source = source.replace(old_footer, new_footer, 1)

old_frame = '''        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        writer.write(frame)
'''
new_frame = '''        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Real pixel zoom: crop unused left/right black margins and rescale to 1280x720.
        # 634 px measured orbit footprint becomes approximately 1068 px, or 83.4% width.
        frame = frame[:, 260:1020]
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

        if j == 0:
            b, g, rr = cv2.split(frame)
            mask = (g > 115) & (b > 70) & (rr < 190)
            mask[:65, :] = False
            mask[-75:, :] = False
            ys, xs = np.where(mask)
            if xs.size < 100:
                raise RuntimeError("Orbit verification failed: orbit pixels not detected.")
            orbit_w = int(xs.max() - xs.min() + 1)
            orbit_h = int(ys.max() - ys.min() + 1)
            wf = orbit_w / WIDTH
            hf = orbit_h / HEIGHT
            print(f"\\nVerified zoomed orbit footprint: {orbit_w}×{orbit_h} px = {wf*100:.1f}% width × {hf*100:.1f}% height")
            if wf < 0.78 or hf < 0.68:
                raise RuntimeError("Zoom verification failed before rendering.")

        writer.write(frame)
'''
if old_frame not in source:
    raise RuntimeError("Frame writer block not found.")
source = source.replace(old_frame, new_frame, 1)

required = [
    'frame = frame[:, 260:1020]',
    'panel.set_visible(False)',
    'Verified zoomed orbit footprint',
    '3X_CROP_NO_PANEL_H264.mp4',
    'year_text =',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Final verification failed; missing {token}")

exec(compile(source, "EMBOPSA_V0002Z_GENERATED.py", "exec"), globals(), globals())
