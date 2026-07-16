# EMBOPSA_V0003H
# Daily eight-planet video with user-supplied galaxy background.
# Deterministic scientific graphics only. No AI-generated imagery.

from __future__ import annotations

import requests

VERSION = "V0003H"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "9668e61204321bd4aca4c2300aec0568656cc9ef/EMBOPSA_V0003E.py"
)

from google.colab import files
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("Upload the selected galaxy image.")
IMAGE_NAME = next(iter(uploaded))

r = requests.get(BASE_URL, timeout=300)
r.raise_for_status()
source = r.text

patches = [
    ('VERSION = "V0003E"', 'VERSION = "V0003H"'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y_GALAXY_BG.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_GALAXY_BG_H264.mp4"'),
]
for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source token not found: {old}")
    source = source.replace(old, new, 1)

marker = 'font = cv2.FONT_HERSHEY_SIMPLEX\n'
insert = f'''_bg = cv2.imread({IMAGE_NAME!r}, cv2.IMREAD_COLOR)
if _bg is None:
    raise RuntimeError("Galaxy image could not be decoded.")
_h, _w = _bg.shape[:2]
_target = WIDTH / HEIGHT
_current = _w / _h
if _current > _target:
    _crop_w = int(round(_h * _target))
    _bg = _bg[:, :_crop_w]
else:
    _crop_h = int(round(_w / _target))
    _y0 = max(0, (_h - _crop_h) // 2)
    _bg = _bg[_y0:_y0 + _crop_h, :]
background = cv2.resize(_bg, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
background = cv2.convertScaleAbs(background, alpha=0.50, beta=0)
'''
if marker not in source:
    raise RuntimeError("Background insertion point not found.")
source = source.replace(marker, marker + insert, 1)

old_frame = '        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)\n'
if old_frame not in source:
    raise RuntimeError("Frame token not found.")
source = source.replace(old_frame, '        frame = background.copy()\n', 1)

source = source.replace(
    'print(f"\\nSaved: {OUT_MP4}")',
    'print(f"\\nSaved: {OUT_MP4} | Galaxy background: uploaded image, left crop")',
    1,
)

for token in ['frame = background.copy()', 'GALAXY_BG_H264.mp4', 'Galaxy background: uploaded image']:
    if token not in source:
        raise RuntimeError(f"Final verification failed; missing: {token}")

exec(compile(source, "EMBOPSA_V0003H_GENERATED.py", "exec"), globals(), globals())
