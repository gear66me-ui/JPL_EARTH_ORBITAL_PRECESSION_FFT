# EMBOPSA_V0002Y
# Verified 3x zoom via deterministic center crop, no side panel, year text only.
# Deterministic scientific animation only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0002Y"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "3cdfe4c2ecc37dad74c0b2c68ef2b0dce9d51441/EMBOPSA_V0002X.py"
)

response = requests.get(BASE_URL, timeout=300)
response.raise_for_status()
source = response.text

source = source.replace('VERSION = "V0002X"', 'VERSION = "V0002Y"', 1)
source = source.replace(
    'EMBOPSA_TEMP_TRUE_ZOOM_3X_NO_PANEL.avi',
    'EMBOPSA_TEMP_TRUE_ZOOM_3X_CROPPED_NO_PANEL.avi',
    1,
)
source = source.replace(
    'EMBOPSA_EARTH_ORBIT_500KYR_231KYR_TRUE_ZOOM_3X_NO_PANEL_H264.mp4',
    'EMBOPSA_EARTH_ORBIT_500KYR_231KYR_TRUE_ZOOM_3X_CROPPED_NO_PANEL_H264.mp4',
    1,
)

old_resize = '''        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Verify the actual rendered orbit fills the frame before spending 20 minutes.
'''

new_resize = '''        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Deterministic center crop: remove unused black margins, then restore 1280x720.
        # Original verified footprint was 634 px wide. Cropping to 760 px raises it to
        # about 83% of the final frame width while retaining the title and year text.
        crop_left = 260
        crop_right = 1020
        frame = frame[:, crop_left:crop_right]
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

        # Verify the actual rendered orbit fills the frame before spending 20 minutes.
'''

if old_resize not in source:
    raise RuntimeError("Resize/verification block not found in V0002X.")
source = source.replace(old_resize, new_resize, 1)

source = source.replace(
    'if width_fraction < 0.78 or height_fraction < 0.48:',
    'if width_fraction < 0.78 or height_fraction < 0.68:',
    1,
)
source = source.replace(
    '"3x zoom verification failed: orbit does not occupy enough of the frame."',
    '"3x cropped zoom verification failed: orbit does not occupy enough of the frame."',
    1,
)

required = [
    'crop_left = 260',
    'crop_right = 1020',
    'TRUE_ZOOM_3X_CROPPED_NO_PANEL_H264.mp4',
    'width_fraction < 0.78 or height_fraction < 0.68',
    'panel.set_visible(False)',
    'Bottom year text only',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Final source verification failed; missing: {token}")

exec(compile(source, "EMBOPSA_V0002Y_GENERATED.py", "exec"), globals(), globals())
