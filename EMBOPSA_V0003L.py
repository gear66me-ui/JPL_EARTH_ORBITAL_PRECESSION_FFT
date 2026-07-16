# EMBOPSA_V0003L
# Eight-planet JPL DE440s video spanning one full Neptune orbital period.
# Black background, five-day cadence, 3x playback speed relative to V0003E.

from __future__ import annotations

import requests

VERSION = "V0003L"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "9668e61204321bd4aca4c2300aec0568656cc9ef/EMBOPSA_V0003E.py"
)

response = requests.get(SOURCE_URL, timeout=300)
response.raise_for_status()
source = response.text

patches = [
    ('# EMBOPSA_V0003E', '# EMBOPSA_V0003L'),
    ('# Daily-cadence eight-planet heliocentric video for 2016-07-16 through 2026-07-16.',
     '# Five-day-cadence eight-planet heliocentric video for 1861-07-16 through 2026-07-16.'),
    ('VERSION = "V0003E"', 'VERSION = "V0003L"'),
    ('START_DATE = date(2016, 7, 16)', 'START_DATE = date(1861, 7, 16)'),
    ('FPS = 12', 'FPS = 36'),
    ('TRAIL_DAYS = 365', 'TRAIL_DAYS = 12050'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_NEPTUNE_PERIOD_3X.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_NEPTUNE_PERIOD_1861_2026_3X_H264.mp4"'),
    ('n_days = (END_DATE - START_DATE).days + 1\ndates = [START_DATE + timedelta(days=i) for i in range(n_days)]',
     'STEP_DAYS = 5\nn_days = (END_DATE - START_DATE).days // STEP_DAYS + 1\ndates = [START_DATE + timedelta(days=i * STEP_DAYS) for i in range(n_days)]'),
    ('"Eight-Planet Motion — Daily Cadence"', '"Eight-Planet Motion — Neptune Full Period"'),
    ('day_label = f"Day {j + 1:,} of {n_days:,}"',
     'day_label = f"Frame {j + 1:,} of {n_days:,}"'),
    ('"One frame = one day"', '"One frame = five days • 36 FPS"'),
    ('print(f"Daily cadence: {n_days:,} frames | Duration: {n_days / FPS:.1f} s | FPS: {FPS}")',
     'print(f"Five-day cadence: {n_days:,} frames | Duration: {n_days / FPS:.1f} s | FPS: {FPS}")'),
]

for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source token not found: {old}")
    source = source.replace(old, new, 1)

for token in [
    'START_DATE = date(1861, 7, 16)',
    'FPS = 36',
    'STEP_DAYS = 5',
    'NEPTUNE_PERIOD_1861_2026_3X_H264.mp4',
]:
    if token not in source:
        raise RuntimeError(f"Final verification failed; missing: {token}")

exec(compile(source, "EMBOPSA_V0003L_GENERATED.py", "exec"), globals(), globals())
