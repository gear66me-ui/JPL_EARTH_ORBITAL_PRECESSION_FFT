from __future__ import annotations
import requests

VERSION = "V0003N"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/9668e61204321bd4aca4c2300aec0568656cc9ef/EMBOPSA_V0003E.py"

response = requests.get(SOURCE_URL, timeout=300)
response.raise_for_status()
source = response.text

replacements = [
    ('# EMBOPSA_V0003E', '# EMBOPSA_V0003N'),
    ('# Daily-cadence eight-planet heliocentric video for 2016-07-16 through 2026-07-16.',
     '# Daily-cadence eight-planet heliocentric video for 1861-07-16 through 2026-07-16.'),
    ('VERSION = "V0003E"', 'VERSION = "V0003N"'),
    ('START_DATE = date(2016, 7, 16)', 'START_DATE = date(1861, 7, 16)'),
    ('FPS = 12', 'FPS = 36'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_NEPTUNE_PERIOD_3X.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_NEPTUNE_PERIOD_1861_2026_3X_H264.mp4"'),
    ('"Eight-Planet Motion — Daily Cadence"',
     '"Eight-Planet Motion — Full Neptune Period — 3× Speed"'),
    ('"One frame = one day"', '"One frame = one day • 36 FPS"'),
]

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"Required source token not found: {old}")
    source = source.replace(old, new, 1)

required_planets = [
    'Mercury', 'Venus', 'Earth', 'Mars',
    'Jupiter', 'Saturn', 'Uranus', 'Neptune'
]
for planet in required_planets:
    if f'("{planet}",' not in source:
        raise RuntimeError(f"Missing planet in generated source: {planet}")

if source.count('("Earth",') != 1:
    raise RuntimeError("Earth verification failed.")
if 'FPS = 36' not in source:
    raise RuntimeError("3x playback verification failed.")
if 'START_DATE = date(1861, 7, 16)' not in source:
    raise RuntimeError("Neptune-period start-date verification failed.")

print(f"[{VERSION}] Verified 8 planets: " + ", ".join(required_planets))
exec(compile(source, "EMBOPSA_V0003N_GENERATED.py", "exec"), globals(), globals())
