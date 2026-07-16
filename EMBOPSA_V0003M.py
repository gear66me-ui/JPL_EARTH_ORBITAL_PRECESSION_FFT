from __future__ import annotations
import requests
VERSION="V0003M"
url="https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/9668e61204321bd4aca4c2300aec0568656cc9ef/EMBOPSA_V0003E.py"
r=requests.get(url,timeout=300); r.raise_for_status(); source=r.text
repls={
'# EMBOPSA_V0003E':'# EMBOPSA_V0003M',
'VERSION = "V0003E"':'VERSION = "V0003M"',
'START_DATE = date(2016, 7, 16)':'START_DATE = date(1861, 7, 16)',
'FPS = 12':'FPS = 36',
'TRAIL_DAYS = 365':'TRAIL_DAYS = 12050',
'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y.avi"':'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_SEVEN_PLANETS_NEPTUNE_PERIOD_3X.avi"',
'OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_H264.mp4"':'OUT_MP4 = OUT_DIR / "EMBOPSA_SEVEN_PLANETS_NEPTUNE_PERIOD_1861_2026_3X_H264.mp4"',
'n_days = (END_DATE - START_DATE).days + 1\ndates = [START_DATE + timedelta(days=i) for i in range(n_days)]':'STEP_DAYS = 5\nn_days = (END_DATE - START_DATE).days // STEP_DAYS + 1\ndates = [START_DATE + timedelta(days=i * STEP_DAYS) for i in range(n_days)]',
'"Eight-Planet Motion — Daily Cadence"':'"Seven-Planet Motion — Neptune Full Period"',
'"One frame = one day"':'"One frame = five days • 36 FPS"'
}
for a,b in repls.items():
    if a not in source: raise RuntimeError(f"Missing source token: {a}")
    source=source.replace(a,b,1)
earth='    ("Earth", "earth", (255, 214, 82), 7),\n'
if earth not in source: raise RuntimeError("Earth row not found")
source=source.replace(earth,'',1)
for name in ["Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune"]:
    if f'("{name}",' not in source: raise RuntimeError(f"Missing planet: {name}")
exec(compile(source,"EMBOPSA_V0003M_GENERATED.py","exec"),globals(),globals())
