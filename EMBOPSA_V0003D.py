# EMBOPSA_V0003D
# Eight-planet video slowed to 0.25x playback speed (80 seconds).
# Deterministic scientific graphics only. NO AI-GENERATED IMAGES.

from __future__ import annotations

import requests

VERSION = "V0003D"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "69c4b63fe5493eb4a019bb22bc242dcea75e1424/EMBOPSA_V0003C.py"
)

r = requests.get(BASE_URL, timeout=300)
r.raise_for_status()
source = r.text

patches = [
    ('VERSION = "V0003C"', 'VERSION = "V0003D"'),
    ('TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS.avi"',
     'TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_025X.avi"'),
    ('OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_250MYR_H264.mp4"',
     'OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_250MYR_025X_SPEED_H264.mp4"'),
]

for old, new in patches:
    if old not in source:
        raise RuntimeError(f"Required source block not found: {old}")
    source = source.replace(old, new, 1)

old_ffmpeg = '''subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(TEMP_AVI),
                "-c:v","libx264","-preset","medium","-crf","18",
                "-pix_fmt","yuv420p","-movflags","+faststart",str(OUT_MP4)],check=True)'''

new_ffmpeg = '''subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(TEMP_AVI),
                "-vf","setpts=4.0*PTS,fps=25",
                "-c:v","libx264","-preset","medium","-crf","18",
                "-pix_fmt","yuv420p","-movflags","+faststart",str(OUT_MP4)],check=True)'''

if old_ffmpeg not in source:
    raise RuntimeError("FFmpeg encoding block not found.")
source = source.replace(old_ffmpeg, new_ffmpeg, 1)

source = source.replace(
    'print(f"\\nSaved: {OUT_MP4}")',
    'print(f"\\nSaved: {OUT_MP4} | Playback speed: 0.25x | Duration: 80 s")',
    1,
)

required = [
    'setpts=4.0*PTS,fps=25',
    '025X_SPEED_H264.mp4',
    'Playback speed: 0.25x',
]
for token in required:
    if token not in source:
        raise RuntimeError(f"Final verification failed; missing: {token}")

exec(compile(source, "EMBOPSA_V0003D_GENERATED.py", "exec"), globals(), globals())
