# EMBOPSA_V0003E
# Daily-cadence eight-planet heliocentric video for 2016-07-16 through 2026-07-16.
# JPL DE440s ephemeris via Skyfield. Deterministic scientific graphics only.

from __future__ import annotations

import sys
import subprocess
import shutil
from datetime import date, timedelta
from pathlib import Path

try:
    import skyfield  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "skyfield"], check=True)

import cv2
import numpy as np
from skyfield.api import load
from IPython.display import Video, display

VERSION = "V0003E"
START_DATE = date(2016, 7, 16)
END_DATE = date(2026, 7, 16)
FPS = 12
WIDTH, HEIGHT = 1280, 720
TRAIL_DAYS = 365
OUT_DIR = Path("/content/embopsa_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS_DAILY_10Y.avi"
OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_DAILY_2016_2026_H264.mp4"

PLANETS = [
    ("Mercury", "mercury", (184, 184, 184), 5),
    ("Venus", "venus", (122, 194, 230), 6),
    ("Earth", "earth", (255, 214, 82), 7),
    ("Mars", "mars barycenter", (79, 107, 224), 6),
    ("Jupiter", "jupiter barycenter", (140, 176, 216), 9),
    ("Saturn", "saturn barycenter", (155, 210, 231), 8),
    ("Uranus", "uranus barycenter", (232, 215, 141), 8),
    ("Neptune", "neptune barycenter", (232, 120, 90), 8),
]

print(f"[{VERSION}] Loading JPL DE440s ephemeris…")
eph = load("de440s.bsp")
ts = load.timescale()

n_days = (END_DATE - START_DATE).days + 1
dates = [START_DATE + timedelta(days=i) for i in range(n_days)]
t = ts.utc([d.year for d in dates], [d.month for d in dates], [d.day for d in dates])
sun = eph["sun"]

positions = {}
for label, key, color, radius in PLANETS:
    body = eph[key]
    positions[label] = (body.at(t) - sun.at(t)).position.au.T.astype(np.float64)

# Radial display compression preserves direction while keeping Mercury through Neptune visible.
def compress_xyz(xyz: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(xyz, axis=1)
    display_r = np.sqrt(np.maximum(r, 1e-12))
    return xyz * (display_r / np.maximum(r, 1e-12))[:, None]

# Fixed oblique projection.
def project(xyz: np.ndarray) -> np.ndarray:
    az = np.deg2rad(36.0)
    el = np.deg2rad(23.0)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    X = np.cos(az) * x - np.sin(az) * y
    Y = np.sin(el) * (np.sin(az) * x + np.cos(az) * y) + np.cos(el) * z
    return np.column_stack((X, 1.75 * Y))

projected = {name: project(compress_xyz(xyz)) for name, xyz in positions.items()}
all_xy = np.vstack(list(projected.values()))
lim_x = max(6.1, float(np.max(np.abs(all_xy[:, 0]))) * 1.08)
lim_y = max(3.3, float(np.max(np.abs(all_xy[:, 1]))) * 1.12)

plot_left, plot_right = 60, WIDTH - 60
plot_top, plot_bottom = 75, HEIGHT - 78


def to_px(xy: np.ndarray) -> np.ndarray:
    px = plot_left + (xy[:, 0] + lim_x) / (2.0 * lim_x) * (plot_right - plot_left)
    py = plot_bottom - (xy[:, 1] + lim_y) / (2.0 * lim_y) * (plot_bottom - plot_top)
    return np.column_stack((px, py)).astype(np.int32)

pixel_tracks = {name: to_px(xy) for name, xy in projected.items()}

writer = cv2.VideoWriter(
    str(TEMP_AVI), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
)
if not writer.isOpened():
    raise RuntimeError("Could not open video writer.")

font = cv2.FONT_HERSHEY_SIMPLEX
sun_px = tuple(to_px(np.array([[0.0, 0.0]], dtype=float))[0])

try:
    for j, current_date in enumerate(dates):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        cv2.putText(frame, "Eight-Planet Motion — Daily Cadence", (320, 38), font, 0.95,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "JPL DE440s heliocentric positions • radial display scale = sqrt(AU)",
                    (326, 64), font, 0.43, (155, 165, 180), 1, cv2.LINE_AA)

        cv2.circle(frame, sun_px, 10, (102, 209, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, sun_px, 11, (255, 255, 255), 1, cv2.LINE_AA)

        first = max(0, j - TRAIL_DAYS)
        for label, key, color, radius in PLANETS:
            pts = pixel_tracks[label]
            trail = pts[first:j + 1]
            if len(trail) > 1:
                cv2.polylines(frame, [trail.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
            px, py = map(int, pts[j])
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, label, (px + 9, py - 7), font, 0.38, color, 1, cv2.LINE_AA)

        date_label = current_date.strftime("%Y-%m-%d")
        day_label = f"Day {j + 1:,} of {n_days:,}"
        cv2.putText(frame, date_label, (500, HEIGHT - 34), font, 0.78,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, day_label, (20, HEIGHT - 20), font, 0.42,
                    (150, 160, 175), 1, cv2.LINE_AA)
        cv2.putText(frame, "One frame = one day", (WIDTH - 205, HEIGHT - 20), font, 0.42,
                    (150, 160, 175), 1, cv2.LINE_AA)

        writer.write(frame)
        if j % 100 == 0 or j == n_days - 1:
            print(f"\rRendering {j + 1}/{n_days}", end="")
finally:
    writer.release()

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg is unavailable.")
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error", "-i", str(TEMP_AVI),
    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT_MP4)
], check=True)
TEMP_AVI.unlink(missing_ok=True)

if not OUT_MP4.exists() or OUT_MP4.stat().st_size < 100000:
    raise RuntimeError("MP4 output was not created correctly.")

print(f"\nSaved: {OUT_MP4}")
print(f"Daily cadence: {n_days:,} frames | Duration: {n_days / FPS:.1f} s | FPS: {FPS}")
display(Video(str(OUT_MP4), embed=True, width=960))
try:
    from google.colab import files
    files.download(str(OUT_MP4))
except Exception as exc:
    print(exc)
