# EMBOPSA_V0002S
# Deterministic 20-second Earth-orbit precession animation from La2010a.
# Thin orbit lines plus accelerated Earth marker. NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.display import Video, display

VERSION = "V0002S"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)

OUT_DIR = Path("/content/embopsa_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_THIN_LINES.avi"
OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_500KYR_231KYR_THIN_EARTH_H264.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 25
DURATION_S = 20
N_FRAMES = FPS * DURATION_S
START_KYR = -500.0
END_KYR = 0.0
TRAIL_INTERVAL_KYR = 10.0
TRAIL_MEMORY_KYR = 100.0
ORBIT_SAMPLES = 360
REFERENCE_PERIOD_KYR = 231.0
EARTH_REVOLUTIONS_IN_VIDEO = 24.0

print(f"[{VERSION}] Loading La2010a…")
response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()
raw = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)
for c in raw.columns:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)

subset = raw[(raw["t_kyr"] >= START_KYR) & (raw["t_kyr"] <= END_KYR)].copy()
if len(subset) < 2:
    raise RuntimeError("Requested 500-kyr interval is not available.")

t_src = subset["t_kyr"].to_numpy(float)
a_src = subset["a"].to_numpy(float)
k_src = subset["k"].to_numpy(float)
h_src = subset["h"].to_numpy(float)
q_src = subset["q"].to_numpy(float)
p_src = subset["p"].to_numpy(float)

e_src = np.hypot(k_src, h_src)
varpi_src = np.unwrap(np.arctan2(h_src, k_src))
Omega_src = np.unwrap(np.arctan2(p_src, q_src))
inc_src = 2.0 * np.arcsin(np.clip(np.hypot(p_src, q_src), 0.0, 1.0))
omega_src = np.unwrap(varpi_src - Omega_src)

t_frames = np.linspace(START_KYR, END_KYR, N_FRAMES)
a_f = np.interp(t_frames, t_src, a_src)
e_f = np.interp(t_frames, t_src, e_src)
inc_f = np.interp(t_frames, t_src, inc_src)
Omega_f = np.interp(t_frames, t_src, Omega_src)
omega_f = np.interp(t_frames, t_src, omega_src)

nu_grid = np.linspace(0.0, 2.0 * np.pi, ORBIT_SAMPLES, endpoint=True)

def rotate_orbit(a, e, inc, Omega, omega, nu):
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(nu))
    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)
    cw, sw = np.cos(omega), np.sin(omega)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    x1 = cw * x_pf - sw * y_pf
    y1 = sw * x_pf + cw * y_pf
    x2 = x1
    y2 = ci * y1
    z2 = si * y1
    x = cO * x2 - sO * y2
    y = sO * x2 + cO * y2
    z = z2
    return x, y, z

def orbit_normal(inc, Omega):
    return np.array([
        np.sin(inc) * np.sin(Omega),
        -np.sin(inc) * np.cos(Omega),
        np.cos(inc),
    ])

def frame_to_bgr(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

writer = cv2.VideoWriter(
    str(TEMP_AVI),
    cv2.VideoWriter_fourcc(*"MJPG"),
    FPS,
    (WIDTH, HEIGHT),
)
if not writer.isOpened():
    raise RuntimeError("OpenCV could not open the temporary MJPEG writer.")

trail_step = max(1, int(round(
    TRAIL_INTERVAL_KYR / ((END_KYR - START_KYR) / (N_FRAMES - 1))
)))
trail_keep = max(1, int(round(TRAIL_MEMORY_KYR / TRAIL_INTERVAL_KYR)))

try:
    for j in range(N_FRAMES):
        fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor="black")
        ax = fig.add_axes([0.035, 0.08, 0.70, 0.86], projection="3d", facecolor="black")
        panel = fig.add_axes([0.755, 0.08, 0.225, 0.86], facecolor="black")
        panel.axis("off")

        lim = 1.15
        gx = np.linspace(-lim, lim, 2)
        gy = np.linspace(-lim, lim, 2)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.025, color="white", linewidth=0)

        first = max(0, j - trail_step * trail_keep)
        for ti in range(first, j, trail_step):
            age = (j - ti) / max(1, trail_step * trail_keep)
            alpha = 0.025 + 0.14 * (1.0 - age)
            xh, yh, zh = rotate_orbit(
                a_f[ti], e_f[ti], inc_f[ti], Omega_f[ti], omega_f[ti], nu_grid
            )
            ax.plot(xh, yh, zh, color="#52D6FF", linewidth=0.28, alpha=alpha)

        x, y, z = rotate_orbit(a_f[j], e_f[j], inc_f[j], Omega_f[j], omega_f[j], nu_grid)
        ax.plot(x, y, z, color="#35E0A1", linewidth=0.72, alpha=0.95)

        earth_nu = 2.0 * np.pi * EARTH_REVOLUTIONS_IN_VIDEO * j / (N_FRAMES - 1)
        ex, ey, ez = rotate_orbit(
            a_f[j], e_f[j], inc_f[j], Omega_f[j], omega_f[j], earth_nu
        )
        ax.scatter([ex], [ey], [ez], s=28, color="#52D6FF", edgecolor="white", linewidth=0.35)
        ax.scatter([0], [0], [0], s=95, color="#FFD166", edgecolor="white", linewidth=0.35)

        nvec = orbit_normal(inc_f[j], Omega_f[j])
        ax.quiver(
            0, 0, 0,
            0.72*nvec[0], 0.72*nvec[1], 0.72*nvec[2],
            color="#FF9F43", linewidth=0.65, arrow_length_ratio=0.08,
        )

        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_zlim(-0.45, 0.45)
        ax.set_box_aspect((1, 1, 0.42))
        ax.view_init(elev=24, azim=38)
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((0, 0, 0, 0))
        ax.set_title(
            "Earth Orbital-Plane Evolution — La2010a",
            color="white", fontsize=19, pad=16, fontweight="bold"
        )

        epoch_kyr = t_frames[j]
        elapsed_kyr = epoch_kyr - START_KYR
        phase = (elapsed_kyr % REFERENCE_PERIOD_KYR) / REFERENCE_PERIOD_KYR
        cycles = elapsed_kyr / REFERENCE_PERIOD_KYR

        panel.text(0.00, 0.97, "500,000-year animation", color="white",
                   fontsize=16, fontweight="bold", va="top")
        panel.text(0.00, 0.90, "Reference spectral cycle", color="#52D6FF",
                   fontsize=12, va="top")
        panel.text(0.00, 0.855, "231,000 years", color="#35E0A1",
                   fontsize=22, fontweight="bold", va="top")
        panel.text(0.00, 0.74, f"Epoch: {epoch_kyr:,.0f} kyr", color="white", fontsize=14)
        panel.text(0.00, 0.68, f"Cycle count: {cycles:,.3f}", color="white", fontsize=14)
        panel.text(0.00, 0.62, f"Cycle phase: {phase*360.0:,.1f}°", color="white", fontsize=14)
        panel.text(0.00, 0.53, f"Inclination: {np.degrees(inc_f[j]):.6f}°", color="white", fontsize=14)
        panel.text(0.00, 0.47, f"Eccentricity: {e_f[j]:.8f}", color="white", fontsize=14)
        panel.text(0.00, 0.41, f"Semimajor axis: {a_f[j]:.9f} AU", color="white", fontsize=14)
        panel.text(0.00, 0.30, "Trail cadence", color="#52D6FF", fontsize=12)
        panel.text(0.00, 0.255, "10,000 years", color="white", fontsize=16, fontweight="bold")
        panel.text(0.00, 0.19, "Earth marker", color="#52D6FF", fontsize=12)
        panel.text(0.00, 0.145, "24 accelerated revolutions", color="white", fontsize=14, fontweight="bold")
        panel.plot([0.00, 0.92], [0.06, 0.06], color="#28364A", linewidth=3.0, solid_capstyle="round")
        panel.plot([0.00, 0.92*phase], [0.06, 0.06], color="#35E0A1", linewidth=3.0, solid_capstyle="round")

        fig.text(
            0.035, 0.025,
            "Actual La2010a osculating orbit geometry • thin 10-kyr trail snapshots • accelerated Earth marker",
            color="#BFC7D5", fontsize=10
        )

        frame = frame_to_bgr(fig)
        plt.close(fig)
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        if j % 25 == 0 or j == N_FRAMES - 1:
            print(f"\rRendering {j+1:3d}/{N_FRAMES} frames", end="")
finally:
    writer.release()

print("\nEncoding H.264 MP4…")
if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg is not available in this Colab runtime.")
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", str(TEMP_AVI),
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-r", str(FPS),
    str(OUT_MP4),
], check=True)
TEMP_AVI.unlink(missing_ok=True)

if not OUT_MP4.exists() or OUT_MP4.stat().st_size < 100_000:
    raise RuntimeError("The H.264 MP4 was not created correctly.")

print(f"Saved: {OUT_MP4}")
print(f"Size: {OUT_MP4.stat().st_size/1_000_000:.3f} MB | Duration: {DURATION_S} s | FPS: {FPS}")
display(Video(str(OUT_MP4), embed=True, width=960))

try:
    from google.colab import files
    files.download(str(OUT_MP4))
except Exception as exc:
    print(f"Automatic download was not triggered: {exc}")
