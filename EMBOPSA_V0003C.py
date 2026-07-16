# EMBOPSA_V0003C
# Deterministic eight-planet long-term orbital visualization.
# Earth uses La2010a geometry; other planets use an illustrative secular realization.
# NO AI-GENERATED IMAGES.

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

VERSION = "V0003C"
DATA_URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
OUT_DIR = Path("/content/embopsa_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_EIGHT_PLANETS.avi"
OUT_MP4 = OUT_DIR / "EMBOPSA_EIGHT_PLANETS_250MYR_H264.mp4"

WIDTH, HEIGHT = 1280, 720
FPS, DURATION_S = 25, 20
N_FRAMES = FPS * DURATION_S
START_MYR, END_MYR = -250.0, 0.0
NU = np.linspace(0.0, 2.0*np.pi, 420)

# J2000 mean elements; angles in degrees. Secular terms below are illustrative only.
PLANETS = [
    ("Mercury", 0.387098, 0.205630, 7.005, 48.331, 29.125, 87.969, "#B8B8B8", 15),
    ("Venus",   0.723332, 0.006772, 3.395, 76.680, 54.884, 224.701, "#E6C27A", 18),
    ("Earth",   1.000000, 0.016708, 0.000, -11.261, 114.208, 365.256, "#52D6FF", 21),
    ("Mars",    1.523679, 0.093400, 1.850, 49.558, 286.502, 686.980, "#E06B4F", 18),
    ("Jupiter", 5.202600, 0.048900, 1.304, 100.464, 273.867, 4332.59, "#D8B08C", 26),
    ("Saturn",  9.554900, 0.056500, 2.485, 113.665, 339.392, 10759.2, "#E7D29B", 24),
    ("Uranus", 19.218400, 0.046300, 0.773, 74.006, 96.998, 30688.5, "#8DD7E8", 22),
    ("Neptune",30.110400, 0.009500, 1.770, 131.784, 273.187, 60182.0, "#5A78E8", 22),
]

print(f"[{VERSION}] Loading La2010a Earth solution…")
r = requests.get(DATA_URL, timeout=300)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None,
                 names=["t","a","l","k","h","q","p"], engine="python")
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().sort_values("t")

# La2010a time is kyr; use full -250 Myr to present record.
earth = df[(df.t >= -250000.0) & (df.t <= 0.0)].copy()
if len(earth) < 1000:
    raise RuntimeError("La2010a Earth interval is unavailable.")
et = earth.t.to_numpy(float) / 1000.0
ea = earth.a.to_numpy(float)
ee = np.hypot(earth.k.to_numpy(float), earth.h.to_numpy(float))
evarpi = np.unwrap(np.arctan2(earth.h.to_numpy(float), earth.k.to_numpy(float)))
eOmega = np.unwrap(np.arctan2(earth.p.to_numpy(float), earth.q.to_numpy(float)))
einc = 2.0*np.arcsin(np.clip(np.hypot(earth.p.to_numpy(float), earth.q.to_numpy(float)),0,1))
eomega = np.unwrap(evarpi - eOmega)

times = np.linspace(START_MYR, END_MYR, N_FRAMES)

# Display radius compression makes Mercury through Neptune visible together.
def display_radius(r_au):
    return np.sqrt(np.maximum(r_au, 0.0))

def orbit_xyz(a,e,inc,Omega,omega,nu):
    rr = a*(1-e*e)/(1+e*np.cos(nu))
    xp, yp = rr*np.cos(nu), rr*np.sin(nu)
    cw,sw=np.cos(omega),np.sin(omega)
    cO,sO=np.cos(Omega),np.sin(Omega)
    ci,si=np.cos(inc),np.sin(inc)
    x1=cw*xp-sw*yp; y1=sw*xp+cw*yp
    x=cO*x1-sO*ci*y1
    y=sO*x1+cO*ci*y1
    z=si*y1
    scale = display_radius(rr)/np.maximum(rr,1e-12)
    return x*scale, y*scale, z*scale

def project(x,y,z):
    az=np.deg2rad(38.0); el=np.deg2rad(24.0)
    X=np.cos(az)*x-np.sin(az)*y
    Y=np.sin(el)*(np.sin(az)*x+np.cos(az)*y)+np.cos(el)*z
    return X, 2.0*Y

def secular_elements(name,a0,e0,i0,O0,w0,t_myr):
    # Deterministic low-order secular realization, not an exact ancient ephemeris.
    idx = [p[0] for p in PLANETS].index(name) + 1
    phase = 0.61*idx
    e = np.clip(e0 + 0.015*np.sin(2*np.pi*t_myr/(0.7+0.23*idx)+phase), 0.001, 0.28)
    inc = np.deg2rad(max(0.05, i0 + 1.2*np.sin(2*np.pi*t_myr/(1.1+0.31*idx)+0.7*phase)))
    Omega = np.deg2rad(O0) + 2*np.pi*t_myr/(0.45+0.18*idx)
    omega = np.deg2rad(w0) - 2*np.pi*t_myr/(0.32+0.14*idx)
    return a0,e,inc,Omega,omega

def frame_bgr(fig):
    canvas=FigureCanvas(fig); canvas.draw()
    return cv2.cvtColor(np.asarray(canvas.buffer_rgba()),cv2.COLOR_RGBA2BGR)

writer=cv2.VideoWriter(str(TEMP_AVI),cv2.VideoWriter_fourcc(*"MJPG"),FPS,(WIDTH,HEIGHT))
if not writer.isOpened():
    raise RuntimeError("Could not open video writer.")

try:
    for j,t_myr in enumerate(times):
        fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor="black")
        ax=fig.add_axes([0.02,0.10,0.96,0.83],facecolor="black")
        ax.axis("off")
        ax.set_aspect("equal")

        for name,a0,e0,i0,O0,w0,period_d,color,size in PLANETS:
            if name == "Earth":
                a=np.interp(t_myr,et,ea)
                e=np.interp(t_myr,et,ee)
                inc=np.interp(t_myr,et,einc)
                Omega=np.interp(t_myr,et,eOmega)
                omega=np.interp(t_myr,et,eomega)
            else:
                a,e,inc,Omega,omega=secular_elements(name,a0,e0,i0,O0,w0,t_myr)

            x,y,z=orbit_xyz(a,e,inc,Omega,omega,NU)
            X,Y=project(x,y,z)
            ax.plot(X,Y,color=color,linewidth=0.52,alpha=0.72)

            revs = 3.0 + 0.8*np.sqrt(365.256/period_d)
            nu = 2*np.pi*revs*j/(N_FRAMES-1) + 0.8*[p[0] for p in PLANETS].index(name)
            px,py,pz=orbit_xyz(a,e,inc,Omega,omega,nu)
            PX,PY=project(px,py,pz)
            ax.scatter([PX],[PY],s=size,color=color,edgecolor="white",linewidth=0.25,zorder=5)
            ax.text(PX+0.045,PY+0.035,name,color=color,fontsize=7,ha="left",va="bottom")

        ax.scatter([0],[0],s=115,color="#FFD166",edgecolor="white",linewidth=0.35,zorder=8)
        lim=6.35
        ax.set_xlim(-lim,lim); ax.set_ylim(-3.55,3.55)

        fig.text(0.5,0.965,"Eight-Planet Orbital Architecture — 250 Myr to Present",
                 color="white",fontsize=18,fontweight="bold",ha="center",va="top")
        label = "Present — J2000" if t_myr >= -0.0005 else f"{abs(t_myr):,.3f} million years ago"
        fig.text(0.5,0.035,label,color="white",fontsize=15,fontweight="bold",ha="center",va="center")
        fig.text(0.985,0.010,
                 "Earth: La2010a  •  Other planets: illustrative secular realization  •  radial display scale = √AU",
                 color="#9AA5B5",fontsize=7,ha="right",va="bottom")

        frame=frame_bgr(fig); plt.close(fig)
        if frame.shape[:2] != (HEIGHT,WIDTH):
            frame=cv2.resize(frame,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
        writer.write(frame)
        if j%25==0 or j==N_FRAMES-1:
            print(f"\rRendering {j+1}/{N_FRAMES}",end="")
finally:
    writer.release()

if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg is unavailable.")
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(TEMP_AVI),
                "-c:v","libx264","-preset","medium","-crf","18",
                "-pix_fmt","yuv420p","-movflags","+faststart",str(OUT_MP4)],check=True)
TEMP_AVI.unlink(missing_ok=True)
if not OUT_MP4.exists() or OUT_MP4.stat().st_size < 100000:
    raise RuntimeError("MP4 output was not created correctly.")
print(f"\nSaved: {OUT_MP4}")
display(Video(str(OUT_MP4),embed=True,width=960))
try:
    from google.colab import files
    files.download(str(OUT_MP4))
except Exception as exc:
    print(exc)
