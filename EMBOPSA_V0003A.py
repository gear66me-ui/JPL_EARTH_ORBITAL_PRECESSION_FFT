# EMBOPSA_V0003A
# Standalone full-frame La2010a Earth-orbit animation.
# No side panel; bottom year text only; deterministic graphics only.

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

VERSION = "V0003A"
DATA_URL = "https://raw.githubusercontent.com/gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
OUT_DIR = Path("/content/embopsa_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AVI = OUT_DIR / "EMBOPSA_TEMP_FULL_FRAME_ZOOM.avi"
OUT_MP4 = OUT_DIR / "EMBOPSA_EARTH_ORBIT_FULL_FRAME_ZOOM_H264.mp4"

WIDTH, HEIGHT = 1280, 720
FPS, DURATION_S = 25, 20
N_FRAMES = FPS * DURATION_S
START_KYR, END_KYR = -500.0, 0.0
TRAIL_INTERVAL_KYR = 10.0
TRAIL_MEMORY_KYR = 100.0
EARTH_REVOLUTIONS = 24.0
VERTICAL_EXAGGERATION = 2.35

print(f"[{VERSION}] Loading La2010a…")
r = requests.get(DATA_URL, timeout=300)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", header=None,
                 names=["t","a","l","k","h","q","p"], engine="python")
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().sort_values("t")
df = df[(df.t >= START_KYR) & (df.t <= END_KYR)]
if len(df) < 2:
    raise RuntimeError("Requested interval is unavailable.")

t = df.t.to_numpy(float)
a = df.a.to_numpy(float)
e = np.hypot(df.k.to_numpy(float), df.h.to_numpy(float))
varpi = np.unwrap(np.arctan2(df.h.to_numpy(float), df.k.to_numpy(float)))
Omega = np.unwrap(np.arctan2(df.p.to_numpy(float), df.q.to_numpy(float)))
inc = 2.0*np.arcsin(np.clip(np.hypot(df.p.to_numpy(float), df.q.to_numpy(float)),0,1))
omega = np.unwrap(varpi - Omega)

tf = np.linspace(START_KYR, END_KYR, N_FRAMES)
af = np.interp(tf,t,a); ef=np.interp(tf,t,e)
Of = np.interp(tf,t,Omega); inf=np.interp(tf,t,inc); wf=np.interp(tf,t,omega)
nu_grid = np.linspace(0,2*np.pi,420)

def xyz(a,e,i,O,w,nu):
    rr = a*(1-e*e)/(1+e*np.cos(nu))
    xp, yp = rr*np.cos(nu), rr*np.sin(nu)
    cw,sw=np.cos(w),np.sin(w); cO,sO=np.cos(O),np.sin(O); ci,si=np.cos(i),np.sin(i)
    x1=cw*xp-sw*yp; y1=sw*xp+cw*yp
    x=cO*x1-sO*ci*y1
    y=sO*x1+cO*ci*y1
    z=si*y1
    return x,y,z

def project(x,y,z):
    az=np.deg2rad(38.0); el=np.deg2rad(24.0)
    X=np.cos(az)*x-np.sin(az)*y
    Y=np.sin(el)*(np.sin(az)*x+np.cos(az)*y)+np.cos(el)*z
    return X, VERTICAL_EXAGGERATION*Y

def frame_bgr(fig):
    c=FigureCanvas(fig); c.draw()
    return cv2.cvtColor(np.asarray(c.buffer_rgba()),cv2.COLOR_RGBA2BGR)

writer=cv2.VideoWriter(str(TEMP_AVI),cv2.VideoWriter_fourcc(*"MJPG"),FPS,(WIDTH,HEIGHT))
if not writer.isOpened():
    raise RuntimeError("Could not open video writer.")

trail_step=max(1,int(round(TRAIL_INTERVAL_KYR/((END_KYR-START_KYR)/(N_FRAMES-1)))))
trail_keep=max(1,int(round(TRAIL_MEMORY_KYR/TRAIL_INTERVAL_KYR)))

try:
    for j in range(N_FRAMES):
        fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor="black")
        ax=fig.add_axes([0.01,0.105,0.98,0.82],facecolor="black")
        ax.set_aspect("auto")
        ax.axis("off")

        first=max(0,j-trail_step*trail_keep)
        for ti in range(first,j,trail_step):
            x,y,z=xyz(af[ti],ef[ti],inf[ti],Of[ti],wf[ti],nu_grid)
            X,Y=project(x,y,z)
            age=(j-ti)/max(1,trail_step*trail_keep)
            ax.plot(X,Y,color="#52D6FF",linewidth=0.30,alpha=0.03+0.14*(1-age))

        x,y,z=xyz(af[j],ef[j],inf[j],Of[j],wf[j],nu_grid)
        X,Y=project(x,y,z)
        ax.plot(X,Y,color="#35E0A1",linewidth=0.62,alpha=0.96)

        nu=2*np.pi*EARTH_REVOLUTIONS*j/(N_FRAMES-1)
        ex,ey,ez=xyz(af[j],ef[j],inf[j],Of[j],wf[j],nu)
        EX,EY=project(ex,ey,ez)
        ax.scatter([EX],[EY],s=34,color="#52D6FF",edgecolor="white",linewidth=0.35,zorder=5)
        ax.scatter([0],[0],s=105,color="#FFD166",edgecolor="white",linewidth=0.35,zorder=6)

        ax.set_xlim(-1.12,1.12)
        ax.set_ylim(-1.02,1.02)
        fig.text(0.5,0.965,"Earth Orbital-Plane Evolution — La2010a",
                 color="white",fontsize=18,fontweight="bold",ha="center",va="top")
        epoch=tf[j]
        label="Present — J2000" if epoch>=-0.0005 else f"{abs(epoch)*1000:,.0f} years ago   |   epoch {epoch/1000:+.6f} Myr"
        fig.text(0.5,0.035,label,color="white",fontsize=15,fontweight="bold",ha="center",va="center")
        fig.text(0.985,0.012,"Display projection: vertical scale ×2.35",
                 color="#8F99A8",fontsize=7,ha="right",va="bottom")

        frame=frame_bgr(fig); plt.close(fig)
        if frame.shape[:2]!=(HEIGHT,WIDTH):
            frame=cv2.resize(frame,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)

        if j==0:
            b,g,rr=cv2.split(frame)
            mask=(g>105)&(b>55)&(rr<210)
            mask[:55,:]=False; mask[-65:,:]=False
            ys,xs=np.where(mask)
            if xs.size<100:
                raise RuntimeError("Orbit pixels were not detected.")
            ow=int(xs.max()-xs.min()+1); oh=int(ys.max()-ys.min()+1)
            print(f"Verified first frame footprint: {ow}×{oh}px = {100*ow/WIDTH:.1f}% width × {100*oh/HEIGHT:.1f}% height")

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
if not OUT_MP4.exists() or OUT_MP4.stat().st_size<100000:
    raise RuntimeError("MP4 output was not created correctly.")
print(f"\nSaved: {OUT_MP4}")
display(Video(str(OUT_MP4),embed=True,width=960))
try:
    from google.colab import files
    files.download(str(OUT_MP4))
except Exception as exc:
    print(exc)
