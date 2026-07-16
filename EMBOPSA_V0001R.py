# EMBOPSA_V0001R
# NO AI-GENERATED IMAGES.
# Deterministic Python / NumPy / Pandas / Matplotlib only.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from IPython.display import display

VERSION = "V0001R"

DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

C = {
    "bg": "#000000",
    "grid": "#28364A",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "gold": "#FFD166",
    "red": "#FF5C7A",
    "green": "#35E0A1",
}

response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()

df = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna().sort_values("t_kyr").reset_index(drop=True)

df["age_myr"] = -df["t_kyr"] / 1000.0
df["pole_radius"] = np.hypot(df["p"], df["q"])
df["inclination_deg"] = np.degrees(
    2.0 * np.arcsin(np.clip(df["pole_radius"], 0.0, 1.0))
)

today_row = df.loc[df["t_kyr"].abs().idxmin()]
today_i = float(today_row["inclination_deg"])

max_row = df.loc[df["inclination_deg"].idxmax()]
max_i = float(max_row["inclination_deg"])
max_age = float(max_row["age_myr"])

min_row = df.loc[df["inclination_deg"].idxmin()]
min_i = float(min_row["inclination_deg"])
min_age = float(min_row["age_myr"])

# One point every 100 kyr = every 100 original rows.
sample = df.iloc[::100].copy()

plt.close("all")
plt.ioff()
plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor": C["bg"],
    "savefig.facecolor": C["bg"],
    "text.color": C["text"],
    "axes.labelcolor": C["text"],
    "xtick.color": C["text"],
    "ytick.color": C["text"],
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(15, 7.5))

ax.plot(
    sample["age_myr"],
    sample["inclination_deg"],
    color=C["cyan"],
    linewidth=1.05,
    label="Earth orbital inclination",
)

ax.axhline(
    today_i,
    color=C["gold"],
    linestyle="--",
    linewidth=1.2,
    label=f"J2000 inclination = {today_i:.6f}°",
)

ax.axhline(
    3.0,
    color=C["red"],
    linestyle=":",
    linewidth=1.1,
    label="3° reference",
)

ax.scatter(
    [max_age],
    [max_i],
    color=C["green"],
    edgecolor=C["text"],
    s=70,
    zorder=5,
)

ax.annotate(
    f"Maximum = {max_i:.6f}°\nAge = {max_age:.3f} Myr",
    xy=(max_age, max_i),
    xytext=(max_age - 42, max_i + 0.15),
    color=C["text"],
    fontsize=10,
    arrowprops=dict(color=C["green"], width=0.8, headwidth=6),
    bbox=dict(facecolor=C["bg"], edgecolor=C["grid"], alpha=0.9),
)

ax.set_title(
    "Earth Orbital-Plane Inclination Relative to the La2010 Reference Plane",
    fontsize=17,
    fontweight="bold",
    pad=16,
)

ax.set_xlabel("Millions of years before J2000")
ax.set_ylabel("Inclination (degrees)")
ax.set_xlim(0, float(df["age_myr"].max()))
ax.set_ylim(0, max(3.25, max_i + 0.25))
ax.grid(True, color=C["grid"], linewidth=0.55, alpha=0.8)

for spine in ax.spines.values():
    spine.set_color(C["grid"])

legend = ax.legend(
    loc="upper right",
    facecolor=C["bg"],
    edgecolor=C["grid"],
    framealpha=0.95,
)

for text in legend.get_texts():
    text.set_color(C["text"])

summary = (
    f"J2000 inclination : {today_i:.6f}°\n"
    f"Minimum inclination: {min_i:.6f}° at {min_age:.3f} Myr before J2000\n"
    f"Maximum inclination: {max_i:.6f}° at {max_age:.3f} Myr before J2000\n"
    f"Displayed cadence  : 100 kyr\n"
    f"Data range         : 0 to {df['age_myr'].max():.3f} Myr before J2000"
)

ax.text(
    0.015,
    0.025,
    summary,
    transform=ax.transAxes,
    color=C["text"],
    fontsize=10,
    va="bottom",
    ha="left",
    family="monospace",
    bbox=dict(facecolor=C["bg"], edgecolor=C["grid"], alpha=0.88),
)

fig.tight_layout()

png_path = f"/content/EMBOPSA_EARTH_INCLINATION_250MYR_{VERSION}.png"
csv_path = f"/content/EMBOPSA_EARTH_INCLINATION_100KYR_{VERSION}.csv"

fig.savefig(png_path, dpi=220, bbox_inches="tight")
sample[["age_myr", "inclination_deg"]].to_csv(csv_path, index=False)

display(fig)
plt.close(fig)

print(summary)
print(png_path)
print(csv_path)
