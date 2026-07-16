# V0001H
# PROJECT CONTRACT
# NO AI-GENERATED IMAGES. PYTHON/MATPLOTLIB ONLY.
# Extrapolation visualization from the published V0001D harmonic coefficients.

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display, Image

VERSION = "V0001H"
REPO = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"

COEFFICIENT_URL = (
    f"{RAW}/spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv"
)

OUT = Path(
    "/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "EMBOPSA_V0001H_OUTPUT"
)
OUT.mkdir(parents=True, exist_ok=True)

FULL_PNG = OUT / "EMBOPSA_BLACK_EXTRAPOLATION_PM500K_V0001H.png"
ZOOM_PNG = OUT / "EMBOPSA_BLACK_EXTRAPOLATION_PM200K_V0001H.png"

ARCSEC_PER_RAD = 206264.80624709636
START_YEAR = -500_000
END_YEAR = 500_000
CADENCE_YEARS = 500

C = {
    "bg": "#000000",
    "panel": "#000000",
    "grid": "#263247",
    "text": "#F4F7FB",
    "muted": "#AAB4C3",
    "cyan": "#52D6FF",
    "blue": "#5B8CFF",
    "violet": "#B388FF",
    "magenta": "#FF6EC7",
    "orange": "#FF9F43",
    "gold": "#FFD166",
    "green": "#35E0A1",
    "red": "#FF5C7A",
}

def fetch_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["panel"])
    ax.set_title(
        title,
        color=C["text"],
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

def save(fig, path):
    fig.patch.set_facecolor(C["bg"])
    fig.savefig(
        path,
        dpi=320,
        bbox_inches="tight",
        facecolor=C["bg"],
    )
    plt.close(fig)

def reconstruct(coefficients: pd.DataFrame, years: np.ndarray) -> np.ndarray:
    t = years.astype(float)

    required = {
        "frequency_cycles_per_year",
        "cosine_real_rad",
        "cosine_imag_rad",
        "sine_real_rad",
        "sine_imag_rad",
    }
    missing = required.difference(coefficients.columns)
    if missing:
        raise RuntimeError(f"REJECTED missing coefficient columns: {sorted(missing)}")

    z = np.zeros_like(t, dtype=np.complex128)

    for row in coefficients.itertuples(index=False):
        f = float(row.frequency_cycles_per_year)
        a = complex(
            float(row.cosine_real_rad),
            float(row.cosine_imag_rad),
        )
        b = complex(
            float(row.sine_real_rad),
            float(row.sine_imag_rad),
        )
        phase = 2.0 * np.pi * f * t
        z += a * np.cos(phase) + b * np.sin(phase)

    return z

def plot_track(years, z, path, title, xmin, xmax):
    mask = (years >= xmin) & (years <= xmax)
    yy = years[mask]
    zz = z[mask]

    p_arcsec = zz.real * ARCSEC_PER_RAD
    q_arcsec = zz.imag * ARCSEC_PER_RAD
    amplitude_arcsec = np.abs(zz) * ARCSEC_PER_RAD

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.plot(
        yy,
        p_arcsec,
        color=C["cyan"],
        linewidth=1.05,
        label="p(t)",
    )
    ax.plot(
        yy,
        q_arcsec,
        color=C["violet"],
        linewidth=1.05,
        label="q(t)",
    )
    ax.plot(
        yy,
        amplitude_arcsec,
        color=C["gold"],
        linewidth=1.15,
        label="|z(t)|",
    )
    ax.axvline(0, color=C["red"], linestyle="--", linewidth=1.0)
    ax.axhline(0, color=C["grid"], linewidth=0.8)

    style(
        ax,
        title,
        "Astronomical year",
        "Tangent-plane coordinate / amplitude (arcsec)",
    )

    legend = ax.legend(
        facecolor=C["bg"],
        edgecolor=C["grid"],
        ncol=3,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color(C["text"])

    ax.text(
        0.01,
        0.02,
        (
            f"V0001D 20-harmonic extrapolation | "
            f"{CADENCE_YEARS}-year cadence | "
            "Python/Matplotlib only"
        ),
        transform=ax.transAxes,
        color=C["muted"],
        fontsize=8.5,
    )

    save(fig, path)

def main():
    plt.close("all")
    plt.ioff()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["panel"],
        "savefig.facecolor": C["bg"],
        "text.color": C["text"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["text"],
        "ytick.color": C["text"],
    })

    coefficients = fetch_csv(COEFFICIENT_URL).sort_values("rank")

    years = np.arange(
        START_YEAR,
        END_YEAR + CADENCE_YEARS,
        CADENCE_YEARS,
        dtype=float,
    )

    z = reconstruct(coefficients, years)

    plot_track(
        years,
        z,
        FULL_PNG,
        "EMB Harmonic Extrapolation: −500,000 to +500,000 Years",
        -500_000,
        500_000,
    )

    plot_track(
        years,
        z,
        ZOOM_PNG,
        "EMB Harmonic Extrapolation: Central ±200,000 Years",
        -200_000,
        200_000,
    )

    display(Image(filename=str(FULL_PNG)))
    display(Image(filename=str(ZOOM_PNG)))

    plt.close("all")

if __name__ == "__main__":
    main()
# V0001H
