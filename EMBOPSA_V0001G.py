# V0001G
# PROJECT CONTRACT: NO AI-GENERATED IMAGES. PYTHON/MATPLOTLIB ONLY.
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import display, Image

REPO = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
OUT = Path("/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001G_OUTPUT")
OUT.mkdir(parents=True, exist_ok=True)

URL = {
    "freq": f"{RAW}/spectral/EMBOPSA_FFT_FREQUENCIES_V0001D.csv",
    "coef": f"{RAW}/spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv",
    "rec": f"{RAW}/reconstruction/EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv",
}

PNG = [
    OUT / "EMBOPSA_BLACK_FFT_SPECTRUM_V0001G.png",
    OUT / "EMBOPSA_BLACK_COEFFICIENTS_V0001G.png",
    OUT / "EMBOPSA_BLACK_COMPLEX_COEFFICIENTS_V0001G.png",
    OUT / "EMBOPSA_BLACK_RESIDUALS_V0001G.png",
    OUT / "EMBOPSA_BLACK_ERRORS_V0001G.png",
    OUT / "EMBOPSA_BLACK_EQUATION_V0001G.png",
]

C = {
    "bg": "#000000",
    "panel": "#000000",
    "grid": "#2B3445",
    "text": "#F5F7FA",
    "muted": "#AAB4C3",
    "cyan": "#54D6FF",
    "blue": "#5B8CFF",
    "violet": "#B388FF",
    "magenta": "#FF6EC7",
    "orange": "#FF9F43",
    "gold": "#FFD166",
    "green": "#35E0A1",
    "red": "#FF5C7A",
}


def csv(url):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C["panel"])
    ax.set_title(title, color=C["text"], fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, color=C["muted"])
    ax.set_ylabel(ylabel, color=C["muted"])
    ax.tick_params(colors=C["text"])
    ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.75)
    for s in ax.spines.values():
        s.set_color(C["grid"])


def save(fig, path):
    fig.patch.set_facecolor(C["bg"])
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)


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

    freq = csv(URL["freq"]).sort_values("rank")
    coef = csv(URL["coef"]).sort_values("rank")
    rec = csv(URL["rec"]).sort_values("astronomical_year")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(freq.period_years, freq.fft_complex_amplitude_arcsec,
            color=C["cyan"], marker="o", markersize=5,
            markerfacecolor=C["gold"], markeredgecolor=C["bg"])
    ax.fill_between(freq.period_years, freq.fft_complex_amplitude_arcsec,
                    color=C["blue"], alpha=0.22)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "EMB Orbital-Normal FFT Spectrum",
          "Period (years, log scale)", "Amplitude (arcsec, log scale)")
    save(fig, PNG[0])

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    colors = [C["gold"] if r <= 3 else C["blue"] if r <= 10 else C["green"]
              for r in coef["rank"]]
    ax.bar(coef["rank"], coef["combined_amplitude_arcsec"],
           color=colors, edgecolor=C["text"], linewidth=0.45)
    style(ax, "Fitted Harmonic Coefficients",
          "Harmonic rank", "Combined amplitude (arcsec)")
    save(fig, PNG[1])

    cr = (coef.cosine_real_rad + 1j * coef.cosine_imag_rad) * 206264.80624709636
    sr = (coef.sine_real_rad + 1j * coef.sine_imag_rad) * 206264.80624709636
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.scatter(cr.real, cr.imag, s=72, color=C["cyan"], edgecolor=C["text"], label="Cosine")
    ax.scatter(sr.real, sr.imag, s=72, color=C["magenta"], edgecolor=C["text"], marker="s", label="Sine")
    ax.axhline(0, color=C["grid"], lw=0.8)
    ax.axvline(0, color=C["grid"], lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    style(ax, "Complex Harmonic Coefficients",
          "Real component (arcsec)", "Imaginary component (arcsec)")
    leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    save(fig, PNG[2])

    years = rec.astronomical_year.to_numpy()
    cut = rec.loc[rec["set"].eq("HOLDOUT"), "astronomical_year"].iloc[0]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(years, rec.p_residual_arcsec, color=C["cyan"], label="p residual")
    ax.plot(years, rec.q_residual_arcsec, color=C["violet"], label="q residual")
    ax.axvspan(cut, years.max(), color=C["gold"], alpha=0.12, label="Holdout")
    ax.axhline(0, color=C["grid"], lw=0.8)
    style(ax, "Harmonic Reconstruction Residuals",
          "Astronomical year", "Residual (arcsec)")
    leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    save(fig, PNG[3])

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    tr = rec["set"].eq("TRAIN")
    ho = rec["set"].eq("HOLDOUT")
    ax.plot(rec.loc[tr, "astronomical_year"], rec.loc[tr, "angular_error_arcsec"],
            color=C["green"], label="Training error")
    ax.plot(rec.loc[ho, "astronomical_year"], rec.loc[ho, "angular_error_arcsec"],
            color=C["red"], label="Holdout error")
    ax.axvline(cut, color=C["gold"], linestyle="--", linewidth=1.1, label="Holdout begins")
    style(ax, "JPL Versus 20-Harmonic Model Error",
          "Astronomical year", "Angular error (arcsec)")
    leg = ax.legend(facecolor=C["bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    save(fig, PNG[4])

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["panel"])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.88, "EMB Orbital-Normal Harmonic Reconstruction",
            ha="center", color=C["text"], fontsize=19, fontweight="bold")
    eqs = [
        ("h(t) = [r(t) × v(t)] / |r(t) × v(t)|", C["cyan"]),
        ("p(t) = [h(t) · eₚ] / [h(t) · h̄]", C["green"]),
        ("q(t) = [h(t) · e_q] / [h(t) · h̄]", C["green"]),
        ("z(t) = p(t) + i q(t)", C["magenta"]),
        ("ẑ(t) = c₀ + c₁t + Σₖ[aₖ cos(2πfₖt) + bₖ sin(2πfₖt)]", C["orange"]),
        ("ε(t) = 206264.806247 |z(t) − ẑ(t)| arcsec", C["red"]),
    ]
    for (txt, col), y in zip(eqs, [0.73, 0.62, 0.52, 0.41, 0.28, 0.15]):
        ax.text(0.5, y, txt, ha="center", va="center", color=col, fontsize=15)
    save(fig, PNG[5])

    for path in PNG:
        display(Image(filename=str(path)))

    plt.close("all")


if __name__ == "__main__":
    main()
