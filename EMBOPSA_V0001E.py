# V0001E
# PROJECT CONTRACT
# 1. No AI-generated images.
# 2. All figures are produced deterministically with Python/Matplotlib only.
# 3. Scientific data are loaded only from the published V0001D CSV outputs.
# 4. No synthetic coefficients, frequencies, residuals, or errors are introduced.
# 5. Publication figures use explicit labels, units, legends, and source provenance.

from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

VERSION = "V0001E"
TZ = ZoneInfo("America/Bogota")
REPO = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

OUT = Path(
    "/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "EMBOPSA_V0001E_OUTPUT"
)

URLS = {
    "frequencies": f"{RAW}/spectral/EMBOPSA_FFT_FREQUENCIES_V0001D.csv",
    "coefficients": f"{RAW}/spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv",
    "reconstruction": f"{RAW}/reconstruction/EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv",
    "validation": f"{RAW}/validation/EMBOPSA_HOLDOUT_VALIDATION_V0001D.csv",
    "metrics": f"{RAW}/validation/EMBOPSA_VALIDATION_METRICS_V0001D.csv",
}

FILES = {
    "spectrum": OUT / "EMBOPSA_PUBLICATION_FFT_SPECTRUM_V0001E.png",
    "coefficients": OUT / "EMBOPSA_PUBLICATION_COEFFICIENTS_V0001E.png",
    "complex_plane": OUT / "EMBOPSA_PUBLICATION_COMPLEX_COEFFICIENTS_V0001E.png",
    "residuals": OUT / "EMBOPSA_PUBLICATION_RESIDUALS_V0001E.png",
    "errors": OUT / "EMBOPSA_PUBLICATION_HOLDOUT_ERRORS_V0001E.png",
    "equation": OUT / "EMBOPSA_PUBLICATION_EQUATION_V0001E.png",
    "table": OUT / "EMBOPSA_PUBLICATION_COEFFICIENTS_V0001E.csv",
    "audit": OUT / "EMBOPSA_PUBLICATION_AUDIT_V0001E.csv",
}

REMOTE = {
    FILES["spectrum"]: f"figures/{FILES['spectrum'].name}",
    FILES["coefficients"]: f"figures/{FILES['coefficients'].name}",
    FILES["complex_plane"]: f"figures/{FILES['complex_plane'].name}",
    FILES["residuals"]: f"figures/{FILES['residuals'].name}",
    FILES["errors"]: f"figures/{FILES['errors'].name}",
    FILES["equation"]: f"figures/{FILES['equation'].name}",
    FILES["table"]: f"spectral/{FILES['table'].name}",
    FILES["audit"]: f"validation/{FILES['audit'].name}",
}

COLORS = {
    "navy": "#0B1F3A",
    "blue": "#1F77B4",
    "cyan": "#17BECF",
    "teal": "#0F9D8A",
    "gold": "#D4A017",
    "orange": "#E67E22",
    "red": "#C0392B",
    "magenta": "#8E44AD",
    "gray": "#6B7280",
    "light": "#F4F7FB",
    "grid": "#CAD3DF",
    "white": "#FFFFFF",
}

def token() -> str | None:
    value = os.getenv("GITHUB_TOKEN")
    if value:
        return value.strip()
    try:
        from google.colab import userdata
        value = userdata.get("GITHUB_TOKEN")
        return value.strip() if value else None
    except Exception:
        return None

def fetch_csv(url: str) -> pd.DataFrame:
    response = requests.get(
        url,
        params={"cachebust": str(datetime.now().timestamp())},
        timeout=180,
    )
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))

def publish(local_path: Path, remote_path: str, github_token: str) -> str:
    url = f"https://api.github.com/repos/{REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    existing = requests.get(
        url,
        headers=headers,
        params={"ref": BRANCH},
        timeout=60,
    )
    payload = {
        "message": f"Publish {local_path.name}",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if existing.status_code == 200:
        payload["sha"] = existing.json()["sha"]
    elif existing.status_code != 404:
        raise RuntimeError(
            f"GitHub lookup failed HTTP {existing.status_code}: "
            f"{existing.text[:300]}"
        )
    response = requests.put(url, headers=headers, json=payload, timeout=180)
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub upload failed HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return f"{RAW}/{remote_path}"

def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(COLORS["white"])
    ax.set_title(title, fontsize=16, fontweight="bold", color=COLORS["navy"], pad=14)
    ax.set_xlabel(xlabel, fontsize=11, color=COLORS["navy"])
    ax.set_ylabel(ylabel, fontsize=11, color=COLORS["navy"])
    ax.tick_params(colors=COLORS["navy"], labelsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

def save_figure(fig, path: Path) -> None:
    fig.patch.set_facecolor(COLORS["light"])
    fig.savefig(
        path,
        dpi=320,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    plt.ioff()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 320,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "font.size": 10,
    })

    print(f"OUTPUT VERSION {VERSION}")
    print("PROJECT CONTRACT")
    print("AI-GENERATED IMAGES  : PROHIBITED")
    print("FIGURE ENGINE        : PYTHON / MATPLOTLIB ONLY")
    print("SCIENTIFIC SOURCE    : PUBLISHED V0001D CSV FILES")

    freq = fetch_csv(URLS["frequencies"])
    coef = fetch_csv(URLS["coefficients"])
    rec = fetch_csv(URLS["reconstruction"])
    val = fetch_csv(URLS["validation"])
    metrics = fetch_csv(URLS["metrics"])

    required_freq = {
        "rank",
        "frequency_cycles_per_year",
        "period_years",
        "fft_complex_amplitude_arcsec",
    }
    required_coef = {
        "rank",
        "period_years",
        "cosine_real_rad",
        "cosine_imag_rad",
        "sine_real_rad",
        "sine_imag_rad",
        "combined_amplitude_arcsec",
    }
    required_rec = {
        "astronomical_year",
        "set",
        "p_residual_arcsec",
        "q_residual_arcsec",
        "angular_error_arcsec",
    }
    if not required_freq.issubset(freq.columns):
        raise RuntimeError("REJECTED frequency CSV columns")
    if not required_coef.issubset(coef.columns):
        raise RuntimeError("REJECTED coefficient CSV columns")
    if not required_rec.issubset(rec.columns):
        raise RuntimeError("REJECTED reconstruction CSV columns")

    freq = freq.sort_values("rank").reset_index(drop=True)
    coef = coef.sort_values("rank").reset_index(drop=True)
    rec = rec.sort_values("astronomical_year").reset_index(drop=True)
    val = val.sort_values("astronomical_year").reset_index(drop=True)

    coef_export = coef.copy()
    coef_export["cosine_magnitude_arcsec"] = (
        np.hypot(
            coef_export["cosine_real_rad"],
            coef_export["cosine_imag_rad"],
        ) * 206264.80624709636
    )
    coef_export["sine_magnitude_arcsec"] = (
        np.hypot(
            coef_export["sine_real_rad"],
            coef_export["sine_imag_rad"],
        ) * 206264.80624709636
    )
    coef_export.to_csv(FILES["table"], index=False, float_format="%.15e")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.plot(
        freq["period_years"],
        freq["fft_complex_amplitude_arcsec"],
        marker="o",
        markersize=5,
        color=COLORS["blue"],
        markerfacecolor=COLORS["gold"],
        markeredgecolor=COLORS["navy"],
    )
    ax.fill_between(
        freq["period_years"],
        freq["fft_complex_amplitude_arcsec"],
        alpha=0.14,
        color=COLORS["cyan"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        "Earth–Moon Barycenter Orbital-Normal Spectrum",
        "Period (years, logarithmic scale)",
        "FFT complex amplitude (arcsec, logarithmic scale)",
    )
    for _, row in freq.head(6).iterrows():
        ax.annotate(
            f"{row['period_years']:.1f} y",
            (row["period_years"], row["fft_complex_amplitude_arcsec"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["navy"],
        )
    ax.text(
        0.01,
        0.02,
        "Source: JPL Horizons EMB orbital normals, 5-year cadence, V0001D spectral audit",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )
    save_figure(fig, FILES["spectrum"])

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.bar(
        coef["rank"],
        coef["combined_amplitude_arcsec"],
        color=[
            COLORS["gold"] if rank <= 3 else
            COLORS["blue"] if rank <= 10 else
            COLORS["teal"]
            for rank in coef["rank"]
        ],
        edgecolor=COLORS["navy"],
        linewidth=0.5,
    )
    style_axes(
        ax,
        "Harmonic Coefficient Amplitudes",
        "Harmonic rank",
        "Combined fitted amplitude (arcsec)",
    )
    for bar, value in zip(bars[:6], coef["combined_amplitude_arcsec"].head(6)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["navy"],
        )
    ax.text(
        0.01,
        0.02,
        "Gold: dominant three harmonics | Blue: ranks 4–10 | Teal: ranks 11–20",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )
    save_figure(fig, FILES["coefficients"])

    cosine_complex = (
        coef["cosine_real_rad"].to_numpy()
        + 1j * coef["cosine_imag_rad"].to_numpy()
    ) * 206264.80624709636
    sine_complex = (
        coef["sine_real_rad"].to_numpy()
        + 1j * coef["sine_imag_rad"].to_numpy()
    ) * 206264.80624709636

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.scatter(
        cosine_complex.real,
        cosine_complex.imag,
        s=70,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        label="Cosine coefficients",
        zorder=3,
    )
    ax.scatter(
        sine_complex.real,
        sine_complex.imag,
        s=70,
        color=COLORS["orange"],
        edgecolor=COLORS["navy"],
        marker="s",
        label="Sine coefficients",
        zorder=3,
    )
    for rank, cval, sval in zip(coef["rank"], cosine_complex, sine_complex):
        if rank <= 8:
            ax.annotate(
                str(int(rank)),
                (cval.real, cval.imag),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["navy"],
            )
            ax.annotate(
                str(int(rank)),
                (sval.real, sval.imag),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=COLORS["navy"],
            )
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(
        ax,
        "Complex Harmonic Coefficients",
        "Real component (arcsec)",
        "Imaginary component (arcsec)",
    )
    ax.legend(frameon=True, facecolor=COLORS["white"], edgecolor=COLORS["grid"])
    save_figure(fig, FILES["complex_plane"])

    years = rec["astronomical_year"].to_numpy()
    holdout_start = rec.loc[rec["set"].eq("HOLDOUT"), "astronomical_year"].iloc[0]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.plot(
        years,
        rec["p_residual_arcsec"],
        color=COLORS["blue"],
        label="p residual",
    )
    ax.plot(
        years,
        rec["q_residual_arcsec"],
        color=COLORS["magenta"],
        label="q residual",
    )
    ax.axvspan(
        holdout_start,
        years.max(),
        color=COLORS["gold"],
        alpha=0.12,
        label="Holdout interval",
    )
    ax.axhline(0, color=COLORS["navy"], linewidth=0.8)
    style_axes(
        ax,
        "Orbital-Normal Harmonic Reconstruction Residuals",
        "Astronomical year",
        "Residual (arcsec)",
    )
    ax.legend(frameon=True, facecolor=COLORS["white"], edgecolor=COLORS["grid"])
    save_figure(fig, FILES["residuals"])

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    train = rec["set"].eq("TRAIN")
    holdout = rec["set"].eq("HOLDOUT")
    ax.plot(
        rec.loc[train, "astronomical_year"],
        rec.loc[train, "angular_error_arcsec"],
        color=COLORS["teal"],
        label="Training error",
    )
    ax.plot(
        rec.loc[holdout, "astronomical_year"],
        rec.loc[holdout, "angular_error_arcsec"],
        color=COLORS["red"],
        label="Holdout error",
    )
    ax.axvline(
        holdout_start,
        color=COLORS["gold"],
        linewidth=1.2,
        linestyle="--",
        label="Holdout begins",
    )
    style_axes(
        ax,
        "JPL Versus 20-Harmonic Model Error",
        "Astronomical year",
        "Tangent-plane angular error (arcsec)",
    )
    ax.legend(frameon=True, facecolor=COLORS["white"], edgecolor=COLORS["grid"])
    save_figure(fig, FILES["errors"])

    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    fig.patch.set_facecolor(COLORS["light"])
    ax.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.04, 0.05),
            0.92,
            0.88,
            facecolor=COLORS["white"],
            edgecolor=COLORS["navy"],
            linewidth=1.5,
        )
    )
    ax.text(
        0.5,
        0.88,
        "EMB Orbital-Normal Harmonic Reconstruction",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLORS["navy"],
    )
    equations = [
        "h(t) = [r(t) × v(t)] / |r(t) × v(t)|",
        "p(t) = [h(t) · eₚ] / [h(t) · h̄]",
        "q(t) = [h(t) · e_q] / [h(t) · h̄]",
        "z(t) = p(t) + i q(t)",
        "ẑ(t) = c₀ + c₁t + Σₖ [aₖ cos(2π fₖ t) + bₖ sin(2π fₖ t)]",
        "ε(t) = 206264.806247 |z(t) − ẑ(t)| arcsec",
    ]
    y_positions = [0.74, 0.63, 0.53, 0.42, 0.29, 0.16]
    equation_colors = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["teal"],
        COLORS["magenta"],
        COLORS["orange"],
        COLORS["red"],
    ]
    for equation, ypos, color in zip(equations, y_positions, equation_colors):
        ax.text(
            0.5,
            ypos,
            equation,
            ha="center",
            va="center",
            fontsize=15,
            color=color,
            fontweight="bold" if ypos in (0.74, 0.29, 0.16) else "normal",
        )
    ax.text(
        0.5,
        0.08,
        "Deterministic Matplotlib rendering — no AI-generated imagery",
        ha="center",
        fontsize=9,
        color=COLORS["gray"],
    )
    save_figure(fig, FILES["equation"])

    audit = pd.DataFrame({
        "quantity": [
            "version",
            "source_frequency_rows",
            "source_coefficient_rows",
            "source_reconstruction_rows",
            "source_holdout_rows",
            "ai_generated_images",
            "rendering_engine",
            "publication_figures",
            "publication_csv_files",
        ],
        "value": [
            VERSION,
            len(freq),
            len(coef),
            len(rec),
            len(val),
            "PROHIBITED",
            "PYTHON_MATPLOTLIB_ONLY",
            6,
            2,
        ],
    })
    audit.to_csv(FILES["audit"], index=False)

    github_token = token()
    if github_token:
        print("GITHUB OUTPUT STATUS  : PUBLISHED")
        for local_path, remote_path in REMOTE.items():
            print("GITHUB RAW OUTPUT     :", publish(local_path, remote_path, github_token))
    else:
        print("GITHUB OUTPUT STATUS  : NOT PUBLISHED | GITHUB_TOKEN unavailable")

    print("OUTPUT SUMMARY")
    for path in FILES.values():
        print(path)
    print("FINAL VERSION V0001E")
    plt.close("all")

if __name__ == "__main__":
    main()
# V0001E
