# V0001D
# Audit reference: first EMB orbital-normal FFT and harmonic reconstruction
from __future__ import annotations

import base64
import io
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

VERSION = "V0001D"
LOCAL_TZ = ZoneInfo("America/Bogota")

REPOSITORY = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
BRANCH = "main"
NORMAL_URL = (
    "https://raw.githubusercontent.com/"
    f"{REPOSITORY}/{BRANCH}/data/derived/"
    "EMBOPSA_ORBITAL_NORMAL_3999_V0001C.csv"
)

OUT = Path(
    "/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "EMBOPSA_V0001D_OUTPUT"
)
N_HARMONICS = 20
TRAIN_FRACTION = 0.80
ARCSEC_PER_RAD = 206264.80624709636

FREQUENCY_CSV = OUT / "EMBOPSA_FFT_FREQUENCIES_V0001D.csv"
COEFFICIENT_CSV = OUT / "EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv"
RECONSTRUCTION_CSV = OUT / "EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv"
VALIDATION_CSV = OUT / "EMBOPSA_HOLDOUT_VALIDATION_V0001D.csv"
METRICS_CSV = OUT / "EMBOPSA_VALIDATION_METRICS_V0001D.csv"

COEFFICIENT_PNG = OUT / "EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.png"
RESIDUAL_PNG = OUT / "EMBOPSA_RESIDUALS_V0001D.png"
ERROR_PNG = OUT / "EMBOPSA_ANGULAR_ERRORS_V0001D.png"
EQUATION_PNG = OUT / "EMBOPSA_MODEL_EQUATION_V0001D.png"
SPECTRUM_PNG = OUT / "EMBOPSA_FFT_SPECTRUM_V0001D.png"

REMOTE_FILES = {
    FREQUENCY_CSV: "spectral/EMBOPSA_FFT_FREQUENCIES_V0001D.csv",
    COEFFICIENT_CSV: "spectral/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.csv",
    RECONSTRUCTION_CSV: "reconstruction/EMBOPSA_HARMONIC_RECONSTRUCTION_V0001D.csv",
    VALIDATION_CSV: "validation/EMBOPSA_HOLDOUT_VALIDATION_V0001D.csv",
    METRICS_CSV: "validation/EMBOPSA_VALIDATION_METRICS_V0001D.csv",
    COEFFICIENT_PNG: "figures/EMBOPSA_HARMONIC_COEFFICIENTS_V0001D.png",
    RESIDUAL_PNG: "figures/EMBOPSA_RESIDUALS_V0001D.png",
    ERROR_PNG: "figures/EMBOPSA_ANGULAR_ERRORS_V0001D.png",
    EQUATION_PNG: "figures/EMBOPSA_MODEL_EQUATION_V0001D.png",
    SPECTRUM_PNG: "figures/EMBOPSA_FFT_SPECTRUM_V0001D.png",
}


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    try:
        from google.colab import userdata
        value = userdata.get("GITHUB_TOKEN")
        return value.strip() if value else None
    except Exception:
        return None


def publish_file(local_path: Path, remote_path: str, token: str) -> str:
    api_url = (
        f"https://api.github.com/repos/{REPOSITORY}/contents/{remote_path}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    lookup = requests.get(
        api_url,
        headers=headers,
        params={"ref": BRANCH},
        timeout=60,
    )
    payload: dict[str, str] = {
        "message": f"Publish {local_path.name}",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if lookup.status_code == 200:
        payload["sha"] = lookup.json()["sha"]
    elif lookup.status_code != 404:
        raise RuntimeError(
            f"GitHub lookup failed HTTP {lookup.status_code}: "
            f"{lookup.text[:300]}"
        )
    response = requests.put(
        api_url,
        headers=headers,
        json=payload,
        timeout=180,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub upload failed HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return (
        f"https://raw.githubusercontent.com/"
        f"{REPOSITORY}/{BRANCH}/{remote_path}"
    )


def load_normals() -> pd.DataFrame:
    response = requests.get(NORMAL_URL, timeout=180)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    required = {
        "sample_index",
        "astronomical_year",
        "jd_tdb",
        "normal_x",
        "normal_y",
        "normal_z",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"REJECTED missing normal columns: {missing}")
    frame = frame.sort_values("astronomical_year").reset_index(drop=True)
    if len(frame) != 3999:
        raise RuntimeError(
            f"REJECTED expected 3999 orbital normals, obtained {len(frame)}"
        )
    years = frame["astronomical_year"].to_numpy(dtype=float)
    if not np.allclose(np.diff(years), 5.0, rtol=0.0, atol=0.0):
        raise RuntimeError("REJECTED nonuniform five-year cadence")
    normals = frame[
        ["normal_x", "normal_y", "normal_z"]
    ].to_numpy(dtype=float)
    closure = np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0))
    if closure > 2.0e-12:
        raise RuntimeError(
            f"REJECTED unit-normal closure {closure:.12e}"
        )
    return frame


def fixed_tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(mean_normal, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    e_p = np.cross(reference, mean_normal)
    e_p /= np.linalg.norm(e_p)
    e_q = np.cross(mean_normal, e_p)
    e_q /= np.linalg.norm(e_q)

    orientation = float(np.dot(np.cross(e_p, e_q), mean_normal))
    if orientation < 0.0:
        e_q *= -1.0

    return mean_normal, e_p, e_q


def tangent_coordinates(
    normals: np.ndarray,
    mean_normal: np.ndarray,
    e_p: np.ndarray,
    e_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = normals @ mean_normal
    if np.any(denominator <= 0.0):
        raise RuntimeError("REJECTED tangent-plane hemisphere crossing")
    p = (normals @ e_p) / denominator
    q = (normals @ e_q) / denominator
    return p, q


def design_matrix(
    t: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    columns = [
        np.ones_like(t, dtype=float),
        t,
    ]
    for frequency in frequencies:
        phase = 2.0 * np.pi * frequency * t
        columns.append(np.cos(phase))
        columns.append(np.sin(phase))
    return np.column_stack(columns)


def fit_complex_model(
    t: np.ndarray,
    z: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = design_matrix(t, frequencies)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, z, rcond=None)
    reconstruction = matrix @ coefficients
    return coefficients, reconstruction


def predict_complex(
    t: np.ndarray,
    frequencies: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return design_matrix(t, frequencies) @ coefficients


def angular_error_arcsec(
    p_true: np.ndarray,
    q_true: np.ndarray,
    p_model: np.ndarray,
    q_model: np.ndarray,
) -> np.ndarray:
    return np.hypot(
        p_true - p_model,
        q_true - q_model,
    ) * ARCSEC_PER_RAD


def pearson_and_r2(
    observed: np.ndarray,
    modeled: np.ndarray,
) -> tuple[float, float]:
    observed_real = np.concatenate([observed.real, observed.imag])
    modeled_real = np.concatenate([modeled.real, modeled.imag])
    correlation = float(np.corrcoef(observed_real, modeled_real)[0, 1])
    residual = observed_real - modeled_real
    total = observed_real - np.mean(observed_real)
    r_squared = 1.0 - float(
        np.sum(residual * residual) / np.sum(total * total)
    )
    return correlation, r_squared


def frequency_table(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
) -> pd.DataFrame:
    amplitude = 2.0 * np.abs(spectrum) / len(spectrum)
    period = np.divide(
        1.0,
        frequencies,
        out=np.full_like(frequencies, np.inf),
        where=frequencies != 0.0,
    )
    return pd.DataFrame({
        "rank": np.arange(1, len(frequencies) + 1, dtype=int),
        "frequency_cycles_per_year": frequencies,
        "period_years": period,
        "fft_complex_amplitude_rad": amplitude,
        "fft_complex_amplitude_arcsec": amplitude * ARCSEC_PER_RAD,
    })


def coefficient_table(
    frequencies: np.ndarray,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, frequency in enumerate(frequencies, start=1):
        cosine = coefficients[2 + 2 * (index - 1)]
        sine = coefficients[3 + 2 * (index - 1)]
        positive_rotating = 0.5 * (cosine - 1j * sine)
        negative_rotating = 0.5 * (cosine + 1j * sine)
        combined_amplitude = math.sqrt(
            abs(cosine) ** 2 + abs(sine) ** 2
        )
        rows.append({
            "rank": index,
            "frequency_cycles_per_year": frequency,
            "period_years": 1.0 / frequency,
            "cosine_real_rad": cosine.real,
            "cosine_imag_rad": cosine.imag,
            "sine_real_rad": sine.real,
            "sine_imag_rad": sine.imag,
            "combined_amplitude_rad": combined_amplitude,
            "combined_amplitude_arcsec": (
                combined_amplitude * ARCSEC_PER_RAD
            ),
            "positive_rotating_real_rad": positive_rotating.real,
            "positive_rotating_imag_rad": positive_rotating.imag,
            "negative_rotating_real_rad": negative_rotating.real,
            "negative_rotating_imag_rad": negative_rotating.imag,
        })
    return pd.DataFrame(rows)


def save_plots(
    years: np.ndarray,
    train_end: int,
    frequency_audit: pd.DataFrame,
    coefficient_audit: pd.DataFrame,
    p: np.ndarray,
    q: np.ndarray,
    p_model: np.ndarray,
    q_model: np.ndarray,
    errors: np.ndarray,
) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "font.size": 9,
    })

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(
        frequency_audit["period_years"],
        frequency_audit["fft_complex_amplitude_arcsec"],
        marker="o",
        markersize=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Period (years)")
    ax.set_ylabel("FFT complex amplitude (arcsec)")
    ax.set_title("EMB orbital-normal FFT spectrum — ranked components")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(SPECTRUM_PNG, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ranks = coefficient_audit["rank"].to_numpy(dtype=int)
    amplitudes = coefficient_audit[
        "combined_amplitude_arcsec"
    ].to_numpy(dtype=float)
    ax.bar(ranks, amplitudes)
    ax.set_xlabel("Harmonic rank")
    ax.set_ylabel("Combined coefficient amplitude (arcsec)")
    ax.set_title("Fitted harmonic coefficients")
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(COEFFICIENT_PNG, bbox_inches="tight")
    plt.close(fig)

    residual_p = (p - p_model) * ARCSEC_PER_RAD
    residual_q = (q - q_model) * ARCSEC_PER_RAD
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(years, residual_p, label="p residual")
    ax.plot(years, residual_q, label="q residual")
    ax.axvline(
        years[train_end],
        linewidth=0.7,
        linestyle="--",
        label="holdout begins",
    )
    ax.set_xlabel("Astronomical year")
    ax.set_ylabel("Residual (arcsec)")
    ax.set_title("Harmonic reconstruction residuals")
    ax.legend()
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(RESIDUAL_PNG, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(years, errors)
    ax.axvline(
        years[train_end],
        linewidth=0.7,
        linestyle="--",
        label="holdout begins",
    )
    ax.set_xlabel("Astronomical year")
    ax.set_ylabel("Tangent-plane angular error (arcsec)")
    ax.set_title("JPL versus harmonic-model angular error")
    ax.legend()
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.tight_layout()
    fig.savefig(ERROR_PNG, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.axis("off")
    equation_lines = [
        r"$\mathbf{h}(t)=\frac{\mathbf{r}(t)\times\mathbf{v}(t)}"
        r"{\left|\mathbf{r}(t)\times\mathbf{v}(t)\right|}$",
        r"$p(t)=\frac{\mathbf{h}(t)\cdot\mathbf{e}_{p}}"
        r"{\mathbf{h}(t)\cdot\overline{\mathbf{h}}},\qquad"
        r"q(t)=\frac{\mathbf{h}(t)\cdot\mathbf{e}_{q}}"
        r"{\mathbf{h}(t)\cdot\overline{\mathbf{h}}}$",
        r"$z(t)=p(t)+i\,q(t)$",
        r"$\widehat{z}(t)=c_{0}+c_{1}t+\sum_{k=1}^{K}"
        r"\left[a_{k}\cos(2\pi f_{k}t)+b_{k}\sin(2\pi f_{k}t)\right]$",
        r"$\epsilon(t)=206264.806247\,"
        r"\left|z(t)-\widehat{z}(t)\right|\ \mathrm{arcsec}$",
    ]
    y_positions = [0.87, 0.68, 0.51, 0.32, 0.12]
    for equation, y_position in zip(equation_lines, y_positions):
        ax.text(
            0.5,
            y_position,
            equation,
            ha="center",
            va="center",
            fontsize=15,
        )
    ax.set_title(
        "EMB orbital-normal tangent-plane harmonic model",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(EQUATION_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"OUTPUT VERSION {VERSION}")
    print("CODE INPUTS")
    print("Source normals        : V0001C repository CSV")
    print("JPL states            : 3999")
    print("Year range            : -9995 to +9995")
    print("Cadence               : 5 years")
    print(f"Training fraction     : {TRAIN_FRACTION:.2f}")
    print(f"Harmonic components   : {N_HARMONICS}")
    print("Detrending            : complex constant + linear trend")
    print("Validation            : final contiguous 20% JPL holdout")

    frame = load_normals()
    years = frame["astronomical_year"].to_numpy(dtype=float)
    normals = frame[
        ["normal_x", "normal_y", "normal_z"]
    ].to_numpy(dtype=float)

    mean_normal, e_p, e_q = fixed_tangent_basis(normals)
    p, q = tangent_coordinates(normals, mean_normal, e_p, e_q)
    z = p + 1j * q

    time_origin = years[0]
    t = years - time_origin
    train_end = int(math.floor(len(frame) * TRAIN_FRACTION))
    if train_end <= 2 * N_HARMONICS + 2:
        raise RuntimeError("REJECTED insufficient training samples")

    t_train = t[:train_end]
    z_train = z[:train_end]
    cadence_years = float(np.median(np.diff(t_train)))

    trend_matrix = np.column_stack([
        np.ones_like(t_train),
        t_train,
    ])
    trend_coefficients, _, _, _ = np.linalg.lstsq(
        trend_matrix,
        z_train,
        rcond=None,
    )
    detrended_train = z_train - trend_matrix @ trend_coefficients

    fft_values = np.fft.fft(detrended_train)
    fft_frequencies = np.fft.fftfreq(
        len(detrended_train),
        d=cadence_years,
    )
    positive = fft_frequencies > 0.0
    positive_frequencies = fft_frequencies[positive]
    positive_spectrum = fft_values[positive]

    ranked_indices = np.argsort(
        np.abs(positive_spectrum)
    )[::-1]
    selected_indices = ranked_indices[:N_HARMONICS]
    selected_frequencies = positive_frequencies[selected_indices]
    selected_spectrum = positive_spectrum[selected_indices]

    ordering = np.argsort(np.abs(selected_spectrum))[::-1]
    selected_frequencies = selected_frequencies[ordering]
    selected_spectrum = selected_spectrum[ordering]

    coefficients, train_reconstruction = fit_complex_model(
        t_train,
        z_train,
        selected_frequencies,
    )
    full_reconstruction = predict_complex(
        t,
        selected_frequencies,
        coefficients,
    )

    p_model = full_reconstruction.real
    q_model = full_reconstruction.imag
    errors = angular_error_arcsec(
        p, q, p_model, q_model
    )

    frequency_audit = frequency_table(
        selected_frequencies,
        selected_spectrum,
    )
    coefficient_audit = coefficient_table(
        selected_frequencies,
        coefficients,
    )

    train_correlation, train_r_squared = pearson_and_r2(
        z[:train_end],
        full_reconstruction[:train_end],
    )
    holdout_correlation, holdout_r_squared = pearson_and_r2(
        z[train_end:],
        full_reconstruction[train_end:],
    )

    train_errors = errors[:train_end]
    holdout_errors = errors[train_end:]
    holdout_years = years[train_end:]
    holdout_slope = float(np.polyfit(
        holdout_years - holdout_years[0],
        holdout_errors,
        1,
    )[0])

    reconstruction = pd.DataFrame({
        "sample_index": frame["sample_index"].to_numpy(dtype=int),
        "astronomical_year": years.astype(int),
        "jd_tdb": frame["jd_tdb"].to_numpy(dtype=float),
        "set": np.where(
            np.arange(len(frame)) < train_end,
            "TRAIN",
            "HOLDOUT",
        ),
        "p_jpl_rad": p,
        "q_jpl_rad": q,
        "p_model_rad": p_model,
        "q_model_rad": q_model,
        "p_residual_arcsec": (
            (p - p_model) * ARCSEC_PER_RAD
        ),
        "q_residual_arcsec": (
            (q - q_model) * ARCSEC_PER_RAD
        ),
        "angular_error_arcsec": errors,
    })

    validation = reconstruction.iloc[train_end:].copy()

    metrics = pd.DataFrame({
        "quantity": [
            "training_rows",
            "holdout_rows",
            "harmonic_components",
            "training_pearson_correlation",
            "training_r_squared",
            "training_rms_error_arcsec",
            "training_max_error_arcsec",
            "holdout_pearson_correlation",
            "holdout_r_squared",
            "holdout_rms_error_arcsec",
            "holdout_max_error_arcsec",
            "holdout_error_trend_arcsec_per_year",
            "mean_normal_x",
            "mean_normal_y",
            "mean_normal_z",
            "tangent_ep_x",
            "tangent_ep_y",
            "tangent_ep_z",
            "tangent_eq_x",
            "tangent_eq_y",
            "tangent_eq_z",
        ],
        "value": [
            train_end,
            len(frame) - train_end,
            N_HARMONICS,
            train_correlation,
            train_r_squared,
            float(np.sqrt(np.mean(train_errors ** 2))),
            float(np.max(train_errors)),
            holdout_correlation,
            holdout_r_squared,
            float(np.sqrt(np.mean(holdout_errors ** 2))),
            float(np.max(holdout_errors)),
            holdout_slope,
            mean_normal[0],
            mean_normal[1],
            mean_normal[2],
            e_p[0],
            e_p[1],
            e_p[2],
            e_q[0],
            e_q[1],
            e_q[2],
        ],
    })

    frequency_audit.to_csv(
        FREQUENCY_CSV,
        index=False,
        float_format="%.15e",
    )
    coefficient_audit.to_csv(
        COEFFICIENT_CSV,
        index=False,
        float_format="%.15e",
    )
    reconstruction.to_csv(
        RECONSTRUCTION_CSV,
        index=False,
        float_format="%.15e",
    )
    validation.to_csv(
        VALIDATION_CSV,
        index=False,
        float_format="%.15e",
    )
    metrics.to_csv(
        METRICS_CSV,
        index=False,
        float_format="%.15e",
    )

    save_plots(
        years,
        train_end,
        frequency_audit,
        coefficient_audit,
        p,
        q,
        p_model,
        q_model,
        errors,
    )

    print("RESULTS")
    print(
        f"TRAINING RMS ERROR    : "
        f"{np.sqrt(np.mean(train_errors ** 2)):.6f} arcsec"
    )
    print(
        f"HOLDOUT RMS ERROR     : "
        f"{np.sqrt(np.mean(holdout_errors ** 2)):.6f} arcsec"
    )
    print(
        f"HOLDOUT MAX ERROR     : "
        f"{np.max(holdout_errors):.6f} arcsec"
    )
    print(
        f"HOLDOUT CORRELATION   : "
        f"{holdout_correlation:.12f}"
    )
    print(
        f"HOLDOUT R-SQUARED     : "
        f"{holdout_r_squared:.12f}"
    )
    print(
        f"HOLDOUT ERROR TREND   : "
        f"{holdout_slope:.12e} arcsec/year"
    )

    token = github_token()
    if token:
        published = []
        for local_path, remote_path in REMOTE_FILES.items():
            published.append(
                publish_file(local_path, remote_path, token)
            )
        print("GITHUB OUTPUT STATUS  : PUBLISHED")
        for url in published:
            print(f"GITHUB RAW OUTPUT     : {url}")
    else:
        print(
            "GITHUB OUTPUT STATUS  : NOT PUBLISHED | "
            "GITHUB_TOKEN unavailable"
        )

    print("OUTPUT SUMMARY")
    for path in (
        FREQUENCY_CSV,
        COEFFICIENT_CSV,
        RECONSTRUCTION_CSV,
        VALIDATION_CSV,
        METRICS_CSV,
        SPECTRUM_PNG,
        COEFFICIENT_PNG,
        RESIDUAL_PNG,
        ERROR_PNG,
        EQUATION_PNG,
    ):
        print(path)

    print("EQUATION STATUS")
    print(
        "VERIFIED tangent coordinates: "
        "p=(h·ep)/(h·hbar), q=(h·eq)/(h·hbar)."
    )
    print(
        "VERIFIED model: complex linear trend plus selected "
        "FFT harmonic cosine/sine terms."
    )
    print(
        "VERIFIED angular error: "
        "206264.806247 * |z_JPL - z_model| arcsec."
    )
    print(
        datetime.now(LOCAL_TZ).strftime(
            "LOCAL TIMESTAMP %Y-%m-%d %H:%M:%S %Z"
        )
    )
    print(f"FINAL VERSION {VERSION}")


if __name__ == "__main__":
    main()
# V0001D
