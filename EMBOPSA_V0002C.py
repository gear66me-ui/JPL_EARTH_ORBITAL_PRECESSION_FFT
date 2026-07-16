# EMBOPSA_V0002C
# Pearson correlation and regression audit:
# La2010a actual inclination vs 180-harmonic spectral equation.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from IPython.display import display

VERSION = "V0002C"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "La2010a_alkhqp3L.dat"
)

TRAIN_END_KYR = -50_000.0
N_HARMONICS = 180

response = requests.get(DATA_URL, timeout=300)
response.raise_for_status()

raw = pd.read_csv(
    io.StringIO(response.text),
    sep=r"\s+",
    header=None,
    names=["t_kyr", "a", "l", "k", "h", "q", "p"],
    engine="python",
)

for column in raw.columns:
    raw[column] = pd.to_numeric(raw[column], errors="coerce")

raw = raw.dropna().sort_values("t_kyr").reset_index(drop=True)
raw["inclination_actual_deg"] = np.degrees(
    2.0 * np.arcsin(
        np.clip(np.hypot(raw["p"], raw["q"]), 0.0, 1.0)
    )
)

train = raw[raw["t_kyr"] <= TRAIN_END_KYR].copy()
dt_kyr = float(np.median(np.diff(train["t_kyr"])))
t_origin = float(train["t_kyr"].iloc[0])
n_train = len(train)

def fit_spectral(y: np.ndarray, n_harmonics: int):
    mean = float(np.mean(y))
    centered = y - mean
    coeff = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(len(centered), d=dt_kyr)

    candidates = np.arange(1, len(coeff))
    strongest = candidates[
        np.argsort(np.abs(coeff[candidates]))[-n_harmonics:]
    ]
    strongest = strongest[np.argsort(freq[strongest])]
    return mean, freq[strongest], coeff[strongest]

def evaluate_spectral(
    t_kyr: np.ndarray,
    mean: float,
    freq: np.ndarray,
    coeff: np.ndarray,
    chunk_size: int = 6000,
):
    result = np.empty(len(t_kyr), dtype=float)
    tau = np.asarray(t_kyr, dtype=float) - t_origin

    for start in range(0, len(tau), chunk_size):
        stop = min(len(tau), start + chunk_size)
        phase = np.exp(
            2j * np.pi * np.outer(tau[start:stop], freq)
        )
        result[start:stop] = (
            mean + (2.0 / n_train) * np.real(phase @ coeff)
        )
    return result

p_mean, p_freq, p_coeff = fit_spectral(
    train["p"].to_numpy(float), N_HARMONICS
)
q_mean, q_freq, q_coeff = fit_spectral(
    train["q"].to_numpy(float), N_HARMONICS
)

t_all = raw["t_kyr"].to_numpy(float)
raw["p_model"] = evaluate_spectral(t_all, p_mean, p_freq, p_coeff)
raw["q_model"] = evaluate_spectral(t_all, q_mean, q_freq, q_coeff)
raw["inclination_model_deg"] = np.degrees(
    2.0 * np.arcsin(
        np.clip(
            np.hypot(raw["p_model"], raw["q_model"]),
            0.0,
            1.0,
        )
    )
)

def regression_row(label: str, frame: pd.DataFrame):
    x = frame["inclination_model_deg"].to_numpy(float)
    y = frame["inclination_actual_deg"].to_numpy(float)

    fit = linregress(x, y)
    r, p_value = pearsonr(x, y)
    residual = y - x

    return {
        "Interval": label,
        "N": len(frame),
        "Intercept β₀ (deg)": fit.intercept,
        "Slope β₁": fit.slope,
        "Pearson r": r,
        "Pearson p": p_value,
        "R² = r²": r * r,
        "RMSE raw (deg)": np.sqrt(np.mean(residual**2)),
        "MAE raw (deg)": np.mean(np.abs(residual)),
        "Bias model−actual (deg)": np.mean(x - y),
        "Regression stderr": fit.stderr,
    }

intervals = {
    "Training −250 to −50 Myr": raw[raw["t_kyr"] <= TRAIN_END_KYR],
    "Validation −50 to 0 Myr": raw[raw["t_kyr"] > TRAIN_END_KYR],
    "Full −250 to 0 Myr": raw,
}

results = pd.DataFrame(
    [regression_row(label, frame) for label, frame in intervals.items()]
)

component_rows = []
for label, frame in intervals.items():
    for component in ("p", "q"):
        r, p_value = pearsonr(
            frame[f"{component}_model"].to_numpy(float),
            frame[component].to_numpy(float),
        )
        component_rows.append({
            "Interval": label,
            "Component": component,
            "Pearson r": r,
            "Pearson p": p_value,
            "R² = r²": r * r,
        })

components = pd.DataFrame(component_rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda value: f"{value:.12f}")

print("=" * 92)
print("LA2010a ACTUAL vs 180-HARMONIC SPECTRAL MODEL — PEARSON / REGRESSION AUDIT")
print("=" * 92)
print()
print("Regression definition:")
print("  inclination_actual = β₀ + β₁ × inclination_model")
print()
print(results.to_string(index=False))
print()
print("P AND Q COMPONENT CORRELATIONS")
print(components.to_string(index=False))

validation = intervals["Validation −50 to 0 Myr"].copy()
fit_v = linregress(
    validation["inclination_model_deg"],
    validation["inclination_actual_deg"],
)

print()
print("VALIDATION BEST-FIT EQUATION")
print(
    f"  i_actual = {fit_v.intercept:.12f} "
    f"+ ({fit_v.slope:.12f}) × i_model"
)
print(f"  Pearson r = {fit_v.rvalue:.12f}")
print(f"  R²        = {fit_v.rvalue**2:.12f}")
print(f"  p-value   = {fit_v.pvalue:.12e}")

stride = max(1, len(validation) // 15000)
plot_data = validation.iloc[::stride].copy()

x = plot_data["inclination_model_deg"].to_numpy(float)
y = plot_data["inclination_actual_deg"].to_numpy(float)
x_line = np.linspace(x.min(), x.max(), 300)
y_line = fit_v.intercept + fit_v.slope * x_line

plt.close("all")
plt.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "savefig.facecolor": "#000000",
    "text.color": "#F4F7FB",
    "axes.labelcolor": "#F4F7FB",
    "xtick.color": "#F4F7FB",
    "ytick.color": "#F4F7FB",
})

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(x, y, s=4, alpha=0.28)
ax.plot(x_line, y_line, linewidth=2, label="Least-squares regression")
ax.plot(
    [min(x.min(), y.min()), max(x.max(), y.max())],
    [min(x.min(), y.min()), max(x.max(), y.max())],
    linestyle="--",
    linewidth=1.2,
    label="Perfect agreement",
)

ax.set_title(
    "La2010a Withheld Validation\nActual vs Spectral-Model Inclination",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Spectral-model inclination (degrees)")
ax.set_ylabel("Actual La2010a inclination (degrees)")
ax.grid(True, alpha=0.25)
ax.legend()

summary = (
    f"β₀ = {fit_v.intercept:.6f}°\n"
    f"β₁ = {fit_v.slope:.6f}\n"
    f"r = {fit_v.rvalue:.6f}\n"
    f"R² = {fit_v.rvalue**2:.6f}"
)
ax.text(
    0.04,
    0.96,
    summary,
    transform=ax.transAxes,
    va="top",
    ha="left",
    family="monospace",
    bbox=dict(facecolor="#000000", edgecolor="#445064", alpha=0.9),
)

fig.tight_layout()
display(fig)
plt.close(fig)

results.to_csv(
    f"/content/EMBOPSA_REGRESSION_RESULTS_{VERSION}.csv",
    index=False,
)
components.to_csv(
    f"/content/EMBOPSA_COMPONENT_CORRELATIONS_{VERSION}.csv",
    index=False,
)

print()
print(f"/content/EMBOPSA_REGRESSION_RESULTS_{VERSION}.csv")
print(f"/content/EMBOPSA_COMPONENT_CORRELATIONS_{VERSION}.csv")
