# EMBOPSA_V0002M
# La2010a FFT-3 coefficient compression audit for Pearson r >= 0.999900.
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import traceback
import numpy as np
import pandas as pd
import requests
import ipywidgets as widgets
from scipy.stats import pearsonr
from IPython.display import display, HTML

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

VERSION = "V0002M"
DATA_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/La2010a_alkhqp3L.dat"
)
TARGET_R = 0.999900
SOURCE_SIZE_MB = 21.0

status = widgets.HTML(
    "<b style='color:#35E0A1'>Loading La2010a and auditing FFT compression…</b>"
)
display(status)

try:
    response = requests.get(DATA_URL, timeout=300)
    response.raise_for_status()

    raw_bytes = len(response.content)

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

    p = raw["p"].to_numpy(np.float64)
    q = raw["q"].to_numpy(np.float64)
    n = len(raw)

    i_actual = np.degrees(
        2.0 * np.arcsin(np.clip(np.hypot(p, q), 0.0, 1.0))
    )

    P = np.fft.rfft(p)
    Q = np.fft.rfft(q)
    bins = len(P)

    joint_energy = np.abs(P) ** 2 + np.abs(Q) ** 2
    non_dc = np.arange(1, bins, dtype=np.int64)
    ranked_non_dc = non_dc[np.argsort(joint_energy[non_dc])[::-1]]

    def reconstruct(count: int, dtype=np.complex128):
        count = int(count)
        chosen = ranked_non_dc[:max(0, count - 1)]
        indices = np.concatenate(([0], chosen)).astype(np.int64)

        P_kept = np.zeros(bins, dtype=dtype)
        Q_kept = np.zeros(bins, dtype=dtype)
        P_kept[indices] = P[indices].astype(dtype)
        Q_kept[indices] = Q[indices].astype(dtype)

        p_rec = np.fft.irfft(P_kept, n=n)
        q_rec = np.fft.irfft(Q_kept, n=n)

        i_rec = np.degrees(
            2.0 * np.arcsin(
                np.clip(np.hypot(p_rec, q_rec), 0.0, 1.0)
            )
        )

        r = float(np.corrcoef(i_actual, i_rec)[0, 1])
        err = i_rec - i_actual

        return {
            "count": count,
            "indices": indices,
            "P": P[indices].astype(dtype),
            "Q": Q[indices].astype(dtype),
            "r": r,
            "r2": r * r,
            "rmse_deg": float(np.sqrt(np.mean(err * err))),
            "mae_deg": float(np.mean(np.abs(err))),
            "max_deg": float(np.max(np.abs(err))),
            "energy_fraction": float(
                np.sum(joint_energy[indices]) / np.sum(joint_energy)
            ),
        }

    def find_minimum(dtype):
        lo = 1
        hi = 2

        while hi < bins:
            test = reconstruct(hi, dtype)
            status.value = (
                f"<b style='color:#35E0A1'>Searching {dtype.__name__}: "
                f"{hi:,} bins, r={test['r']:.9f}</b>"
            )
            if test["r"] >= TARGET_R:
                break
            lo = hi
            hi = min(bins, hi * 2)

        if hi == bins and reconstruct(hi, dtype)["r"] < TARGET_R:
            return reconstruct(hi, dtype)

        while lo + 1 < hi:
            mid = (lo + hi) // 2
            test = reconstruct(mid, dtype)
            status.value = (
                f"<b style='color:#35E0A1'>Binary search {dtype.__name__}: "
                f"{mid:,} bins, r={test['r']:.9f}</b>"
            )
            if test["r"] >= TARGET_R:
                hi = mid
            else:
                lo = mid

        return reconstruct(hi, dtype)

    def packaged_size(result):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            version=np.array([VERSION]),
            n=np.array([n], dtype=np.int64),
            dt_kyr=np.array([1.0], dtype=np.float64),
            indices=result["indices"].astype(np.uint32),
            P=result["P"],
            Q=result["Q"],
        )
        return len(buffer.getvalue())

    result128 = find_minimum(np.complex128)
    result64 = find_minimum(np.complex64)

    size128 = packaged_size(result128)
    size64 = packaged_size(result64)

    raw_pair_size = n * 2 * 8
    source_actual_mb = raw_bytes / 1_000_000.0

    status.value = (
        "<b style='color:#35E0A1'>FFT compression audit complete.</b>"
    )

    def row(label, result, packaged):
        scalar_bytes = result["P"].dtype.itemsize
        theoretical = result["count"] * (
            4 + scalar_bytes + scalar_bytes
        )
        return (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td>{result['count']:,}</td>"
            f"<td>{result['r']:.12f}</td>"
            f"<td>{result['r2']:.12f}</td>"
            f"<td>{result['energy_fraction']*100:.9f}%</td>"
            f"<td>{result['rmse_deg']:.9e}°</td>"
            f"<td>{result['max_deg']:.9e}°</td>"
            f"<td>{theoretical/1024:.3f} KiB</td>"
            f"<td>{packaged/1024:.3f} KiB</td>"
            f"<td>{raw_bytes/packaged:.2f}:1</td>"
            f"</tr>"
        )

    report = widgets.HTML(
        value=(
            "<h3>La2010a FFT-3 — minimum storage for Pearson "
            "r ≥ 0.999900</h3>"
            f"<b>Source rows:</b> {n:,}<br>"
            f"<b>Source text size actually downloaded:</b> "
            f"{raw_bytes:,} bytes = {source_actual_mb:.6f} MB<br>"
            f"<b>Direct binary p,q float64:</b> "
            f"{raw_pair_size:,} bytes = {raw_pair_size/1_000_000:.6f} MB<br>"
            f"<b>Available nonredundant FFT bins:</b> {bins:,}<br><br>"
            "<table style='border-collapse:collapse'>"
            "<tr>"
            "<th style='padding:6px'>Storage</th>"
            "<th style='padding:6px'>Bins</th>"
            "<th style='padding:6px'>Pearson r</th>"
            "<th style='padding:6px'>R²</th>"
            "<th style='padding:6px'>Energy</th>"
            "<th style='padding:6px'>RMSE</th>"
            "<th style='padding:6px'>Max error</th>"
            "<th style='padding:6px'>Raw coefficient payload</th>"
            "<th style='padding:6px'>Compressed NPZ</th>"
            "<th style='padding:6px'>Ratio vs source</th>"
            "</tr>"
            + row("complex128", result128, size128)
            + row("complex64", result64, size64)
            + "</table><br>"
            "<b>Payload formula:</b> one uint32 index + one complex "
            "P coefficient + one complex Q coefficient per retained bin.<br>"
            "<b style='color:#ff9f43'>This measures historical "
            "reconstruction correlation only. It does not validate future "
            "extrapolation.</b>"
        )
    )

    display(report)

except Exception:
    display(
        HTML(
            "<pre style='color:#ff6b6b'>"
            + traceback.format_exc()
            + "</pre>"
        )
    )
