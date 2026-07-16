# V0001
# Audit reference: EMBOPSA 6000-point JPL Earth-Moon barycenter acquisition

from __future__ import annotations

import base64
import contextlib
import io
import math
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from astroquery.jplhorizons import Horizons
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "astroquery"])
    from astroquery.jplhorizons import Horizons

VERSION = "V0001"
AUDIT_NAME = "EMBOPSA"
LOCAL_TZ = ZoneInfo("America/Bogota")

TARGET_ID = "3"
TARGET_NAME = "Earth-Moon barycenter"
CENTER = "@10"
REFPLANE = "ecliptic"
ABERRATIONS = "geometric"

START_YEAR = -13000
STOP_YEAR = 16995
STEP_YEARS = 5
EXPECTED_ROWS = 6000
CHUNK_SIZE = 150
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 4.0

REPO = "gear66me-ui/JPL_EARTH_ORBITAL_PRECESSION_FFT"
BRANCH = "main"
REMOTE_STATES = "data/jpl/EMBOPSA_JPL_EMB_STATES_6000_V0001.csv"
REMOTE_NORMALS = "data/derived/EMBOPSA_ORBITAL_NORMAL_6000_V0001.csv"
REMOTE_AUDIT = "validation/EMBOPSA_ACQUISITION_AUDIT_V0001.csv"

OUT = Path("/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001_OUTPUT")
STATE_CSV = OUT / "EMBOPSA_JPL_EMB_STATES_6000_V0001.csv"
NORMAL_CSV = OUT / "EMBOPSA_ORBITAL_NORMAL_6000_V0001.csv"
AUDIT_CSV = OUT / "EMBOPSA_ACQUISITION_AUDIT_V0001.csv"


def gregorian_jd(astronomical_year: int, month: int = 1, day: float = 1.0) -> float:
    """Proleptic Gregorian calendar to JD using astronomical year numbering."""
    year = int(astronomical_year)
    mon = int(month)
    if mon <= 2:
        year -= 1
        mon += 12
    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (mon + 1))
        + float(day)
        + b
        - 1524.5
    )


def chunks(values: list[tuple[int, float]], size: int) -> Iterable[list[tuple[int, float]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def query_chunk(epoch_chunk: list[tuple[int, float]], chunk_index: int, chunk_total: int) -> pd.DataFrame:
    years = [item[0] for item in epoch_chunk]
    requested_jd = np.asarray([item[1] for item in epoch_chunk], dtype=float)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                table = Horizons(
                    id=TARGET_ID,
                    id_type="majorbody",
                    location=CENTER,
                    epochs=requested_jd.tolist(),
                ).vectors(refplane=REFPLANE, aberrations=ABERRATIONS)

            frame = table.to_pandas()
            required = {"datetime_jd", "x", "y", "z", "vx", "vy", "vz"}
            missing = required.difference(frame.columns)
            if missing:
                raise RuntimeError(f"REJECTED missing JPL columns: {sorted(missing)}")
            if len(frame) != len(epoch_chunk):
                raise RuntimeError(
                    f"REJECTED chunk {chunk_index}/{chunk_total}: requested {len(epoch_chunk)}, returned {len(frame)}"
                )

            frame = frame.sort_values("datetime_jd").reset_index(drop=True)
            returned_jd = frame["datetime_jd"].to_numpy(dtype=float)
            if not np.allclose(returned_jd, requested_jd, rtol=0.0, atol=2.0e-7):
                max_delta = float(np.max(np.abs(returned_jd - requested_jd)))
                raise RuntimeError(f"REJECTED JPL epoch mismatch; max |delta JD|={max_delta:.12e}")

            result = pd.DataFrame(
                {
                    "sample_index": np.arange(len(years), dtype=int),
                    "astronomical_year": np.asarray(years, dtype=int),
                    "jd_tdb_requested": requested_jd,
                    "jd_tdb_returned": returned_jd,
                    "x_au": frame["x"].to_numpy(dtype=float),
                    "y_au": frame["y"].to_numpy(dtype=float),
                    "z_au": frame["z"].to_numpy(dtype=float),
                    "vx_au_per_day": frame["vx"].to_numpy(dtype=float),
                    "vy_au_per_day": frame["vy"].to_numpy(dtype=float),
                    "vz_au_per_day": frame["vz"].to_numpy(dtype=float),
                }
            )
            print(
                f"JPL CHUNK {chunk_index:02d}/{chunk_total:02d} | "
                f"years {years[0]:+07d} to {years[-1]:+07d} | rows={len(result)}"
            )
            return result
        except Exception as exc:
            last_error = exc
            print(
                f"DEBUG retry {attempt}/{MAX_RETRIES} for chunk {chunk_index}/{chunk_total}: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(f"REJECTED JPL query failed for chunk {chunk_index}/{chunk_total}: {last_error}")


def derive_orbital_normals(states: pd.DataFrame) -> pd.DataFrame:
    r = states[["x_au", "y_au", "z_au"]].to_numpy(dtype=float)
    v = states[["vx_au_per_day", "vy_au_per_day", "vz_au_per_day"]].to_numpy(dtype=float)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h, axis=1)
    if np.any(~np.isfinite(h_mag)) or np.any(h_mag <= 0.0):
        raise RuntimeError("REJECTED invalid orbital angular-momentum magnitude")
    n = h / h_mag[:, None]
    inclination_deg = np.degrees(np.arccos(np.clip(n[:, 2], -1.0, 1.0)))
    node_longitude_deg = np.mod(np.degrees(np.arctan2(n[:, 0], -n[:, 1])), 360.0)

    return pd.DataFrame(
        {
            "sample_index": states["sample_index"].to_numpy(dtype=int),
            "astronomical_year": states["astronomical_year"].to_numpy(dtype=int),
            "jd_tdb": states["jd_tdb_returned"].to_numpy(dtype=float),
            "normal_x": n[:, 0],
            "normal_y": n[:, 1],
            "normal_z": n[:, 2],
            "h_magnitude_au2_per_day": h_mag,
            "inclination_deg_ecliptic": inclination_deg,
            "ascending_node_longitude_deg": node_longitude_deg,
        }
    )


def validate(states: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    years = states["astronomical_year"].to_numpy(dtype=int)
    jd = states["jd_tdb_returned"].to_numpy(dtype=float)
    n = normals[["normal_x", "normal_y", "normal_z"]].to_numpy(dtype=float)
    norm_error = np.abs(np.linalg.norm(n, axis=1) - 1.0)
    year_steps = np.diff(years)
    jd_steps = np.diff(jd)

    checks = {
        "requested_rows": EXPECTED_ROWS,
        "returned_rows": len(states),
        "unique_years": int(states["astronomical_year"].nunique()),
        "duplicate_year_rows": int(states["astronomical_year"].duplicated().sum()),
        "first_astronomical_year": int(years[0]),
        "last_astronomical_year": int(years[-1]),
        "minimum_year_step": int(np.min(year_steps)),
        "maximum_year_step": int(np.max(year_steps)),
        "minimum_jd_step_days": float(np.min(jd_steps)),
        "maximum_jd_step_days": float(np.max(jd_steps)),
        "maximum_epoch_return_error_days": float(
            np.max(np.abs(states["jd_tdb_returned"] - states["jd_tdb_requested"]))
        ),
        "maximum_unit_normal_error": float(np.max(norm_error)),
        "finite_state_values": bool(
            np.isfinite(
                states[
                    ["x_au", "y_au", "z_au", "vx_au_per_day", "vy_au_per_day", "vz_au_per_day"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "finite_normal_values": bool(np.isfinite(n).all()),
    }

    if len(states) != EXPECTED_ROWS:
        raise RuntimeError(f"REJECTED expected {EXPECTED_ROWS} rows, obtained {len(states)}")
    if checks["duplicate_year_rows"] != 0:
        raise RuntimeError("REJECTED duplicate astronomical years")
    if checks["minimum_year_step"] != STEP_YEARS or checks["maximum_year_step"] != STEP_YEARS:
        raise RuntimeError("REJECTED nonuniform astronomical-year cadence")
    if not checks["finite_state_values"] or not checks["finite_normal_values"]:
        raise RuntimeError("REJECTED nonfinite JPL or derived values")
    if checks["maximum_unit_normal_error"] > 2.0e-12:
        raise RuntimeError("REJECTED orbital-normal normalization error")

    return pd.DataFrame({"quantity": list(checks.keys()), "value": list(checks.values())})


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


def upload_text_file(local_path: Path, remote_path: str, token: str) -> str:
    api_url = f"https://api.github.com/repos/{REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    existing = requests.get(api_url, headers=headers, params={"ref": BRANCH}, timeout=60)
    payload = {
        "message": f"Publish {local_path.name} from {AUDIT_NAME} {VERSION}",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if existing.status_code == 200:
        payload["sha"] = existing.json()["sha"]
    elif existing.status_code != 404:
        raise RuntimeError(f"REJECTED GitHub lookup failed {existing.status_code}: {existing.text[:300]}")

    response = requests.put(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"REJECTED GitHub upload failed {response.status_code}: {response.text[:500]}")
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{remote_path}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    years = np.arange(START_YEAR, STOP_YEAR + STEP_YEARS, STEP_YEARS, dtype=int)
    if len(years) != EXPECTED_ROWS:
        raise RuntimeError(f"REJECTED generated {len(years)} epochs instead of {EXPECTED_ROWS}")
    epochs = [(int(year), gregorian_jd(int(year), 1, 1.0)) for year in years]
    epoch_chunks = list(chunks(epochs, CHUNK_SIZE))

    print(f"OUTPUT VERSION {VERSION}")
    print("CODE INPUTS")
    print(f"Notebook abbreviation : {AUDIT_NAME}")
    print(f"JPL target            : {TARGET_ID} | {TARGET_NAME}")
    print(f"JPL center            : {CENTER} | Sun center")
    print(f"Reference plane       : {REFPLANE}")
    print(f"Aberrations           : {ABERRATIONS}")
    print(f"Year range            : {START_YEAR:+d} to {STOP_YEAR:+d}")
    print(f"Cadence               : {STEP_YEARS} years")
    print(f"Requested samples     : {EXPECTED_ROWS}")
    print(f"Query chunks          : {len(epoch_chunks)} x up to {CHUNK_SIZE} epochs")

    print("COMMENTS")
    print("Astronomical year numbering is used: year 0 equals 1 BCE.")
    print("Epochs are January 1 00:00 on the proleptic Gregorian calendar, supplied to JPL as explicit JD values.")
    print("No manual orbital elements or published precession coefficients are used.")

    frames = [query_chunk(chunk, index, len(epoch_chunks)) for index, chunk in enumerate(epoch_chunks, start=1)]
    states = pd.concat(frames, ignore_index=True)
    states["sample_index"] = np.arange(len(states), dtype=int)
    normals = derive_orbital_normals(states)
    audit = validate(states, normals)

    states.to_csv(STATE_CSV, index=False, float_format="%.15f")
    normals.to_csv(NORMAL_CSV, index=False, float_format="%.15f")
    audit.to_csv(AUDIT_CSV, index=False)

    print("RESULTS")
    print(f"JPL state rows        : {len(states)}")
    print(f"Orbital-normal rows   : {len(normals)}")
    print(f"First epoch           : year {int(states.iloc[0]['astronomical_year']):+d} | JD {states.iloc[0]['jd_tdb_returned']:.6f}")
    print(f"Last epoch            : year {int(states.iloc[-1]['astronomical_year']):+d} | JD {states.iloc[-1]['jd_tdb_returned']:.6f}")
    print(f"Maximum |returned-requested JD| : {np.max(np.abs(states['jd_tdb_returned'] - states['jd_tdb_requested'])):.12e} day")
    print(f"Maximum unit-normal error       : {np.max(np.abs(np.linalg.norm(normals[['normal_x','normal_y','normal_z']].to_numpy(), axis=1) - 1.0)):.12e}")

    print("OUTPUT SUMMARY")
    print(STATE_CSV)
    print(NORMAL_CSV)
    print(AUDIT_CSV)

    token = github_token()
    if token:
        for local_path, remote_path in (
            (STATE_CSV, REMOTE_STATES),
            (NORMAL_CSV, REMOTE_NORMALS),
            (AUDIT_CSV, REMOTE_AUDIT),
        ):
            raw_url = upload_text_file(local_path, remote_path, token)
            print(f"GITHUB RAW CSV        : {raw_url}")
    else:
        print("REJECTED GitHub upload: GITHUB_TOKEN not available; local CSV files remain valid.")

    print("PAPER COMPARISON")
    print("NOT USED — this acquisition audit is based exclusively on JPL Horizons vectors.")

    print("EQUATION STATUS")
    print("VERIFIED JD conversion: proleptic Gregorian calendar to explicit Julian Date.")
    print("VERIFIED orbital normal: n = (r x v) / |r x v|.")
    print("VERIFIED cadence, uniqueness, finite values, epoch return agreement, and unit-normal closure.")

    print(datetime.now(LOCAL_TZ).strftime("LOCAL TIMESTAMP %Y-%m-%d %H:%M:%S %Z"))
    print(f"FINAL VERSION {VERSION}")


if __name__ == "__main__":
    main()

# V0001