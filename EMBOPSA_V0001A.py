# V0001A
# Audit reference: resilient 6000-state JPL EMB acquisition
from __future__ import annotations

import csv
import io
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

VERSION = "V0001A"
TARGET = "3"
CENTER = "@10"
START_YEAR = -13000
STOP_YEAR = 16995
STEP_YEARS = 5
EXPECTED_ROWS = 6000
CHUNK_SIZE = 25
MAX_RETRIES = 8
TIMEOUT_SECONDS = 180
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUT = Path("/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001A_OUTPUT")
CACHE = OUT / "cache_jpl_chunks"
STATE_CSV = OUT / "EMBOPSA_JPL_EMB_STATES_6000_V0001A.csv"
NORMAL_CSV = OUT / "EMBOPSA_ORBITAL_NORMAL_6000_V0001A.csv"
AUDIT_CSV = OUT / "EMBOPSA_ACQUISITION_AUDIT_V0001A.csv"
LOCAL_TZ = ZoneInfo("America/Bogota")


def gregorian_jd(year: int, month: int = 1, day: float = 1.0) -> float:
    y, m = int(year), int(month)
    if m <= 2:
        y -= 1
        m += 12
    a = math.floor(y / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5)


def params(jds: np.ndarray) -> dict[str, str]:
    return {
        "format": "text",
        "EPHEM_TYPE": "VECTORS",
        "OUT_UNITS": "AU-D",
        "COMMAND": "'3'",
        "CENTER": "'@10'",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "ICRF",
        "VEC_CORR": "NONE",
        "VEC_DELTA_T": "NO",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "YES",
        "OBJ_DATA": "NO",
        "TIME_TYPE": "TDB",
        "TLIST_TYPE": "JD",
        "TLIST": "\n".join(f"{jd:.9f}" for jd in jds),
    }


def parse_vectors(text: str, years: np.ndarray, requested: np.ndarray) -> pd.DataFrame:
    if "$$SOE" not in text or "$$EOE" not in text:
        raise RuntimeError("REJECTED Horizons data markers absent")
    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    rows = [[x.strip() for x in row] for row in csv.reader(io.StringIO(body), skipinitialspace=True) if row and any(x.strip() for x in row)]
    if len(rows) != len(requested):
        raise RuntimeError(f"REJECTED requested {len(requested)} states, received {len(rows)}")
    values = []
    for row in rows:
        if len(row) < 8:
            raise RuntimeError(f"REJECTED malformed Horizons CSV row with {len(row)} fields")
        values.append([float(row[0]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    a = np.asarray(values, dtype=float)
    order = np.argsort(a[:, 0])
    a = a[order]
    requested = np.asarray(requested, dtype=float)[np.argsort(requested)]
    years = np.asarray(years, dtype=int)[np.argsort(requested)]
    if not np.allclose(a[:, 0], requested, rtol=0.0, atol=2.0e-7):
        raise RuntimeError("REJECTED JPL returned epochs do not match requested JDs")
    return pd.DataFrame({
        "astronomical_year": years,
        "jd_tdb_requested": requested,
        "jd_tdb_returned": a[:, 0],
        "x_au": a[:, 1], "y_au": a[:, 2], "z_au": a[:, 3],
        "vx_au_per_day": a[:, 4], "vy_au_per_day": a[:, 5], "vz_au_per_day": a[:, 6],
    })


def acquire_chunk(session: requests.Session, index: int, years: np.ndarray, jds: np.ndarray, total: int) -> pd.DataFrame:
    path = CACHE / f"EMBOPSA_V0001A_CHUNK_{index:03d}.csv"
    if path.exists():
        cached = pd.read_csv(path)
        if len(cached) == len(jds) and np.array_equal(cached["astronomical_year"].to_numpy(dtype=int), years) and np.allclose(cached["jd_tdb_requested"], jds, rtol=0.0, atol=5.0e-10):
            print(f"CACHE CHUNK {index:03d}/{total:03d} | rows={len(cached)}")
            return cached
        print(f"REJECTED cached chunk {index:03d}/{total:03d}; reacquiring")
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params(jds), timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            frame = parse_vectors(response.text, years, jds)
            frame.to_csv(path, index=False, float_format="%.15f")
            print(f"JPL CHUNK {index:03d}/{total:03d} | years {years[0]:+07d} to {years[-1]:+07d} | rows={len(frame)}")
            return frame
        except Exception as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"DEBUG retry {attempt}/{MAX_RETRIES} | chunk {index}/{total} | HTTP {status if status else 'N/A'} {type(exc).__name__}: {str(exc)[:180]}")
            if attempt < MAX_RETRIES:
                time.sleep(5.0 * attempt)
    raise RuntimeError(f"REJECTED JPL query failed for chunk {index}/{total}: {last}")


def derive_normals(states: pd.DataFrame) -> pd.DataFrame:
    r = states[["x_au", "y_au", "z_au"]].to_numpy(float)
    v = states[["vx_au_per_day", "vy_au_per_day", "vz_au_per_day"]].to_numpy(float)
    h = np.cross(r, v)
    hm = np.linalg.norm(h, axis=1)
    if np.any(~np.isfinite(hm)) or np.any(hm <= 0.0):
        raise RuntimeError("REJECTED invalid angular-momentum vectors")
    n = h / hm[:, None]
    return pd.DataFrame({
        "sample_index": states["sample_index"],
        "astronomical_year": states["astronomical_year"],
        "jd_tdb": states["jd_tdb_returned"],
        "normal_x": n[:, 0], "normal_y": n[:, 1], "normal_z": n[:, 2],
        "h_magnitude_au2_per_day": hm,
        "inclination_deg_ecliptic": np.degrees(np.arccos(np.clip(n[:, 2], -1.0, 1.0))),
        "ascending_node_longitude_deg": np.mod(np.degrees(np.arctan2(n[:, 0], -n[:, 1])), 360.0),
    })


def validate(states: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    expected_years = np.arange(START_YEAR, STOP_YEAR + STEP_YEARS, STEP_YEARS, dtype=int)
    years = states["astronomical_year"].to_numpy(dtype=int)
    n = normals[["normal_x", "normal_y", "normal_z"]].to_numpy(float)
    state_values = states[["x_au", "y_au", "z_au", "vx_au_per_day", "vy_au_per_day", "vz_au_per_day"]].to_numpy(float)
    epoch_error = float(np.max(np.abs(states["jd_tdb_returned"] - states["jd_tdb_requested"])))
    normal_error = float(np.max(np.abs(np.linalg.norm(n, axis=1) - 1.0)))
    checks = {
        "requested_rows": EXPECTED_ROWS,
        "returned_rows": len(states),
        "unique_years": int(states["astronomical_year"].nunique()),
        "unique_returned_jd": int(states["jd_tdb_returned"].nunique()),
        "missing_rows": EXPECTED_ROWS - len(states),
        "duplicate_rows": int(states["jd_tdb_returned"].duplicated().sum()),
        "maximum_epoch_error_days": epoch_error,
        "maximum_unit_normal_error": normal_error,
        "finite_state_values": bool(np.isfinite(state_values).all()),
        "finite_normal_values": bool(np.isfinite(n).all()),
    }
    if len(states) != EXPECTED_ROWS or not np.array_equal(years, expected_years):
        raise RuntimeError("REJECTED incomplete or incorrect 6000-epoch sequence")
    if checks["duplicate_rows"] != 0 or checks["unique_returned_jd"] != EXPECTED_ROWS:
        raise RuntimeError("REJECTED duplicate JPL epochs")
    if not checks["finite_state_values"] or not checks["finite_normal_values"]:
        raise RuntimeError("REJECTED nonfinite values")
    if epoch_error > 2.0e-7 or normal_error > 2.0e-12:
        raise RuntimeError("REJECTED epoch or unit-normal closure tolerance")
    checks["acquisition_status"] = "ACCEPTED"
    return pd.DataFrame({"quantity": list(checks), "value": list(checks.values())})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    years = np.arange(START_YEAR, STOP_YEAR + STEP_YEARS, STEP_YEARS, dtype=int)
    if len(years) != EXPECTED_ROWS:
        raise RuntimeError(f"REJECTED generated {len(years)} epochs")
    jds = np.asarray([gregorian_jd(int(y)) for y in years], dtype=float)
    total = math.ceil(EXPECTED_ROWS / CHUNK_SIZE)
    print(f"OUTPUT VERSION {VERSION}")
    print("CODE INPUTS")
    print("Notebook abbreviation : EMBOPSA")
    print("JPL target            : 3 | Earth-Moon barycenter")
    print("JPL center            : @10 | Sun center")
    print("Reference plane       : ecliptic")
    print("Reference system      : ICRF")
    print("Aberrations           : geometric")
    print("Time scale            : TDB")
    print("Epoch list type       : JD")
    print(f"Year range            : {START_YEAR:+d} to {STOP_YEAR:+d}")
    print(f"Cadence               : {STEP_YEARS} years")
    print(f"Requested samples     : {EXPECTED_ROWS}")
    print(f"Query chunks          : {total} x up to {CHUNK_SIZE} epochs")
    print("COMMENTS")
    print("Astronomical year 0 equals 1 BCE.")
    print("Successful chunks are cached and validated before reuse.")
    print("FFT and harmonic fitting are blocked until all 6,000 states pass validation.")
    session = requests.Session()
    session.headers["User-Agent"] = "EMBOPSA-V0001A/1.0"
    frames = []
    try:
        for start in range(0, EXPECTED_ROWS, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, EXPECTED_ROWS)
            frames.append(acquire_chunk(session, start // CHUNK_SIZE + 1, years[start:stop], jds[start:stop], total))
    except Exception:
        print("ACQUISITION STATUS    : REJECTED")
        raise
    states = pd.concat(frames, ignore_index=True).sort_values("astronomical_year").reset_index(drop=True)
    states.insert(0, "sample_index", np.arange(len(states), dtype=int))
    normals = derive_normals(states)
    audit = validate(states, normals)
    states.to_csv(STATE_CSV, index=False, float_format="%.15f")
    normals.to_csv(NORMAL_CSV, index=False, float_format="%.15f")
    audit.to_csv(AUDIT_CSV, index=False)
    print("RESULTS")
    print(f"VALID JPL STATES      : {len(states)}")
    print(f"UNIQUE JULIAN DATES   : {states['jd_tdb_returned'].nunique()}")
    print(f"MISSING STATES        : {EXPECTED_ROWS - len(states)}")
    print(f"DUPLICATE STATES      : {states['jd_tdb_returned'].duplicated().sum()}")
    print("REJECTED STATES       : 0")
    print("ACQUISITION STATUS    : ACCEPTED")
    print("OUTPUT SUMMARY")
    print(STATE_CSV)
    print(NORMAL_CSV)
    print(AUDIT_CSV)
    print("EQUATION STATUS")
    print("VERIFIED orbital normal: n = (r x v) / |r x v|.")
    print("VERIFIED cadence, uniqueness, finite values, epoch agreement, and unit-normal closure.")
    print(datetime.now(LOCAL_TZ).strftime("LOCAL TIMESTAMP %Y-%m-%d %H:%M:%S %Z"))
    print(f"FINAL VERSION {VERSION}")


if __name__ == "__main__":
    main()
# V0001A
