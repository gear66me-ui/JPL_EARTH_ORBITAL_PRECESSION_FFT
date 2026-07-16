# V0001C
# Audit reference: public-Horizons range-corrected EMB acquisition
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

VERSION = "V0001C"
TARGET = "3"
CENTER = "@10"
START_YEAR = -9995
STOP_YEAR = 16995
STEP_YEARS = 5
EXPECTED_ROWS = 5399
ORIGINAL_START_YEAR = -13000
ORIGINAL_EXPECTED_ROWS = 6000
CHUNK_SIZE = 25
MAX_RETRIES = 8
TIMEOUT_SECONDS = 180
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUT = Path("/content/JPL_EARTH_ORBITAL_PRECESSION_FFT/EMBOPSA_V0001C_OUTPUT")
CACHE = OUT / "cache_jpl_chunks"
STATE_CSV = OUT / "EMBOPSA_JPL_EMB_STATES_5399_V0001C.csv"
NORMAL_CSV = OUT / "EMBOPSA_ORBITAL_NORMAL_5399_V0001C.csv"
AUDIT_CSV = OUT / "EMBOPSA_ACQUISITION_AUDIT_V0001C.csv"
LOCAL_TZ = ZoneInfo("America/Bogota")


def gregorian_jd(year: int, month: int = 1, day: float = 1.0) -> float:
    y, m = int(year), int(month)
    if m <= 2:
        y -= 1
        m += 12
    a = math.floor(y / 100.0)
    b = 2 - a + math.floor(a / 4.0)
    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + day + b - 1524.5
    )


def parameters(jds: np.ndarray) -> dict[str, str]:
    return {
        "format": "text",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "OUT_UNITS": "AU-D",
        "COMMAND": "'3'",
        "CENTER": "'@10'",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "ICRF",
        "VEC_TABLE": "2",
        "VEC_CORR": "NONE",
        "VEC_DELTA_T": "NO",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "YES",
        "OBJ_DATA": "NO",
        "TIME_TYPE": "TDB",
        "TLIST_TYPE": "JD",
        "TLIST": " ".join(f"'{jd:.9f}'" for jd in jds),
    }


def parse_vectors(
    text: str,
    years: np.ndarray,
    requested_jd: np.ndarray,
    chunk_index: int,
    chunk_total: int,
) -> pd.DataFrame:
    if "$$SOE" not in text or "$$EOE" not in text:
        response = " ".join(text.split())[:800]
        raise RuntimeError(
            f"REJECTED chunk {chunk_index}/{chunk_total}: "
            f"Horizons data markers absent; response={response}"
        )

    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(body), skipinitialspace=True)
        if row and any(field.strip() for field in row)
    ]
    if len(rows) != len(requested_jd):
        raise RuntimeError(
            f"REJECTED chunk {chunk_index}/{chunk_total}: "
            f"requested {len(requested_jd)}, received {len(rows)}"
        )

    values = []
    for row in rows:
        if len(row) < 8:
            raise RuntimeError(
                f"REJECTED malformed Horizons row with {len(row)} fields"
            )
        values.append([
            float(row[0]), float(row[2]), float(row[3]), float(row[4]),
            float(row[5]), float(row[6]), float(row[7]),
        ])

    array = np.asarray(values, dtype=float)
    order = np.argsort(array[:, 0])
    array = array[order]
    requested_jd = np.asarray(requested_jd, dtype=float)
    req_order = np.argsort(requested_jd)
    requested_jd = requested_jd[req_order]
    years = np.asarray(years, dtype=int)[req_order]

    if not np.allclose(array[:, 0], requested_jd, rtol=0.0, atol=2.0e-7):
        raise RuntimeError("REJECTED JPL returned epochs do not match requests")

    return pd.DataFrame({
        "astronomical_year": years,
        "jd_tdb_requested": requested_jd,
        "jd_tdb_returned": array[:, 0],
        "x_au": array[:, 1],
        "y_au": array[:, 2],
        "z_au": array[:, 3],
        "vx_au_per_day": array[:, 4],
        "vy_au_per_day": array[:, 5],
        "vz_au_per_day": array[:, 6],
    })


def acquire_chunk(
    session: requests.Session,
    index: int,
    total: int,
    years: np.ndarray,
    jds: np.ndarray,
) -> pd.DataFrame:
    cache_path = CACHE / f"EMBOPSA_V0001C_CHUNK_{index:03d}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        valid = (
            len(cached) == len(jds)
            and np.array_equal(
                cached["astronomical_year"].to_numpy(dtype=int), years
            )
            and np.allclose(
                cached["jd_tdb_requested"].to_numpy(dtype=float),
                jds, rtol=0.0, atol=5.0e-10,
            )
        )
        if valid:
            print(f"CACHE CHUNK {index:03d}/{total:03d} | rows={len(cached)}")
            return cached
        print(f"REJECTED cached chunk {index:03d}/{total:03d}; reacquiring")

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params=parameters(jds),
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            frame = parse_vectors(
                response.text, years, jds, index, total
            )
            frame.to_csv(cache_path, index=False, float_format="%.15f")
            print(
                f"JPL CHUNK {index:03d}/{total:03d} | "
                f"years {years[0]:+07d} to {years[-1]:+07d} | "
                f"rows={len(frame)}"
            )
            return frame
        except Exception as exc:
            last_error = exc
            message = str(exc)
            print(
                f"DEBUG retry {attempt}/{MAX_RETRIES} | "
                f"chunk {index}/{total} | "
                f"{type(exc).__name__}: {message[:600]}"
            )
            if "No ephemeris for target" in message:
                raise RuntimeError(
                    "REJECTED deterministic Horizons range limit; "
                    "further retries NOT USED"
                ) from exc
            if attempt < MAX_RETRIES:
                time.sleep(5.0 * attempt)

    raise RuntimeError(
        f"REJECTED chunk {index}/{total} after retries: {last_error}"
    )


def derive_normals(states: pd.DataFrame) -> pd.DataFrame:
    r = states[["x_au", "y_au", "z_au"]].to_numpy(dtype=float)
    v = states[
        ["vx_au_per_day", "vy_au_per_day", "vz_au_per_day"]
    ].to_numpy(dtype=float)
    h = np.cross(r, v)
    hmag = np.linalg.norm(h, axis=1)
    if np.any(~np.isfinite(hmag)) or np.any(hmag <= 0.0):
        raise RuntimeError("REJECTED invalid angular-momentum vectors")
    n = h / hmag[:, None]
    return pd.DataFrame({
        "sample_index": states["sample_index"].to_numpy(dtype=int),
        "astronomical_year": states["astronomical_year"].to_numpy(dtype=int),
        "jd_tdb": states["jd_tdb_returned"].to_numpy(dtype=float),
        "normal_x": n[:, 0],
        "normal_y": n[:, 1],
        "normal_z": n[:, 2],
        "h_magnitude_au2_per_day": hmag,
        "inclination_deg_ecliptic": np.degrees(
            np.arccos(np.clip(n[:, 2], -1.0, 1.0))
        ),
        "ascending_node_longitude_deg": np.mod(
            np.degrees(np.arctan2(n[:, 0], -n[:, 1])), 360.0
        ),
    })


def validate(states: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    expected_years = np.arange(
        START_YEAR, STOP_YEAR + STEP_YEARS, STEP_YEARS, dtype=int
    )
    years = states["astronomical_year"].to_numpy(dtype=int)
    state_values = states[[
        "x_au", "y_au", "z_au",
        "vx_au_per_day", "vy_au_per_day", "vz_au_per_day",
    ]].to_numpy(dtype=float)
    normal_values = normals[
        ["normal_x", "normal_y", "normal_z"]
    ].to_numpy(dtype=float)

    epoch_error = float(np.max(np.abs(
        states["jd_tdb_returned"].to_numpy(dtype=float)
        - states["jd_tdb_requested"].to_numpy(dtype=float)
    )))
    normal_error = float(np.max(np.abs(
        np.linalg.norm(normal_values, axis=1) - 1.0
    )))
    duplicate_rows = int(
        states["jd_tdb_returned"].duplicated().sum()
    )

    if len(states) != EXPECTED_ROWS:
        raise RuntimeError(
            f"REJECTED expected {EXPECTED_ROWS}, obtained {len(states)}"
        )
    if not np.array_equal(years, expected_years):
        raise RuntimeError("REJECTED incorrect epoch sequence")
    if duplicate_rows != 0:
        raise RuntimeError("REJECTED duplicate epochs")
    if not np.isfinite(state_values).all():
        raise RuntimeError("REJECTED nonfinite state values")
    if not np.isfinite(normal_values).all():
        raise RuntimeError("REJECTED nonfinite orbital normals")
    if epoch_error > 2.0e-7:
        raise RuntimeError("REJECTED JPL epoch mismatch")
    if normal_error > 2.0e-12:
        raise RuntimeError("REJECTED unit-normal closure")

    checks = {
        "original_requested_rows": ORIGINAL_EXPECTED_ROWS,
        "original_start_year": ORIGINAL_START_YEAR,
        "original_design_status": "REJECTED_HORIZONS_RANGE",
        "accessible_requested_rows": EXPECTED_ROWS,
        "returned_rows": len(states),
        "unique_years": int(states["astronomical_year"].nunique()),
        "unique_returned_jd": int(states["jd_tdb_returned"].nunique()),
        "missing_rows": EXPECTED_ROWS - len(states),
        "duplicate_rows": duplicate_rows,
        "maximum_epoch_error_days": epoch_error,
        "maximum_unit_normal_error": normal_error,
        "acquisition_status": "ACCEPTED",
    }
    return pd.DataFrame({
        "quantity": list(checks.keys()),
        "value": list(checks.values()),
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    years = np.arange(
        START_YEAR, STOP_YEAR + STEP_YEARS, STEP_YEARS, dtype=int
    )
    if len(years) != EXPECTED_ROWS:
        raise RuntimeError(
            f"REJECTED generated {len(years)} epochs, "
            f"expected {EXPECTED_ROWS}"
        )
    jds = np.asarray(
        [gregorian_jd(int(year)) for year in years], dtype=float
    )
    total_chunks = math.ceil(EXPECTED_ROWS / CHUNK_SIZE)

    print(f"OUTPUT VERSION {VERSION}")
    print("CODE INPUTS")
    print("Notebook abbreviation : EMBOPSA")
    print("JPL target            : 3 | Earth-Moon barycenter")
    print("JPL center            : @10 | Sun center")
    print("Reference plane       : ecliptic")
    print("Reference system      : ICRF")
    print("Aberrations           : geometric")
    print("Time scale            : TDB")
    print("Epoch list type       : JD | individually quoted")
    print(f"Year range            : {START_YEAR:+d} to {STOP_YEAR:+d}")
    print(f"Cadence               : {STEP_YEARS} years")
    print(f"Requested samples     : {EXPECTED_ROWS}")
    print(
        f"Original design       : {ORIGINAL_EXPECTED_ROWS} states | "
        f"start {ORIGINAL_START_YEAR:+d} | REJECTED"
    )
    print(
        f"Query chunks          : {total_chunks} "
        f"x up to {CHUNK_SIZE} epochs"
    )
    print("COMMENTS")
    print("Astronomical year 0 equals 1 BCE.")
    print(
        "REJECTED original start -13000: public Horizons target 3 "
        "reports no ephemeris before B.C. 9999-MAR-21 TDB."
    )
    print(
        "ACCEPTED public-Horizons interval: years -9995 through "
        "+16995 at exact five-year cadence."
    )
    print("Successful chunks are cached and validated before reuse.")
    print(
        "FFT and harmonic fitting are blocked until all 5,399 "
        "accessible states pass validation."
    )

    session = requests.Session()
    session.headers["User-Agent"] = "EMBOPSA-V0001C/1.0"

    frames = []
    try:
        for start in range(0, EXPECTED_ROWS, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, EXPECTED_ROWS)
            frames.append(acquire_chunk(
                session,
                start // CHUNK_SIZE + 1,
                total_chunks,
                years[start:stop],
                jds[start:stop],
            ))
    except Exception:
        print("ACQUISITION STATUS    : REJECTED")
        raise

    states = pd.concat(frames, ignore_index=True)
    states = states.sort_values(
        ["astronomical_year", "jd_tdb_returned"]
    ).reset_index(drop=True)
    states.insert(0, "sample_index", np.arange(len(states), dtype=int))

    normals = derive_normals(states)
    audit = validate(states, normals)

    states.to_csv(STATE_CSV, index=False, float_format="%.15f")
    normals.to_csv(NORMAL_CSV, index=False, float_format="%.15f")
    audit.to_csv(AUDIT_CSV, index=False)

    print("RESULTS")
    print(f"VALID JPL STATES      : {len(states)}")
    print(
        f"UNIQUE JULIAN DATES   : "
        f"{states['jd_tdb_returned'].nunique()}"
    )
    print(f"MISSING STATES        : {EXPECTED_ROWS - len(states)}")
    print(
        f"DUPLICATE STATES      : "
        f"{states['jd_tdb_returned'].duplicated().sum()}"
    )
    print("REJECTED STATES       : 0")
    print("ACQUISITION STATUS    : ACCEPTED")
    print("OUTPUT SUMMARY")
    print(STATE_CSV)
    print(NORMAL_CSV)
    print(AUDIT_CSV)
    print("EQUATION STATUS")
    print("VERIFIED orbital normal: n = (r x v) / |r x v|.")
    print(
        "VERIFIED cadence, uniqueness, finite values, "
        "epoch agreement, and unit-normal closure."
    )
    print(datetime.now(LOCAL_TZ).strftime(
        "LOCAL TIMESTAMP %Y-%m-%d %H:%M:%S %Z"
    ))
    print(f"FINAL VERSION {VERSION}")


if __name__ == "__main__":
    main()
# V0001C
