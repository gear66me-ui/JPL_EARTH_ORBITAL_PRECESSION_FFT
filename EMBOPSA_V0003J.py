# EMBOPSA_V0003J
# Corrected launcher for V0003F embedded galaxy background.
# Repairs Base64 padding before decoding; no file upload is requested.

from __future__ import annotations

import requests

VERSION = "V0003J"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "4b7595766609288fc524535b6ca63f2ad0e0d6eb/EMBOPSA_V0003F.py"
)

response = requests.get(SOURCE_URL, timeout=300)
response.raise_for_status()
source = response.text

old = "_galaxy_bytes = base64.b64decode(GALAXY_JPEG_B64)"
new = (
    "_galaxy_b64 = ''.join(GALAXY_JPEG_B64.split())\n"
    "_galaxy_b64 += '=' * (-len(_galaxy_b64) % 4)\n"
    "_galaxy_bytes = base64.b64decode(_galaxy_b64, validate=False)"
)

if old not in source:
    raise RuntimeError("Base64 decode line was not found in V0003F.")

source = source.replace("# EMBOPSA_V0003F", "# EMBOPSA_V0003J", 1)
source = source.replace('VERSION = "V0003F"', 'VERSION = "V0003J"', 1)
source = source.replace(old, new, 1)
source = source.replace("EMBOPSA_V0003F_GENERATED.py", "EMBOPSA_V0003J_GENERATED.py", 1)

exec(compile(source, "EMBOPSA_V0003J_PATCHED.py", "exec"), globals(), globals())
