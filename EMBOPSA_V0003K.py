# EMBOPSA_V0003K
# Corrected launcher for V0003F embedded galaxy background.
# Repairs the exact Base64 decode expression; no file upload is requested.

from __future__ import annotations

import requests

VERSION = "V0003K"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/"
    "4b7595766609288fc524535b6ca63f2ad0e0d6eb/EMBOPSA_V0003F.py"
)

response = requests.get(SOURCE_URL, timeout=300)
response.raise_for_status()
source = response.text

old = "base64.b64decode(GALAXY_JPEG_B64)"
new = "base64.b64decode(GALAXY_JPEG_B64 + '=' * (-len(GALAXY_JPEG_B64) % 4), validate=False)"

if old not in source:
    raise RuntimeError("Exact Base64 decode expression was not found in V0003F.")

source = source.replace("# EMBOPSA_V0003F", "# EMBOPSA_V0003K", 1)
source = source.replace('VERSION = "V0003F"', 'VERSION = "V0003K"', 1)
source = source.replace(old, new, 1)
source = source.replace("EMBOPSA_V0003F_GENERATED.py", "EMBOPSA_V0003K_GENERATED.py", 1)

exec(compile(source, "EMBOPSA_V0003K_PATCHED.py", "exec"), globals(), globals())
