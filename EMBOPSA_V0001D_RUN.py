# V0001D
# Cache-safe runner for EMBOPSA_V0001D
from __future__ import annotations

import time
import requests
import matplotlib.pyplot as plt

SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "EMBOPSA_V0001D.py"
)

# Clear any stale failed Matplotlib figures before execution.
plt.ioff()
plt.close("all")

response = requests.get(
    SOURCE_URL,
    params={"cachebust": str(time.time_ns())},
    headers={"Cache-Control": "no-cache"},
    timeout=180,
)
response.raise_for_status()
source = response.text

# Repair every legacy unbraced Matplotlib MathText vector command.
repairs = {
    r"\mathbf h": r"\mathbf{h}",
    r"\mathbf r": r"\mathbf{r}",
    r"\mathbf v": r"\mathbf{v}",
    r"\mathbf e_p": r"\mathbf{e}_{p}",
    r"\mathbf e_q": r"\mathbf{e}_{q}",
    r"\bar{\mathbf h}": r"\overline{\mathbf{h}}",
    r"\widehat z": r"\widehat{z}",
}
for old, new in repairs.items():
    source = source.replace(old, new)

if r"\mathbf r" in source or r"\mathbf v" in source:
    raise RuntimeError("REJECTED legacy MathText syntax remains after repair")

compiled = compile(source, "EMBOPSA_V0001D.py", "exec")
try:
    exec(compiled, globals(), globals())
finally:
    # Prevent IPython from redrawing a stale failed figure after FINAL VERSION.
    plt.close("all")

print("RUNNER STATUS         : CLEAN EXIT")
# V0001D
