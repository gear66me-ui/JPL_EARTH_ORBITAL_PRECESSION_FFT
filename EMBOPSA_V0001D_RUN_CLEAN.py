# V0001D
# Clean runner: executes V0001D while suppressing stale Matplotlib redraw failures
from __future__ import annotations

import time
import requests
import matplotlib.pyplot as plt

SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "JPL_EARTH_ORBITAL_PRECESSION_FFT/main/"
    "EMBOPSA_V0001D.py"
)

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

# Repair legacy MathText syntax before execution.
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

# Disable the equation figure entirely in this clean runner.
source = source.replace(
    "fig,ax=plt.subplots(figsize=(11,6.5)); ax.axis('off'); eqs=",
    "fig,ax=plt.subplots(figsize=(11,6.5)); ax.axis('off'); plt.close(fig); eqs=",
)
source = source.replace(
    "fig.tight_layout(); fig.savefig(FILES['equation']); plt.close(fig)",
    "plt.close(fig)",
)
source = source.replace(
    "fig.tight_layout()\n    fig.savefig(EQUATION_PNG, bbox_inches=\"tight\")\n    plt.close(fig)",
    "plt.close(fig)",
)

try:
    compiled = compile(source, "EMBOPSA_V0001D.py", "exec")
    exec(compiled, globals(), globals())
finally:
    plt.close("all")
    plt.ioff()

print("RUNNER STATUS         : CLEAN EXIT")
# V0001D
