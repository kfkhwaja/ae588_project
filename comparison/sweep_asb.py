# do a sweep of mesh sizes for asb
import sys
import subprocess
import os
from pathlib import Path

# Mesh sizes to test
Ns = [10, 16, 32, 64, 128, 256, 512, 1024, 2048]

project_root = Path(__file__).resolve().parents[1]
asb_script = project_root / "aerosandbox" / "asb_rocket_opt.py"

env = os.environ.copy()

for N in Ns:
    print('-'*60)
    print(f"Running AeroSandbox with N = {N}...")
    print('-'*60)

    env = os.environ.copy()
    env["ASB_N"] = str(N)

    subprocess.run(
        [sys.executable, asb_script],
        env=env,
        check=True,
    )

print("\nAeroSandbox mesh sweep complete!")
