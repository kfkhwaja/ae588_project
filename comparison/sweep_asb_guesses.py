# sweep initial guesses
import subprocess, os, sys
from pathlib import Path

N = 69
GUESS_IDS = range(5)

script = Path(__file__).resolve().parents[1] / "aerosandbox" / "asb_rocket_opt.py"

for gid in GUESS_IDS:
    env = os.environ.copy()
    env["ASB_N"] = str(N)
    env["ASB_GUESS_ID"] = str(gid)

    print(f"Running ASB N={N}, guess={gid}")
    subprocess.run([sys.executable, script], env=env, check=True)
