import subprocess, os, sys
from pathlib import Path

segments = [32]     # fixed for robustness study
order = 3
guess_ids = [0, 1, 2, 3, 4]

project_root = Path(__file__).resolve().parents[1]
dymos_script = project_root / "dymos" / "dymos_rocket_opt.py"

for guess in guess_ids:
    print('-'*60)
    print(f"Running Dymos with guess {guess}")
    print('-'*60)

    env = os.environ.copy()
    env["DYMOS_SEG"] = "32"
    env["DYMOS_ORDER"] = str(order)
    env["DYMOS_GUESS_ID"] = str(guess)

    subprocess.run(
        ["python", str(dymos_script)],
        env=env,
        check=True,
    )

print("\nDymos initial guess robustness sweep complete!")
