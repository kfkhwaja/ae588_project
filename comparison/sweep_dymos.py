# sweep dymos mesh sizes
import subprocess
import os
from pathlib import Path

# Segment counts to test
segments = [5, 8, 16, 32, 64, 128]
order = 3

project_root = Path(__file__).resolve().parents[1]
dymos_script = project_root / "dymos" / "dymos_rocket_opt.py"

env = os.environ.copy()
env["DYMOS_ORDER"] = str(order)

for seg in segments:
    print('-'*60)
    print(f"Running Dymos with num_segments = {seg}, order = {order}...")
    print('-'*60)
    env["DYMOS_SEG"] = str(seg)

    subprocess.run(
        ["python", str(dymos_script)],
        env=env,
        check=True,
    )

print("\nDymos mesh sweep complete!")
