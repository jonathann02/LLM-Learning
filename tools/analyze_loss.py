import re
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

LOG_PATH = Path("runs/00-baseline/output.txt")

pattern = re.compile(r"step\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\s+([0-9.]+)")

steps = []
losses = []

def read_log_text(path: Path) -> str:
    raw = path.read_bytes()

    # Common on Windows when using `>` redirection in PowerShell.
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


text = read_log_text(LOG_PATH)
for line in text.splitlines():
    m = pattern.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(3)))

if not losses:
    raise SystemExit("No loss lines found. Check the log path or format.")

n = len(losses)
k = max(10, n // 10)  # 10% of run (at least 10 points)

start_avg = mean(losses[:k])
end_avg = mean(losses[-k:])
overall_avg = mean(losses)

# simple linear regression slope (loss vs step) without numpy
x_mean = mean(steps)
y_mean = overall_avg
num = sum((x - x_mean) * (y - y_mean) for x, y in zip(steps, losses))
den = sum((x - x_mean) ** 2 for x in steps)
slope = num / den if den != 0 else float("nan")

print(f"points: {n}")
print(f"avg(first {k}): {start_avg:.4f}")
print(f"avg(last  {k}): {end_avg:.4f}")
print(f"delta (last-first): {(end_avg - start_avg):.4f}")
print(f"overall avg: {overall_avg:.4f}")
print(f"min/max: {min(losses):.4f} / {max(losses):.4f}")
print(f"slope per step: {slope:.8f}  (negative = improving)")

# moving average for a cleaner picture
window = 50 if n >= 200 else max(5, n // 10)
ma = []
ma_x = []
running = 0.0

for i, y in enumerate(losses):
    running += y
    if i >= window:
        running -= losses[i - window]
    if i >= window - 1:
        ma.append(running / window)
        ma_x.append(steps[i])

plt.figure()
plt.plot(steps, losses)
plt.plot(ma_x, ma)
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"Training loss (window={window})")
plt.tight_layout()

out_path = Path("runs/00-baseline/loss.png")
plt.savefig(out_path)
print(f"saved plot: {out_path}")
