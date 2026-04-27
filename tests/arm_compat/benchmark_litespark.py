import subprocess
import re
import sys

PROMPT = "Hello, how are you?"
MODEL_NAME = "bitnet-2b"
MAX_TOKENS = 32

cmd = [
    "litespark-inference",
    "generate",
    "--model",
    MODEL_NAME,
    "--max-tokens",
    str(MAX_TOKENS),
    PROMPT,
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.stdout:
    print(result.stdout, end="")

if result.stderr:
    print(result.stderr, end="", file=sys.stderr)

if result.returncode != 0:
    raise SystemExit(result.returncode)

match = re.search(
    r"Generated\s+(\d+)\s+tokens\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+tok/s\)",
    result.stdout,
)

if not match:
    raise RuntimeError("Could not parse Litespark throughput output")

tokens = int(match.group(1))
seconds = float(match.group(2))
tok_s = float(match.group(3))

print(f"\nParsed metrics: {tokens} tokens, {seconds:.2f}s, {tok_s:.1f} tok/s")
