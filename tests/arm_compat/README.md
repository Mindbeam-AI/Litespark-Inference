# ARM compatibility tests

These four files come from ARM's validation kit for Litespark
(`litespark-test_run1.zip`, original instructions in
[`litespark_test_instructions.md`](./litespark_test_instructions.md)).
They have been vendored into this repo so a fresh `git clone` + `pip
install` brings everything needed to reproduce ARM's measurements.

| File | Origin | Purpose |
|---|---|---|
| `benchmark_litespark.py` | ARM zip, **unmodified** | shells out to `litespark-inference generate` (which auto-routes to torchless for `bitnet-2b`) and parses the throughput line |
| `benchmark_transformers.py` | ARM zip, **unmodified** | HuggingFace `AutoModelForCausalLM` bf16 baseline |
| `benchmark_repeat_v2.py` | ARM zip, **portability rewrite** | runs each backend 5x under `/usr/bin/time` and aggregates |
| `litespark_test_instructions.md` | ARM zip, **unmodified** | original README from ARM |

## What changed in `benchmark_repeat_v2.py`

The original used `bash -lc "..."` which strips the parent's PATH and
hard-coded `/usr/bin/time -l` (macOS only). The portable version:

- invokes subprocesses via `sys.executable` so the same Python interpreter
  is used and PATH is not reset
- picks `/usr/bin/time -l` on Darwin, `/usr/bin/time -v` on Linux
- accepts both Darwin's "maximum resident set size" (bytes) and GNU
  time's "Maximum resident set size (kbytes)" RSS lines

Aggregation, runs, prompts, and output format are all unchanged.

## Running

From the repo root, after `pip install -e .`:

```bash
# Quick validations of each backend individually:
python tests/arm_compat/benchmark_litespark.py
python tests/arm_compat/benchmark_transformers.py

# Full 5x repeat with RSS capture (writes .log and .csv to ./benchmark_logs/):
python tests/arm_compat/benchmark_repeat_v2.py
```

No environment variables required. No edits required.

## Expected results on Apple Silicon (M-series), bitnet-2b

```
=== Median summary ===
transformers tokens=21 time=~43s   tok/s= 0.50  rss=~5.0 GB
litespark    tokens=32 time=~1.0s  tok/s=~30    rss=~1.2 GB

=== Aggregate comparison ===
Litespark speedup vs transformers:           ~60x
Litespark memory reduction vs transformers:  ~4x
```

The "litespark" row goes through `litespark-inference generate ...`,
which on this branch dispatches to the torchless runtime for
`bitnet-2b` automatically. To force the legacy torch-backed path for
the `litespark` row instead, set `LITESPARK_FORCE_TORCH=1` in your
shell before running.
