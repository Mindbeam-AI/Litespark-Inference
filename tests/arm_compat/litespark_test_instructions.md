  # LiteSpark validation instructions

Please run the attached LiteSpark benchmark locally (via Terminal) so we can compare LiteSpark against a standard Transformers baseline on the same prompt and model family.

## Files

- `benchmark_litespark.py`
- `benchmark_transformers.py`
- `benchmark_repeat_v2.py`

## What the scripts do

### `benchmark_litespark.py`
Runs:

```bash
litespark-inference generate --model bitnet-2b --max-tokens 32 "Hello, how are you?"
```

It parses the throughput line in the output:

```text
Generated X tokens in Ys (Z tok/s)
```

### `benchmark_transformers.py`
Runs the Hugging Face baseline with:

- model: `microsoft/bitnet-b1.58-2B-4T-bf16`
- prompt: `Hello, how are you?`
- warmup runs before measurement

### `benchmark_repeat_v2.py`
Runs both backends 5 times and captures:

- generated tokens
- elapsed time
- tokens/sec
- peak RSS memory
- median and aggregate summaries
- CSV + log output

## Environment setup

1. Make sure Python is installed.
2. Install Python dependencies for the Transformers baseline:

   ```bash
   pip install torch transformers
   ```

3. Make sure `litespark-inference` is installed and available on `PATH`.
4. Put all three scripts in the same directory.

## Run individual checks first

### 1. Validate LiteSpark

```bash
python benchmark_litespark.py
```

Expected outcome:

- command completes successfully
- output includes a line like:

  ```text
  Generated <n> tokens in <s>s (<tok/s> tok/s)
  ```

- script prints:

  ```text
  Parsed metrics: ...
  ```

### 2. Validate Transformers baseline

```bash
python benchmark_transformers.py
```

Expected outcome:

- model loads successfully
- generation completes
- output includes throughput and generated text

## Run the repeated benchmark

```bash
python benchmark_repeat_v2.py
```

This will:

- execute 5 runs for `transformers`
- execute 5 runs for `litespark`
- save outputs under:
  - `benchmark_logs/benchmark_repeat_<timestamp>.log`
  - `benchmark_logs/benchmark_repeat_<timestamp>.csv`

## What to share back

Please send back:

- terminal output from the full benchmark run
- the generated `.log` file
- the generated `.csv` file
- confirmation of:
  - OS
  - CPU
  - RAM
  - Python version
  - LiteSpark version / commit
  - whether any model weights were pre-cached

## Important note

The repeat script currently uses:

```bash
/usr/bin/time -l
```

That works on macOS, but on many Linux systems `-l` is not supported. If you are running on Linux, replace those commands in `benchmark_repeat_v2.py` with:

```python
COMMANDS = {
    "transformers": ["bash", "-lc", "/usr/bin/time -v python benchmark_transformers.py 2>&1"],
    "litespark": ["bash", "-lc", "/usr/bin/time -v python benchmark_litespark.py 2>&1"],
}
```

If you do that, the RSS parsing may also need updating, because Linux `time -v` reports memory differently than macOS.

