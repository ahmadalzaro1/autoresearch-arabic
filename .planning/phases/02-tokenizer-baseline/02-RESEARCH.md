# Phase 2: Tokenizer & Baseline - Research

**Researched:** 2026-03-12
**Domain:** BPE tokenizer training (rustbpe + tiktoken), fertility measurement, MLX baseline training via existing prepare.py + train.py
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TOK-01 | Train BPE tokenizer for D1 condition | `prepare.py::train_tokenizer()` already implements this via `rustbpe.Tokenizer` — gaps: VOCAB_SIZE is hardcoded (needs CLI flag), fertility only prints (needs structured JSON output) |
| TOK-02 | Train BPE tokenizer for D2 condition | Same as TOK-01, different condition. SPLIT_PATTERN already handles Arabic + PUA ranges. |
| TOK-03 | Train BPE tokenizer for D3 condition | Same as TOK-01; D3 PUA codepoints (U+E000–U+EFFF) are already included in SPLIT_PATTERN — BPE will treat each PUA codepoint as an atomic token boundary unit |
| TOK-04 | Measure tokenizer fertility (tokens/word) per condition x vocab size | `train_tokenizer()` already computes fertility on a ~1M char sample and prints it; needs to be (a) exposed per vocab size and (b) written to a structured `fertility_report.json` |
| BASE-01 | Run baseline val_bpb training for D1 | `train.py` runs a full 300s DEPTH=4 training run and prints `val_bpb`; the output is stdout-only — needs to be captured into `baseline_results.json` |
| BASE-02 | Run baseline val_bpb training for D2 | Same as BASE-01 with `AUTORESEARCH_CONDITION=d2` |
| BASE-03 | Run baseline val_bpb training for D3 | Same as BASE-01 with `AUTORESEARCH_CONDITION=d3` |
</phase_requirements>

---

## Summary

Phase 2 has the core machinery already built: `prepare.py` trains BPE tokenizers using `rustbpe` + `tiktoken`, computes fertility, and writes `tokenizer.pkl` + `token_bytes.npy`. `train.py` runs a full 300-second MLX training run and evaluates `val_bpb` using `evaluate_bpb()`. Both scripts are working implementations from Phase 1 infrastructure.

The gap between current code and Phase 2 completion is narrow but requires systematic changes. First, `VOCAB_SIZE = 8192` is a module-level constant in `prepare.py` — it must become a `--vocab-size` CLI argument so multiple vocab sizes can be trained without modifying source. Second, fertility results are printed to stdout but not persisted — a `fertility_report.json` must be written per condition × vocab size. Third, `train.py` outputs `val_bpb` to stdout — a `baseline_results.json` must capture this per condition for Phase 3 to gate on. These are three additive changes to existing scripts.

The data is ready: all three conditions have 30 shards each (29 train + 1 val), 1,434,471 train docs and 29,274 val docs per condition. No tokenizers have been trained yet. The Arabic-aware `SPLIT_PATTERN` in `prepare.py` already handles D1 (combining marks), D2 (plain Arabic), and D3 (PUA codepoints U+E000–U+EFFF) — no changes needed to the split pattern. The `EVAL_TOKENS = 3 * 524288` constant gives the same eval set size used by the autoresearch framework for its English baseline.

**Primary recommendation:** Extend `prepare.py` with a `--vocab-size` flag and structured `fertility_report.json` output, run it for each condition at each vocab size (4096, 8192, 16384 are a reasonable sweep), then run `train.py` per condition and capture `val_bpb` to `baseline_results.json`. Phase 3 reads the baseline JSON to know what to beat.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rustbpe | >=0.1.0 | BPE training engine | Already in use; project's established choice; `rustbpe.Tokenizer` trains from iterator, returns `mergeable_ranks` compatible with tiktoken |
| tiktoken | >=0.11.0 | Tokenizer runtime (encode/decode) | Already in use; wraps `rustbpe` output via `tiktoken.Encoding`; provides `encode_ordinary_batch` for parallel tokenization |
| mlx | >=0.30.0 | Model training + eval | Already in use; `train.py` entire model stack is MLX; Apple Silicon only |
| numpy | >=2.2.6 | `token_bytes.npy` storage | Already in use; stores bytes-per-token lookup for BPB calculation |
| pickle (stdlib) | stdlib | `tokenizer.pkl` persistence | Already in use; stores the `tiktoken.Encoding` object |
| json (stdlib) | stdlib | fertility_report.json, baseline_results.json | No new dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyarrow | >=21.0.0 | Parquet shard reading in `text_iterator()` | Already used in prepare.py for training corpus iteration |
| regex | >=2025.7.34 | `SPLIT_PATTERN` matching in rustbpe | Already in use; required for `\p{L}` Unicode property escapes |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Multiple vocab sizes via `--vocab-size` CLI | Separate config files per vocab size | CLI flag is simpler, no extra files; config files add indirection without benefit |
| `baseline_results.json` captured by wrapper script | Parsing train.py stdout with grep | JSON is structured and machine-readable; stdout parsing is fragile |
| Fixed 8192 vocab for baselines | Sweep 4096/8192/16384 for baselines | Multiple vocab sizes makes TOK-04 fertility table meaningful; only one vocab size (8192) needed for the single baseline val_bpb measurement |

**Installation (already in pyproject.toml):**
```bash
uv sync
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 outputs)

```
prepare.py                            # Modified: add --vocab-size flag, write fertility_report.json
train.py                              # Modified: write baseline_results.json after final eval
~/.cache/autoresearch-arabic/
├── fertility_report.json             # NEW: {d1: {4096: 1.23, 8192: 1.05}, d2: {...}, d3: {...}}
├── baseline_results.json             # NEW: {d1: {val_bpb: X.XX, depth: 4, ...}, d2: {...}, d3: {...}}
├── d1/
│   ├── metadata.txt                  # EXISTS (29 train shards + 1 val)
│   └── tokenizer/                    # NEW per vocab size
│       ├── tokenizer.pkl             # tiktoken.Encoding object
│       └── token_bytes.npy           # bytes-per-token lookup array
├── d2/
│   └── tokenizer/
│       └── ...
└── d3/
    └── tokenizer/
        └── ...
```

**Note on multi-vocab storage:** If multiple vocab sizes are trained, tokenizers can live at `d1/tokenizer_8192/` etc. The baseline training uses a single vocab size (8192 is the current project default). For the fertility sweep, separate tokenizer directories per vocab size prevents caching conflicts.

### Pattern 1: Vocab-Parameterized prepare.py

**What:** Add `--vocab-size` as a CLI argument, defaulting to 8192. Store tokenizer at `tokenizer_{vocab_size}/` subdirectory to avoid cache collisions across sizes.
**When to use:** Always — enables the fertility sweep (TOK-04) without script duplication.

```python
# Source: existing prepare.py pattern, extended
parser.add_argument("--vocab-size", type=int, default=8192,
                   help="BPE vocabulary size (total including special tokens)")
args = parser.parse_args()
VOCAB_SIZE = args.vocab_size

# Tokenizer stored at condition/tokenizer_{vocab_size}/
tokenizer_dir = os.path.join(CACHE_DIR, f"tokenizer_{VOCAB_SIZE}")
```

### Pattern 2: Fertility Report JSON

**What:** After `train_tokenizer()` completes, compute fertility on ~1M chars and merge the result into `fertility_report.json` using the same read-update-write merge pattern established in Phase 1 for `validation_report.json`.
**When to use:** At end of `prepare.py` main, after tokenizer is trained and verified.

```python
# Source: established Phase 1 write pattern (build_dataset.py::write_validation_report)
def write_fertility_report(condition: str, vocab_size: int, fertility: float) -> None:
    report_path = os.path.join(BASE_CACHE, "fertility_report.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            try:
                report = json.load(f)
            except json.JSONDecodeError:
                report = {}
    if condition not in report:
        report[condition] = {}
    report[condition][str(vocab_size)] = round(fertility, 4)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Fertility report updated: {report_path}")
```

### Pattern 3: Baseline Results JSON

**What:** After `evaluate_bpb()` in `train.py`, write the `val_bpb` result plus metadata (depth, vocab_size, params, training_seconds) into `baseline_results.json`. Use the same merge pattern.
**When to use:** At end of `train.py` after the final eval print block.

```python
# Source: train.py output block, extended
import json

baseline_entry = {
    "val_bpb": round(val_bpb, 6),
    "depth": DEPTH,
    "vocab_size": vocab_size,       # tokenizer.get_vocab_size()
    "num_params_M": round(num_params / 1e6, 2),
    "training_seconds": round(total_training_time, 1),
    "total_tokens_M": round(total_tokens / 1e6, 1),
    "window_pattern": WINDOW_PATTERN,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
baseline_path = os.path.join(prepare.BASE_CACHE, "baseline_results.json")
baseline_report = {}
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        try:
            baseline_report = json.load(f)
        except json.JSONDecodeError:
            baseline_report = {}
baseline_report[condition] = baseline_entry
with open(baseline_path, "w", encoding="utf-8") as f:
    json.dump(baseline_report, f, indent=2)
print(f"Baseline results written: {baseline_path}")
```

### Pattern 4: Fertility Sweep Run Order

**What:** Run `prepare.py` for each condition at each vocab size in a defined order. For the baseline training runs (BASE-01/02/03), use a single vocab size (8192 — the current project default).
**When to use:** Codified in tests and in the run sequence documentation.

```bash
# Fertility sweep (TOK-04): three conditions x three vocab sizes
for VS in 4096 8192 16384; do
    uv run prepare.py --condition d1 --vocab-size $VS
    uv run prepare.py --condition d2 --vocab-size $VS
    uv run prepare.py --condition d3 --vocab-size $VS
done

# Baseline training (BASE-01/02/03): one vocab size, 300s each
# IMPORTANT: train.py reads tokenizer from condition/tokenizer/ (not condition/tokenizer_8192/)
# This means the baseline run uses the 8192 tokenizer after renaming/symlinking OR
# prepare.py's default (no --vocab-size) produces tokenizer/ directly.
AUTORESEARCH_CONDITION=d1 uv run train.py
AUTORESEARCH_CONDITION=d2 uv run train.py
AUTORESEARCH_CONDITION=d3 uv run train.py
```

**Critical:** `train.py` calls `Tokenizer.from_directory()` which reads from `TOKENIZER_DIR`. The `init_condition()` sets `TOKENIZER_DIR = get_dirs(condition)[2]` which is `{cache_dir}/tokenizer`. If multi-vocab tokenizers are stored at `tokenizer_8192/`, `train.py` will fail to find them unless `TOKENIZER_DIR` is updated. The cleanest solution: keep the default `prepare.py --vocab-size 8192` producing output at the standard `tokenizer/` directory (for baseline runs), and only use `tokenizer_{size}/` for the fertility sweep.

### Pattern 5: D3 Tokenizer Consideration

**What:** D3 texts contain PUA codepoints (U+E000–U+EFFF). The existing `SPLIT_PATTERN` already includes `[\uE000-\uEFFF]+` as a match group, which means PUA sequences are kept together during pre-tokenization — BPE merges happen within that group. This is exactly correct for D3: each atomic encoded character is one pre-token unit.
**When to use:** No code changes needed. This is a verified correctness property to confirm.

```python
# Source: prepare.py line 45 — SPLIT_PATTERN already correct
SPLIT_PATTERN = r"""[\u0621-\u064A\u064B-\u0652\u0670\uE000-\uEFFF]+|..."""
#                                                    ^^^^^^^^^^^^
#                                                    D3 PUA range included
```

### Anti-Patterns to Avoid

- **Running train.py before prepare.py**: `train.py` calls `Tokenizer.from_directory()` at import time via `prepare.init_condition(condition)` → if `tokenizer.pkl` doesn't exist, `train.py` will fail silently or error. Always run `prepare.py` first per condition.
- **Mixing vocab sizes for baseline**: The baseline val_bpb must use the same vocab size across all three conditions (8192) for a fair comparison. Do not run D1 at 4096 and D2 at 8192.
- **Storing fertility only in stdout**: `print(f"Tokenizer: fertility = {fertility:.3f}")` is transient. Phase 4 paper generation needs the structured table from `fertility_report.json`.
- **Skipping the D2 < D1 sanity check**: The ROADMAP documents: "D2 baseline is lower than D1 baseline (stripping reduces surface complexity — expected sanity check)." This must be verified programmatically in tests, not by visual inspection.
- **Importing prepare.py at top of train.py without init**: `train.py` already imports `prepare` and calls `prepare.init_condition(condition)` before any tokenizer load. This pattern is correct and should not be changed. Adding `--vocab-size` to `prepare.py` must not break this import path.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BPE training from scratch | Custom merge-pair algorithm | `rustbpe.Tokenizer.train_from_iterator()` | Already implemented and working; handles Unicode-aware pre-tokenization via regex pattern |
| Tokenizer serialization | Custom binary format | `pickle.dump(enc, ...)` / `pickle.load(...)` | Already established; `tiktoken.Encoding` pickles cleanly |
| Fertility calculation | Whitespace split + count | Built-in `train_tokenizer()` pattern | Already computed on 1M char sample; just needs to be persisted to JSON |
| BPB evaluation | Custom perplexity calc | `prepare.evaluate_bpb()` | Already implements bits-per-byte correctly using `token_bytes` lookup and MLX tensor ops |
| Merging per-condition JSON results | Replace-on-write | Read-update-write merge pattern (Phase 1 established) | Prevents D1 result from overwriting D2 result |
| Arabic-aware pre-tokenization pattern | Custom regex | Existing `SPLIT_PATTERN` in prepare.py | Already handles Arabic letters, harakat, and D3 PUA range correctly |

**Key insight:** The tokenizer and training infrastructure are complete. Phase 2 is primarily an orchestration and data-capture problem — adding CLI parameterization and JSON output, then running three conditions.

---

## Common Pitfalls

### Pitfall 1: TOKENIZER_DIR Mismatch Between prepare.py and train.py

**What goes wrong:** `prepare.py --vocab-size 4096` writes to `d1/tokenizer_4096/`. Then `AUTORESEARCH_CONDITION=d1 uv run train.py` calls `Tokenizer.from_directory()` which reads from `d1/tokenizer/` — doesn't exist. Training crashes with `FileNotFoundError`.
**Why it happens:** `train.py` uses `TOKENIZER_DIR` set by `init_condition()`, which always resolves to `{cache_dir}/tokenizer` (no size suffix).
**How to avoid:** The baseline training runs must use the default `prepare.py` invocation (no `--vocab-size` flag, or `--vocab-size 8192` which writes to `tokenizer/` directly). Only the fertility sweep uses size-suffixed directories. Make this explicit in the run order. Alternative: add a `--tokenizer-dir` override to `train.py` for non-default vocab sizes.
**Warning signs:** `FileNotFoundError: Missing token_bytes lookup at ... tokenizer/token_bytes.npy`.

### Pitfall 2: D3 Fertility Inflated by Short Tokens

**What goes wrong:** D3 encodes each letter+harakah as a single PUA codepoint. A word like `بِسْمِ` (4 chars + 3 combining) becomes 4 PUA codepoints. BPE at vocab_size=8192 will readily merge frequent PUA pairs, producing fewer tokens per word than D1 (which has the same info spread across 7 Unicode chars). This is the expected result — but if fertility appears equal across D1/D3, the PUA groups may not be splitting correctly.
**Why it happens:** If SPLIT_PATTERN doesn't include the full PUA range, D3 tokens may be handled differently than intended.
**How to avoid:** After training each D3 tokenizer, run the sanity check from `train_tokenizer()`: `enc.encode_ordinary(test)` on a known D3-encoded string and verify roundtrip. The existing sanity check in `prepare.py` only tests D1 ("بِسْمِ اللَّهِ...") and D2 ("بسم الله...") — add a D3 sanity test.
**Warning signs:** D3 fertility is equal to D1 fertility (both > 1.5 tokens/word) — should be lower since each diacritized letter-form is one token.

### Pitfall 3: Baseline train.py Timeout Captured as val_bpb

**What goes wrong:** `train.py` runs for TIME_BUDGET=300s then evaluates. If the machine is under load, the eval itself may not complete within the expected window. There is no timeout on `evaluate_bpb()` — it runs `EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)` steps. With `EVAL_TOKENS = 3 * 524288` and `FINAL_EVAL_BATCH_SIZE = 256`, this is `3 * 524288 / (256 * 2048) = 3` steps — very fast. This is not a real risk, but the val_bpb should be verified to be a plausible value (between 1.5 and 5.0 for Arabic, given English baseline is 1.815).
**Why it happens:** N/A — the eval is efficient. But the sanity range should be checked.
**How to avoid:** Add a plausibility assert in the result-writing code: `assert 1.0 < val_bpb < 10.0` before writing to JSON.
**Warning signs:** `val_bpb: inf` (means `total_bytes = 0` — all tokens were special tokens with 0 byte length).

### Pitfall 4: D2 Baseline Not Lower Than D1

**What goes wrong:** If `D2 val_bpb >= D1 val_bpb`, something is wrong — D2 has fewer unique types (no harakat) so the same model capacity should produce a lower val_bpb. This is the ROADMAP's stated sanity check.
**Why it happens:** Could indicate D2 data was built incorrectly (e.g., stripping was incomplete, or D2 is harder due to more homograph ambiguity). Could also indicate tokenizer vocab size is too small for D1 (more fragmentation).
**How to avoid:** Write a test that reads `baseline_results.json` and asserts `d2_val_bpb < d1_val_bpb`. If this fails, check D2 data integrity with `validate_dataset.py --condition d2` and check fertility table (D2 fertility at same vocab should be <= D1).
**Warning signs:** Fertility table shows D2 tokens/word >= D1 tokens/word at the same vocab size (D2 should have fewer unique forms, so BPE should merge more aggressively, giving lower fertility).

### Pitfall 5: prepare.py Skips Retraining

**What goes wrong:** `train_tokenizer()` has an early return: `if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path): return`. If a broken tokenizer was partially written, subsequent runs skip it silently. This also means the multi-vocab-size feature requires the tokenizer directory to not exist already.
**Why it happens:** Caching is designed for efficiency, but a partial write leaves stale state.
**How to avoid:** Add a `--force` flag to `prepare.py` that deletes and re-trains. For the Phase 2 first run (no tokenizers exist), this is not an issue. Alternatively, verify the pkl file size before trusting it.
**Warning signs:** Training a new vocab size appears to complete instantly (< 1 second) — the cache was hit.

### Pitfall 6: train.py Global Variable Mutation from prepare Import

**What goes wrong:** `train.py` imports `prepare` then calls `prepare.init_condition(condition)`. If `train.py` is run without `AUTORESEARCH_CONDITION` set, `condition = os.environ.get("AUTORESEARCH_CONDITION", "d1")` defaults to "d1". If multiple conditions need to be run in sequence in the same Python process, the globals `CACHE_DIR`, `TOKENIZER_DIR`, etc. would carry over.
**Why it happens:** `prepare.py` uses module-level globals, not encapsulated state.
**How to avoid:** Always run `train.py` as a subprocess with `AUTORESEARCH_CONDITION` set in environment (not in the same Python process). The recommended pattern: `AUTORESEARCH_CONDITION=d2 uv run train.py` — each condition is a separate process.
**Warning signs:** Running all three conditions in a loop produces the same `val_bpb` for all — all read from the same condition's tokenizer.

---

## Code Examples

### Running the Full Phase 2 Sequence

```bash
# Step 1: Train tokenizers at multiple vocab sizes (TOK-01, TOK-02, TOK-03, TOK-04)
# Default vocab (8192) goes to tokenizer/ for baseline use
uv run prepare.py --condition d1
uv run prepare.py --condition d2
uv run prepare.py --condition d3

# Fertility sweep at additional sizes (TOK-04 table completion)
for VS in 4096 16384; do
    uv run prepare.py --condition d1 --vocab-size $VS
    uv run prepare.py --condition d2 --vocab-size $VS
    uv run prepare.py --condition d3 --vocab-size $VS
done

# Step 2: Run baselines (BASE-01, BASE-02, BASE-03) — each ~5 min on Apple Silicon
AUTORESEARCH_CONDITION=d1 uv run train.py
AUTORESEARCH_CONDITION=d2 uv run train.py
AUTORESEARCH_CONDITION=d3 uv run train.py
```

### Reading the Fertility Report

```python
# Source: established project JSON pattern (ensure_ascii=False, json.load)
import json
from pathlib import Path

report_path = Path.home() / ".cache" / "autoresearch-arabic" / "fertility_report.json"
with open(report_path, encoding="utf-8") as f:
    fertility = json.load(f)

# Expected structure:
# {
#   "d1": {"4096": 2.14, "8192": 1.87, "16384": 1.72},
#   "d2": {"4096": 1.83, "8192": 1.61, "16384": 1.48},
#   "d3": {"4096": 1.45, "8192": 1.22, "16384": 1.11}
# }
# D3 fertility should be lowest (each diacritized letter-form is one PUA codepoint)
```

### Verifying BPB Sanity

```python
# Source: train.py output block
assert 1.0 < val_bpb < 10.0, (
    f"val_bpb={val_bpb} is outside plausible range [1.0, 10.0]. "
    f"Check tokenizer and data integrity. English baseline is 1.815."
)
```

### D3 Sanity Check Pattern

```python
# Source: existing prepare.py sanity check, extended for D3
# D3 sanity: encode a known PUA string and verify roundtrip
if CONDITION == "d3":
    # The string "با" (ba + fathah) in D3 encoding would be a single PUA char
    # Build a test string by encoding known text
    from build_dataset import build_atomic_mapping, apply_atomic_encoding
    mapping = build_atomic_mapping()
    test_arabic = "بَسْمِ"
    test_d3 = apply_atomic_encoding(test_arabic, mapping)
    encoded = enc.encode_ordinary(test_d3)
    decoded = enc.decode(encoded)
    assert decoded == test_d3, f"D3 tokenizer roundtrip failed: {test_d3!r} -> {decoded!r}"
    print(f"Tokenizer: D3 sanity check passed ({len(encoded)} tokens for {len(test_d3)} PUA chars)")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single hardcoded VOCAB_SIZE=8192 | Parameterized `--vocab-size` CLI flag | Phase 2 | Enables TOK-04 fertility table across multiple sizes |
| Fertility printed to stdout only | Fertility written to `fertility_report.json` | Phase 2 | Phase 4 can read structured fertility table for paper |
| val_bpb printed to stdout only | val_bpb written to `baseline_results.json` | Phase 2 | Phase 3 has a gate to compare against; Phase 4 has structured data |
| train.py runs one condition hardcoded | `AUTORESEARCH_CONDITION` env var (already exists) | Phase 1 | Each condition is a separate run with clean state |

**Already correct (no changes needed):**
- Arabic-aware `SPLIT_PATTERN` including D3 PUA range
- BPE training from parquet corpus via `text_iterator()`
- `evaluate_bpb()` using `token_bytes` lookup for bits-per-byte calculation
- `token_bytes.npy` array construction (handles special tokens as 0 bytes)

---

## Open Questions

1. **Vocab sizes for the fertility sweep**
   - What we know: ROADMAP says "multiple vocab sizes" but does not specify them. The English baseline uses VOCAB_SIZE=8192 (project default).
   - What's unclear: Whether 3 sizes (4096/8192/16384) or 2 (8192/16384) are sufficient for a publishable fertility table.
   - Recommendation: Use {4096, 8192, 16384} — three points gives a meaningful trend line. The baseline training uses 8192 only (single architecture comparison).

2. **Expected fertility values for D1/D2/D3**
   - What we know: D2 should have lower fertility than D1 (fewer unique forms → more BPE merges). D3 should have lower fertility than D1 (each diacritized letter-form is one PUA codepoint → BPE merges occur on fewer, more distinctive tokens).
   - What's unclear: Whether D3 fertility is lower than D2 at all vocab sizes, or only at smaller vocab sizes.
   - Recommendation: Treat as empirical — run the sweep and record. If D3 is not lower than D2, investigate whether SPLIT_PATTERN is grouping PUA tokens correctly.

3. **Expected val_bpb range for Arabic**
   - What we know: English baseline is 1.815 at depth=4, 11.5M params. Arabic with diacritics (D1) is more complex character-set-wise but the corpus is more structured (classical Arabic, uniform style). D2 has lower surface complexity.
   - What's unclear: Whether Arabic val_bpb will be lower or higher than English 1.815. Arabic text encodes less information per character when fully diacritized (systematic patterns), but the dataset is smaller in tokens than the English training data used for 1.815.
   - Recommendation: Accept the empirical result. The paper's comparison is D1 vs D2 vs D3, not vs English. The English 1.815 is a sanity reference point.

4. **Baseline tokenizer directory for train.py**
   - What we know: `train.py` reads from `{cache_dir}/tokenizer/` hardcoded via `init_condition()`. The current `prepare.py` writes to this exact path.
   - What's unclear: If `--vocab-size` is added and produces `tokenizer_8192/`, how to route `train.py` to the right place.
   - Recommendation: Keep the default `prepare.py` (no `--vocab-size` flag) writing to `tokenizer/` as today. Only add size-suffixed directories when `--vocab-size` is explicitly provided and the size differs from the default. This preserves backward compatibility with `train.py`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 (installed) |
| Config file | none — uses pyproject.toml project root |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TOK-01 | D1 tokenizer trains, produces tokenizer.pkl + token_bytes.npy | unit | `uv run pytest tests/test_tokenizer.py::test_train_tokenizer_d1 -x` | Wave 0 |
| TOK-02 | D2 tokenizer trains, produces same artifacts | unit | `uv run pytest tests/test_tokenizer.py::test_train_tokenizer_d2 -x` | Wave 0 |
| TOK-03 | D3 tokenizer trains; PUA strings encode/decode roundtrip | unit | `uv run pytest tests/test_tokenizer.py::test_train_tokenizer_d3 -x` | Wave 0 |
| TOK-04 | fertility_report.json exists with d1/d2/d3 keys and plausible values | unit | `uv run pytest tests/test_tokenizer.py::test_fertility_report -x` | Wave 0 |
| BASE-01 | baseline_results.json has d1 key with val_bpb in [1.0, 10.0] | integration | `uv run pytest tests/test_baseline.py::test_baseline_d1 -x` | Wave 0 |
| BASE-02 | baseline_results.json has d2 key; d2_val_bpb < d1_val_bpb | integration | `uv run pytest tests/test_baseline.py::test_baseline_d2_sanity -x` | Wave 0 |
| BASE-03 | baseline_results.json has d3 key with val_bpb in [1.0, 10.0] | integration | `uv run pytest tests/test_baseline.py::test_baseline_d3 -x` | Wave 0 |

**Test strategy for TOK-01/02/03:** These are slow (BPE training on 1M chars takes ~minutes). Use a tiny fixture: a `conftest.py` fixture that writes 100 rows of Arabic text to a temp parquet, then calls `train_tokenizer()` with `max_chars=10_000` and `doc_cap=500`. The fixture-based test verifies the pkl and npy are written and the sanity check passes. Mark cache-dependent integration tests as `pytest.mark.skip` until the real data is available.

**Test strategy for BASE-01/02/03:** These cannot be unit-tested (training is 300 seconds). Tests should check that `baseline_results.json` exists and has the expected schema/values — run after the actual training is complete. Mark as `@pytest.mark.requires_baseline` and skip in CI.

### Sampling Rate

- **Per task commit:** `uv run pytest tests/ -x -q` (fast unit tests only; integration tests skipped)
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** All skipped integration tests removed from skip after training runs complete; full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_tokenizer.py` — covers TOK-01 through TOK-04 with tiny fixture corpus
- [ ] `tests/test_baseline.py` — covers BASE-01 through BASE-03 (reads from `~/.cache`; skipped until training runs complete)
- [ ] Update `tests/conftest.py` — add `tiny_corpus_parquet` fixture for tokenizer unit tests (writes 100 rows to tmp_path, usable with `text_iterator`)
- [ ] Extend `prepare.py` — add `--vocab-size` CLI flag and `write_fertility_report()` function
- [ ] Extend `train.py` — add `baseline_results.json` writer after `evaluate_bpb()` call

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection of `prepare.py` — all function signatures, constants, and SPLIT_PATTERN confirmed
- Direct code inspection of `train.py` — GPTConfig defaults (DEPTH=4, WINDOW_PATTERN="SSSL"), TIME_BUDGET=300, EVAL_TOKENS=3*524288, FINAL_EVAL_BATCH_SIZE=256 confirmed
- `~/.cache/autoresearch-arabic/` directory inspection — confirmed data exists: 30 shards per condition, 1,434,471 train + 29,274 val docs
- Phase 1 SUMMARY files — confirmed validation_report.json passes all 3 conditions at corpus scale

### Secondary (MEDIUM confidence)

- `pyproject.toml` dependency versions — confirmed rustbpe>=0.1.0, tiktoken>=0.11.0, mlx>=0.30.0 in lockfile
- ROADMAP.md Phase 2 success criteria — "D2 baseline is lower than D1 baseline" sanity check is an explicit stated requirement

### Tertiary (LOW confidence — marked for validation)

- Expected fertility ranges (D3 < D2 < D1) — theoretically motivated, not yet empirically confirmed for this corpus
- Expected val_bpb range for Arabic at depth=4 — English is 1.815; Arabic estimate is 1.5–4.0 depending on condition

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use and tested; no new dependencies needed
- Architecture patterns: HIGH — derived from existing working code; extensions are additive only
- Expected output values: LOW — fertility and val_bpb values are theoretical until empirically measured

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (rustbpe and MLX APIs are stable; tiktoken encoding format is stable)
