# Phase 5: The Compositionality–Atomicity Trade-off - Research

**Researched:** 2026-03-14
**Domain:** Arabic LM pretraining mechanistic analysis — BPBL metric, embedding extraction, iso-data training
**Confidence:** HIGH (all findings grounded in direct code inspection and filesystem verification)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Experiment 3 (BPBL):**
- Denominator: count of Arabic base letters via `[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]` before encoding
- Numerator: total nats from model inference on the val set
- Formula: `total_nats / base_letter_count / ln(2)`
- Source: re-evaluate existing overnight checkpoints (3 seeds x D1, 3 seeds x D3) — no retraining needed
- Report mean ± std across seeds
- Framing: BPBL confirms D1 beats D3. This is the honest result. Do NOT frame as "proof D3 wins"
- D2 excluded from BPBL tables (predicts less information, not a fair comparison)

**Experiment 4 (Embedding Similarity):**
- Extract wte matrices from D1 and D3 checkpoints
- D3: for each base letter, find all PUA embeddings sharing that letter, compute pairwise cosine similarity within the letter's group
- D1: extract harakah mark embeddings, check whether they cluster separately from base letters
- Visualization: heatmap of cosine similarities grouped by base letter (D3), t-SNE/PCA side by side
- This is the paper's new mechanistic centerpiece

**Experiment 5 (Iso-data Scaling):**
- X-axis: cumulative base letters processed (fertility × steps × batch_size × seq_len)
- Target data points: ~5M, 15M, 30M, 50M, 100M base letters
- Each condition uses its OWN optimal architecture (D1: depth=4/AR=26/HD=128/SS; D3: depth=2/AR=64/HD=96/SS)
- 3 seeds per data point
- ~30 training runs total, ~2.5 hours GPU time
- Plot BPBL vs base-letters-processed

**Dropped:**
- Token boundary analysis — tautological
- Morpheme coverage — already addressed in Experiment 2

### Claude's Discretion
- Exact implementation of base letter counting function
- Plot styling and figure layout for paper
- t-SNE vs PCA vs UMAP for embedding visualization
- How to compute "base letters processed" from training logs (fertility × tokens per batch)
- Error handling for edge cases (non-Arabic characters in val set)

### Deferred Ideas (OUT OF SCOPE)
- D4 factored encoding (separate harakat and consonant prediction streams)
- Downstream task evaluation (NER, sentiment, QA)
- Multi-corpus validation (MSA, dialectal Arabic)
- Large-model validation of whether D3 wins at unconstrained embedding capacity
</user_constraints>

---

## Summary

Phase 5 requires three new experiment scripts that build directly on the existing codebase. The project already has `prepare.py` (with `evaluate_bpb`, `make_dataloader`, `Tokenizer`, `_iter_eval_batches`), `build_dataset.py` (with `ARABIC_LETTER_RE`, `HARAKAT_RE`, `build_atomic_mapping`), and `train.py` (with `GPT`, `GPTConfig`, `AdamW`). No new libraries are needed beyond what is already installed. The critical gap is that `train.py` does NOT save model checkpoints — it only logs metrics — so "re-evaluating overnight checkpoints" means running 3-seed training runs with the best-found hyperparameters and extracting embeddings from in-memory models at the end of training.

The three experiments require: (1) a BPBL evaluator extending `evaluate_bpb` with a base-letter denominator instead of byte denominator; (2) a `GPT` model instantiation, weight extraction via `tree_flatten`, and cosine-similarity analysis of the `wte.weight` matrix grouped by PUA token→base letter reverse mapping; (3) modified training runs that track `steps × TOTAL_BATCH_SIZE × base_letters_per_token` as the x-axis and evaluate BPBL at each budget checkpoint.

**Primary recommendation:** All three experiments go in `experiments/` as standalone scripts (`exp3_bpbl.py`, `exp4_embedding.py`, `exp5_iso_data.py`) following the existing pattern in `paper/run_fixed_arch_ablation.py`.

---

## Area 1: BPBL Metric Implementation

### How val_bpb currently works

`prepare.py:evaluate_bpb()` (lines 320–336) is the existing evaluator. It:
1. Gets `token_bytes` — a precomputed numpy array mapping token_id → UTF-8 byte length (saved as `tokenizer/token_bytes.npy`)
2. Runs `steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)` batches where `EVAL_TOKENS = 3 * 524288`
3. For each batch: calls `model(x, y, reduction="none")` to get per-position cross-entropy, masks out special tokens (where `nbytes == 0`), sums nats weighted by mask, sums byte counts
4. Returns `total_nats / (ln(2) * total_bytes)` = bits per byte

### BPBL extension pattern

BPBL replaces the byte denominator with a base-letter count denominator. The formula is:

```
BPBL = total_nats / base_letter_count / ln(2)
```

Where `base_letter_count` is the number of Arabic base letters (regex `[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]`) in the raw val text BEFORE any encoding. Since D1 and D3 encode the same underlying text, this count is identical for both conditions.

**Implementation approach:** The base letter count is a property of the raw text corpus, not of the tokenized batches. It must be computed once by scanning the val parquet shard directly — not during the batched forward pass. The procedure is:

1. Load val shard via `pyarrow.parquet.read_table`
2. Apply `ARABIC_LETTER_RE.findall(text)` across all val texts, sum up lengths
3. Run the existing batched forward pass (same as `evaluate_bpb`) to get `total_nats`
4. Return `total_nats / base_letter_count / math.log(2)`

The `_document_batches` function in `prepare.py` already isolates the val shard via `VAL_FILENAME`. The val shard path is `os.path.join(DATA_DIR, VAL_FILENAME)`.

### "Overnight checkpoints" — what they actually are

**Critical finding:** `train.py` does NOT save model weights to disk. The word "checkpoint" in CONTEXT.md refers to "model instances trained with the best-found hyperparameters" — i.e., run training again with the search-winner config, 3 different random seeds, extract BPBL at the end of each run while the model is still in memory. No checkpoint loading is required.

**Confirmation:** Searching `train.py` for `save`, `npz`, `safetensors`, `checkpoint` returned zero results. The `~/.cache/autoresearch-arabic/` directory contains only parquet data shards, tokenizer pkl files, and JSON results — no `.npz` or `.safetensors` model weight files.

**Therefore:** exp3_bpbl.py must be a training script that:
1. Patches the train.py hyperparameters for the best D1 config (depth=4, AR=26, HD=128, SS window, from `search_results.json`)
2. Runs training 3 times with different seeds
3. After each run, calls a BPBL evaluator while the model is in memory
4. Saves `[seed, val_bpb, bpbl, base_letter_count]` tuples to JSON
5. Same for D3 best config (depth=2, AR=64, HD=96, SS window)

### Val set base letter count estimate

From `baseline_results.json`:
- D1: `eval_words = 2,990,668`, `eval_targets = 7,514,371` tokens
- D3: `eval_words = 2,990,668` (same text), `eval_targets = 6,532,158` tokens (fewer due to lower fertility)

Arabic diacritized text: approximately 4–5 base letters per word. A rough estimate gives ~13–15 million base letters in the val set. The exact count is computed once by scanning the parquet file.

---

## Area 2: Embedding Extraction

### How to get wte weights

The `GPT` model in `train.py` stores the embedding matrix as `model.wte` — an `mlx.nn.Embedding` instance. Its weight matrix is `model.wte.weight` with shape `(vocab_size, n_embd)`.

After training, `model.wte.weight` is an `mx.array` with dtype `bfloat16`. To get a numpy array for cosine similarity computation:

```python
from mlx.utils import tree_flatten
import mlx.core as mx
import numpy as np

# After training loop ends (model still in memory):
wte_weights = np.array(model.wte.weight.astype(mx.float32))
# shape: (vocab_size, n_embd) — e.g., (8192, 192) for D3 or (8192, 128) for D1
```

The tokenizer is loaded from `Tokenizer.from_directory()`. To map token_id → decoded string, use `tokenizer.decode([token_id])`.

### D3 PUA token grouping

`build_dataset.py:build_atomic_mapping()` returns a dict `{original_seq: pua_char}`. The reverse mapping is `{pua_char: original_seq}` (built by `build_reverse_mapping`). The original sequence is either `letter+harakah`, `letter+shaddah+harakah`, or `letter+shaddah`.

PUA codepoints are in range `U+E000` to `U+E290` (657 entries in the mapping print, 621 in the returned dict — the discrepancy is the standalone harakah entries at the tail that map from bare harakah chars, not from PUA codepoints; use `len(mapping)` as returned).

**D3 grouping procedure:**
1. Load the atomic mapping from `~/.cache/autoresearch-arabic/atomic_mapping.json` (saved as `{key: hex(ord(v))}`)
2. Build reverse: `pua_char → original_seq`
3. Extract base letter from original_seq: the base letter is always `original_seq[0]` (first character is always an Arabic base letter in all three combo types)
4. Build `{base_letter → [pua_token_ids]}` using the tokenizer's encoding of each PUA character
5. For each base letter's group: compute pairwise cosine similarity matrix from `wte_weights[token_ids]`

**Key number:** There are 36 Arabic base letters (standard letters `U+0621-U+063A` and `U+0641-U+064A`). Each letter appears in ~17–18 PUA token variants (9 single harakah + 8 double harakah + 1 shaddah-only = 18 entries per letter). The D3 model has 8192 vocab tokens; roughly 657 of them are PUA tokens.

### D1 harakah embedding analysis

For D1, the harakat marks are standalone combining characters: fathah `U+064E`, dammah `U+064F`, kasrah `U+0650`, etc. In D1's BPE tokenizer, each harakah may appear as a singleton token or merged into multi-character tokens.

**D1 grouping procedure:**
1. For each harakah codepoint, try `tokenizer.enc.encode_single_token(harakah_char)` — if it exists as a singleton, extract its embedding
2. Also extract embeddings for base letter tokens (e.g., `ba` alone without harakah)
3. Compute cosine similarity between harakah tokens and base letter tokens

Note: many harakah marks may not exist as singleton tokens in D1's vocabulary since BPE may always merge them. The analysis should check and report how many standalone harakah tokens exist.

### Visualization tools

The project already uses `pyarrow` and `numpy`. For visualization, the existing `paper/run_fixed_arch_ablation.py` only writes JSON and Markdown — no matplotlib or sklearn are currently installed.

**Required additions:** `matplotlib` and `scikit-learn` for t-SNE/PCA. These are not in `pyproject.toml`.

```bash
uv add matplotlib scikit-learn
```

For the heatmap: `matplotlib.pyplot.imshow` on the pairwise cosine similarity matrix, grouped by base letter. For the embedding space plot: `sklearn.manifold.TSNE` or `sklearn.decomposition.PCA` on the full wte matrix (8192 × n_embd), with color coding by token type (base letter, harakah, PUA-letter-group).

---

## Area 3: Iso-data Scaling Curves

### Fertility and base-letters-processed calculation

Confirmed fertilities from `~/.cache/autoresearch-arabic/fertility_report.json` (8192 vocab):
- D1: 2.5189 tokens/word
- D3: 2.1934 tokens/word
- D3 sees ~12.9% more text per batch at equal token count

`TOTAL_BATCH_SIZE = 2**15 = 32768` tokens/step (from `search_results.json` hyperparameters, both D1 and D3 best configs use `2**15`).

**Base letters per step formula:**
```python
base_letters_per_step = TOTAL_BATCH_SIZE * avg_base_letters_per_token
# where avg_base_letters_per_token = avg_base_letters_per_word / fertility
```

The `avg_base_letters_per_word` is a constant for both D1 and D3 (same underlying text). It can be measured once by counting base letters and words in the val shard.

**Step count per budget target:**
```python
# For a target of N base letters:
steps_needed = N / base_letters_per_step
# D3 needs fewer steps than D1 for the same base-letter budget (higher BL/token)
```

From estimates with ~4.5 base letters/word:
- 5M letters: D1 ~85 steps (~30s), D3 ~74 steps
- 15M letters: D1 ~256 steps (~87s), D3 ~223 steps
- 30M letters: D1 ~512 steps (~174s), D3 ~446 steps
- 50M letters: D1 ~854 steps (~290s), D3 ~744 steps
- 100M letters: D1 ~1708 steps (~581s), D3 ~1487 steps

**Total wall time estimate:** ~1.8 hours for all 30 runs (6 budgets × 2 conditions × 3 seeds, scaling from 5M to 100M). This matches the CONTEXT.md claim of "~2.5 hours GPU time" (slight overestimate for safety margin).

### Training run implementation

The iso-data script must:
1. Replace `TIME_BUDGET` with a `MAX_STEPS` integer (computed from `target_base_letters / base_letters_per_step`)
2. Set `mx.random.seed(seed)` at the start of each run (current `train.py` hardcodes `seed=42`)
3. At the end of each run, evaluate BPBL while the model is in memory
4. Log `{seed, target_base_letters, steps_run, bpbl, val_bpb}` to JSON

**Pattern from `run_fixed_arch_ablation.py`:** It patches `train.py` source code via regex and runs `subprocess.run(["uv", "run", "train.py"], ...)`. For the iso-data experiment, this same subprocess pattern works but requires patching `TIME_BUDGET` (or adding a `MAX_STEPS` env var) and adding a seed parameter.

**Alternative approach:** Import `train.py`'s model/optimizer construction directly and reuse the training loop, avoiding subprocess overhead. This is cleaner but requires refactoring `train.py` to expose a `train_for_steps(n_steps, seed)` function. The CONTEXT.md discretion area says Claude decides how to compute base letters processed — the subprocess approach is simpler and less risky.

### Seed variation

`train.py` hardcodes `mx.random.seed(42)` on line 399. For 3-seed runs, the seed must be varied. Options:
- Patch via env variable: add `seed = int(os.environ.get("AUTORESEARCH_SEED", "42"))` to `train.py`
- Patch via regex in the subprocess driver (as `run_fixed_arch_ablation.py` does for other params)

---

## Area 4: Existing Assets Catalog

### Directly reusable (no modification needed)

| Asset | Location | What it provides |
|-------|----------|-----------------|
| `ARABIC_LETTER_RE` | `build_dataset.py:54` | Regex `[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]` for base letter counting |
| `HARAKAT_RE` | `build_dataset.py:38` | Regex `[\u064B-\u0652\u0670]` for harakat detection |
| `build_atomic_mapping()` | `build_dataset.py:60` | Returns `{original_seq: pua_char}` dict with 621+ entries |
| `build_reverse_mapping()` | `build_dataset.py:131` | Returns `{pua_char: original_seq}` for D3 analysis |
| `evaluate_bpb()` | `prepare.py:320` | Batched eval loop; extend denominator for BPBL |
| `make_dataloader()` | `prepare.py:273` | Val dataloader; reuse for BPBL eval |
| `Tokenizer.from_directory()` | `prepare.py:212` | Loads tokenizer pkl for a given condition |
| `init_condition()` | `prepare.py:104` | Sets globals before any prepare utility is called |
| `GPT`, `GPTConfig` | `train.py:29,136` | Model class; wte is `model.wte.weight` after training |
| `AdamW` | `train.py:231` | Optimizer; same setup as current train.py |
| `get_lr_multiplier()` | `train.py:389` | LR schedule; reuse unchanged |
| `atomic_mapping.json` | `~/.cache/autoresearch-arabic/atomic_mapping.json` | Saved mapping as `{key: hex(ord(v))}` |
| `search_results.json` | `search_results.json` | Best hyperparams for D1 and D3 |
| `run_fixed_arch_ablation.py` | `paper/run_fixed_arch_ablation.py` | Pattern for subprocess-based multi-run driver |

### Needs extension/modification

| Asset | What needs changing |
|-------|-------------------|
| `evaluate_bpb()` | New function `evaluate_bpbl()` with base-letter denominator instead of byte denominator |
| `train.py` seed | Add `AUTORESEARCH_SEED` env var to make seed configurable |
| `train.py` step budget | Add `MAX_STEPS` override to replace time-budget loop |

### Must install (not currently in pyproject.toml)

```bash
uv add matplotlib scikit-learn
```

These are needed for t-SNE/PCA visualization and heatmap plots in Experiment 4.

---

## Area 5: Checkpoint Files — Reality Check

### Finding

There are NO model weight checkpoint files on disk. The `~/.cache/autoresearch-arabic/` directory contains:
```
atomic_mapping.json        — PUA encoding table
baseline_results.json      — scalar metrics (val_bpb, val_npw, etc.)
collision_stats.json/.txt  — Phase 1 homograph analysis
fertility_report.json      — tokenizer fertility measurements
validation_report.json     — data integrity checks
d1/, d2/, d3/              — parquet data shards + tokenizer pkl files
```

No `.npz`, `.safetensors`, or `.pkl` model weight files exist anywhere in the cache or project directory.

### Implication for Exp 3 (BPBL) and Exp 4 (Embedding)

Both experiments require a trained model. Since no checkpoints exist, the experiment scripts must train from scratch (3 seeds × 2 conditions = 6 training runs) and perform extraction/evaluation while the model is live in memory. This is actually how `run_fixed_arch_ablation.py` works for Phase 4 — train, parse metrics from stdout, done.

For Experiment 4 (embedding extraction), the model must be kept in memory after training ends to extract `model.wte.weight`. This means the embedding extraction cannot be a post-hoc subprocess — it must happen within the same Python process that ran training, OR the process must save the wte matrix to a `.npy` file before exiting.

**Recommended pattern:** At the end of each training run, before Python exits:
```python
import numpy as np
wte_np = np.array(model.wte.weight.astype(mx.float32))
np.save(f"experiments/results/wte_{condition}_seed{seed}.npy", wte_np)
```

Then a separate analysis script loads these `.npy` files and computes cosine similarities.

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose |
|---------|---------|---------|
| mlx | >=0.30.0 | Training, model weights, embedding extraction |
| numpy | >=2.2.6 | Array operations, saving wte to .npy |
| pyarrow | >=21.0.0 | Reading val shard parquet files for base letter counting |
| tiktoken | >=0.11.0 | Wrapped by `Tokenizer` class for encode/decode |
| rustbpe | >=0.1.0 | BPE training (not needed for Phase 5) |

### Must Add
| Library | Version | Purpose |
|---------|---------|---------|
| matplotlib | latest | Heatmaps, embedding scatter plots |
| scikit-learn | latest | t-SNE and PCA for embedding visualization |

**Installation:**
```bash
uv add matplotlib scikit-learn
```

---

## Architecture Patterns

### Recommended Project Structure
```
experiments/
├── exp3_bpbl.py           # 3-seed BPBL evaluation for D1 and D3
├── exp4_embedding.py      # wte extraction + cosine similarity analysis
├── exp5_iso_data.py       # iso-data scaling curve training runs
└── results/
    ├── bpbl_results.json
    ├── wte_d1_seed{0,1,2}.npy
    ├── wte_d3_seed{0,1,2}.npy
    ├── embedding_similarity.json
    ├── iso_data_results.json
    └── *.png              # figures
```

### Pattern 1: BPBL Evaluator

```python
# Extend evaluate_bpb() — compute base letter count from raw val text
import re, math
import pyarrow.parquet as pq
import mlx.core as mx

ARABIC_LETTER_RE = re.compile(r'[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]')

def count_val_base_letters(data_dir, val_filename):
    """Count Arabic base letters in val shard BEFORE any encoding."""
    path = os.path.join(data_dir, val_filename)
    table = pq.read_table(path)
    texts = table.column("text").to_pylist()
    return sum(len(ARABIC_LETTER_RE.findall(t)) for t in texts if t)

def evaluate_bpbl(model, tokenizer, batch_size, base_letter_count):
    """Bits per base letter — condition-fair metric."""
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").reshape(-1)
        # All positions contribute (no byte mask needed — denominator is letters not bytes)
        valid = (y.reshape(-1) != -1)
        total_nats += float(mx.sum(loss_flat * valid).item())
    return total_nats / base_letter_count / math.log(2)
```

### Pattern 2: Embedding Extraction and Grouping

```python
import numpy as np
from mlx.utils import tree_flatten
import mlx.core as mx

# After training loop ends, model is in memory:
def extract_wte(model):
    wte_np = np.array(model.wte.weight.astype(mx.float32))
    return wte_np  # shape: (vocab_size, n_embd)

def load_atomic_mapping_reverse():
    """Load PUA->original_seq reverse mapping."""
    import json
    path = os.path.join(os.path.expanduser("~"), ".cache",
                        "autoresearch-arabic", "atomic_mapping.json")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    # saved: {original_seq: hex_str}
    reverse = {chr(int(v, 16)): k for k, v in saved.items()}
    return reverse  # {pua_char: original_seq}

def group_d3_embeddings_by_base_letter(tokenizer, wte_np, reverse_mapping):
    """Returns {base_letter: np.array of shape (n_variants, n_embd)}."""
    groups = {}
    for pua_char, original_seq in reverse_mapping.items():
        base_letter = original_seq[0]  # always Arabic letter
        try:
            token_id = tokenizer.enc.encode_single_token(pua_char)
        except Exception:
            continue  # PUA char not in vocab (shouldn't happen but guard)
        emb = wte_np[token_id]  # (n_embd,)
        if base_letter not in groups:
            groups[base_letter] = []
        groups[base_letter].append(emb)
    return {k: np.stack(v) for k, v in groups.items() if len(v) > 1}

def cosine_similarity_matrix(embeddings):
    """embeddings: (n, d) -> (n, n) cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    return normed @ normed.T
```

### Pattern 3: Iso-data Training Driver

```python
# Patch train.py to use step budget + configurable seed, then subprocess per run

def patch_train_for_iso_data(train_path, max_steps, seed, params):
    """Patch train.py: set MAX_STEPS, AUTORESEARCH_SEED, and best-config params."""
    text = train_path.read_text(encoding="utf-8")
    # patch hyperparams
    for key, value in params.items():
        text = re.sub(rf"^{key}\s*=\s*.+$", f"{key} = {value}", text, flags=re.MULTILINE)
    # add MAX_STEPS override (or set via env)
    train_path.write_text(text, encoding="utf-8")

# Or, simpler: use env vars for seed and step budget, add to train.py:
# seed = int(os.environ.get("AUTORESEARCH_SEED", "42"))
# max_steps = int(os.environ.get("AUTORESEARCH_MAX_STEPS", "0")) or None
```

### Anti-Patterns to Avoid

- **Assuming checkpoints exist:** No `.npz` or `.safetensors` files are saved by `train.py`. Do not write code that tries to load them.
- **Using time budget for iso-data runs:** The time-budget loop means different hardware gets different step counts. Use `MAX_STEPS` directly.
- **Computing base letter count from tokenized batches:** The BPBL denominator must come from the raw pre-encoding text, not from token sequences.
- **Single-seed BPBL:** The overnight variance showed single seeds are noisy; always report mean ± std over 3 seeds.
- **Including D2 in BPBL tables:** D2 has a lower entropy target (no harakat); BPBL for D2 is not comparable to D1/D3 BPBL.
- **Using the same architecture for D1 and D3 in iso-data:** CONTEXT explicitly requires each condition uses its OWN optimal architecture to avoid adding an architecture confound.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Cosine similarity | Custom dot-product loop | `numpy` vector operations — 3 lines |
| t-SNE embedding | Custom dim reduction | `sklearn.manifold.TSNE` |
| PCA embedding | Manual SVD | `sklearn.decomposition.PCA` |
| Arabic regex | Hand-coded character table | `ARABIC_LETTER_RE` from `build_dataset.py` |
| PUA reverse mapping | Manual iteration | `build_reverse_mapping()` from `build_dataset.py` |
| Parquet reading | Manual binary parsing | `pyarrow.parquet.read_table` |

---

## Common Pitfalls

### Pitfall 1: Conflating Token Count and Base Letter Count

**What goes wrong:** Computing BPBL by dividing total_nats by the number of tokens (which is what `evaluate_bpb` uses as the implicit denominator before multiplying by bytes). BPBL must use the count of Arabic base letters in the raw val text, not tokens.

**Why it happens:** The existing `evaluate_bpb` loop accumulates `nbytes` per token. BPBL needs a fixed denominator pre-computed from the parquet file.

**How to avoid:** Compute `base_letter_count` once by scanning the parquet file before starting the eval loop. Pass it as a parameter to `evaluate_bpbl()`.

### Pitfall 2: D1 and D3 Val Shards Encode Different Codepoints for the Same Content

**What goes wrong:** Loading the D3 val shard and counting Arabic base letters (`[\u0621-\u063A...]`) on the PUA-encoded text — PUA codepoints are in `U+E000-U+E290`, so the regex misses them entirely and returns 0 base letters.

**Why it happens:** D3 replaces base letters with PUA codepoints before storing in parquet.

**How to avoid:** Count base letters from the D1 val shard (or any undecoded text). Since D1 and D3 encode the same underlying text, the base letter count is identical — always use D1's val shard for the base-letter count, even when evaluating D3's model.

### Pitfall 3: Embedding Extraction Requires Same vocab_size as Trained Model

**What goes wrong:** Rebuilding the `GPT` model with the wrong `vocab_size` before extracting `wte.weight`. The embedding shape is `(vocab_size, n_embd)`; if vocab_size doesn't match the trained model, the weights won't align with the tokenizer's token IDs.

**How to avoid:** Always load `vocab_size` from the tokenizer (`tokenizer.get_vocab_size()`) and pass it to `GPTConfig` — do not hardcode 8192 in the embedding analysis script.

### Pitfall 4: mlx.array Must Be Explicitly Evaluated Before numpy Conversion

**What goes wrong:** `np.array(model.wte.weight)` may return garbage or fail if the array hasn't been evaluated (MLX uses lazy evaluation).

**How to avoid:** Call `mx.eval(model.parameters())` before extracting any weights — this is already done in `train.py` after `init_weights()` and after each optimizer step.

### Pitfall 5: t-SNE on All 8192 Tokens is Slow

**What goes wrong:** Running t-SNE on the full `(8192, n_embd)` embedding matrix takes several minutes and produces an unreadable plot.

**How to avoid:** Subset to ~500–1000 tokens of interest: PUA tokens for D3, plus base letter tokens and harakah tokens for D1. Use PCA as a faster sanity check before committing to t-SNE.

### Pitfall 6: Seed Variance is Real

**What goes wrong:** Running a single seed and concluding D1 > D3 or D3 > D1 at a specific data budget.

**Evidence:** Phase 3 overnight runs showed meaningful variance across seeds. CONTEXT explicitly requires 3 seeds per data point for the iso-data curves.

**How to avoid:** 3 seeds minimum; report mean ± std in all tables and error bands in all plots.

---

## Code Examples

### Reading atomic_mapping.json (saved format)

```python
# Source: build_dataset.py:587-591 (save) and direct inspection of atomic_mapping.json
import json
mapping_path = os.path.join(os.path.expanduser("~"), ".cache",
                             "autoresearch-arabic", "atomic_mapping.json")
with open(mapping_path, encoding="utf-8") as f:
    saved = json.load(f)
# saved is {original_seq: "0xe000"} — hex string, not the char
# To reconstruct char mapping:
pua_mapping = {k: chr(int(v, 16)) for k, v in saved.items()}
reverse_mapping = {chr(int(v, 16)): k for k, v in saved.items()}
```

### Loading model weights as numpy after training

```python
# Source: train.py GPT class + MLX docs
import numpy as np
import mlx.core as mx

# In the training script, after the training loop:
mx.eval(model.parameters())  # ensure evaluation
wte_np = np.array(model.wte.weight.astype(mx.float32))  # (vocab_size, n_embd)
np.save("experiments/results/wte_d1_seed0.npy", wte_np)
```

### Cosine similarity grouped by base letter (D3)

```python
# For each letter group:
import numpy as np

def mean_intra_group_cosine_sim(embeddings):
    """embeddings: (n_variants, n_embd). Returns mean pairwise cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T  # (n, n)
    n = len(embeddings)
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(n, dtype=bool)
    return sim_matrix[mask].mean()
```

### Mapping steps to base letters processed

```python
def steps_for_budget(target_base_letters, base_letters_per_word,
                     fertility, total_batch_size):
    """Compute steps needed to process target_base_letters."""
    base_letters_per_token = base_letters_per_word / fertility
    base_letters_per_step = total_batch_size * base_letters_per_token
    return int(target_base_letters / base_letters_per_step)

# Example:
TOTAL_BATCH_SIZE = 2**15  # 32768
D1_FERTILITY = 2.5189
# Measure base_letters_per_word from val shard once:
# base_letters_per_word = count_val_base_letters(...) / eval_words
steps = steps_for_budget(
    target_base_letters=50_000_000,
    base_letters_per_word=4.5,  # measure this precisely from val shard
    fertility=D1_FERTILITY,
    total_batch_size=TOTAL_BATCH_SIZE,
)
```

---

## Key Numbers Reference

| Fact | Value | Source |
|------|-------|--------|
| D1 optimal: depth/AR/HD | 4 / 26 / 128 | `search_results.json` |
| D1 optimal model_dim | 128 | Computed: `((4*26+127)//128)*128 = 128` |
| D3 optimal: depth/AR/HD | 2 / 64 / 96 | `search_results.json` |
| D3 optimal model_dim | 192 | Computed: `((2*64+95)//96)*96 = 192` |
| D1 best val_bpb | 0.660090 | `search_results.json` |
| D3 best val_bpb | 0.889682 | `search_results.json` |
| D1 val_npw (baseline run) | 7.470252 nats/word | `baseline_results.json` |
| D3 val_npw (baseline run) | 7.473506 nats/word | `baseline_results.json` |
| D1 fertility at 8192 | 2.5189 tokens/word | `fertility_report.json` |
| D3 fertility at 8192 | 2.1934 tokens/word | `fertility_report.json` |
| D3/D1 fertility ratio | 0.871 | Computed |
| TOTAL_BATCH_SIZE (both conditions) | 32768 (2^15) | `search_results.json` hyperparams |
| Atomic mapping entries | 621 dict / 657 printed | `build_atomic_mapping()` inspection |
| PUA range | U+E000–U+E290 | `build_dataset.py` output |
| Arabic base letters covered | 36 | 26 standard + 10 extended |
| PUA variants per base letter | ~17–18 | 9 single + 8 double + 1 shaddah-only |
| Val eval_words (D1 and D3) | 2,990,668 | `baseline_results.json` |
| Val eval_targets (D1 tokens) | 7,514,371 | `baseline_results.json` |
| Val eval_targets (D3 tokens) | 6,532,158 | `baseline_results.json` |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2+ |
| Config file | none — no pytest.ini; uses `pyproject.toml` dev deps |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Current Test Baseline

3 existing test failures (pre-existing, unrelated to Phase 5):
1. `test_baseline_d3` — asserts `1.0 < val_bpb < 10.0` but D3 val_bpb=0.892976 (stale assertion)
2. `test_baseline_schema` — asserts `window_pattern == "SSSL"` but actual is `"SSS"`
3. `test_load_dataset_with_progress_keyboard_interrupt` — test fixture issue with local dataset path

Phase 5 tests should NOT fix these — they are pre-existing.

### Phase 5 Requirements to Test Map

| Req | Behavior | Test Type | Command |
|-----|----------|-----------|---------|
| EXP3-01 | `count_val_base_letters()` returns non-zero count for D1 val shard | unit | `uv run pytest tests/test_phase5.py::test_count_val_base_letters -x` |
| EXP3-02 | `evaluate_bpbl()` returns float in plausible range (1.0–20.0 bits/letter) | unit (mock model) | `uv run pytest tests/test_phase5.py::test_bpbl_plausible_range -x` |
| EXP3-03 | D3 base letter count equals D1 base letter count (same underlying text) | unit | `uv run pytest tests/test_phase5.py::test_d1_d3_same_base_letter_count -x` |
| EXP4-01 | PUA token grouping covers all 36 base letters with at least 9 variants each | unit | `uv run pytest tests/test_phase5.py::test_pua_grouping_coverage -x` |
| EXP4-02 | `cosine_similarity_matrix()` returns (n,n) matrix with diagonal=1.0 | unit | `uv run pytest tests/test_phase5.py::test_cosine_similarity_matrix -x` |
| EXP5-01 | `steps_for_budget()` returns correct step count for known fertility | unit | `uv run pytest tests/test_phase5.py::test_steps_for_budget -x` |
| EXP3-INT | BPBL experiment writes `bpbl_results.json` with mean/std for D1 and D3 | integration | `uv run pytest tests/test_phase5.py::test_bpbl_results_file -x` |

### Sampling Rate
- Per task commit: `uv run pytest tests/test_phase5.py -x -q`
- Per wave merge: `uv run pytest tests/ -q`
- Phase gate: Full suite (3 pre-existing failures acceptable, 0 new failures)

### Wave 0 Gaps
- [ ] `tests/test_phase5.py` — covers EXP3-01 through EXP5-01
- [ ] `experiments/` directory — must be created
- [ ] `experiments/results/` directory — must be created
- [ ] `uv add matplotlib scikit-learn` — visualization deps not yet installed

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|-----------------|--------|
| Train.py saves checkpoint .npz | train.py only writes JSON metrics; no weights saved | Embedding extraction requires in-process extraction or wte save at end of training |
| Time-budget loop (`TIME_BUDGET = 300`) | Must replace with step-count loop for iso-data | Add `MAX_STEPS` env var or patch |
| `mx.random.seed(42)` hardcoded | Must vary per run for 3-seed experiments | Add `AUTORESEARCH_SEED` env var |
| `evaluate_bpb` uses byte denominator | BPBL uses base-letter denominator | New function `evaluate_bpbl()` needed |

---

## Open Questions

1. **Exact base letters per word in val set**
   - What we know: ~4.5 letters/word estimated; exact count requires scanning val shard
   - What's unclear: fraction with vs. without harakah (D1 has more codepoints per letter)
   - Recommendation: measure `count_val_base_letters / eval_words` from D1 val shard at experiment start

2. **D1 singleton harakah tokens**
   - What we know: BPE may merge harakah with base letters; not all harakah are singleton tokens
   - What's unclear: how many of the 9 harakah types exist as singleton tokens in D1's 8192-vocab tokenizer
   - Recommendation: enumerate via `tokenizer.enc.encode_single_token(char)` and report coverage

3. **Intra-group cosine similarity threshold**
   - What we know: low similarity = D3 fails to learn compositionality; high = it does
   - What's unclear: what threshold distinguishes "good" from "bad" compositionality learning
   - Recommendation: compare to D1's harakah token clustering as a baseline; no fixed threshold needed for a mechanistic narrative

---

## Sources

### Primary (HIGH confidence)
- Direct inspection of `prepare.py`, `build_dataset.py`, `train.py` in project root
- Direct inspection of `~/.cache/autoresearch-arabic/` filesystem
- Direct inspection of `search_results.json`, `baseline_results.json`, `fertility_report.json`
- MLX `nn.Module.load_weights` docs (verified via `help()` in project venv)
- `mx.savez`/`mx.save_safetensors`/`nn.Module.load_weights` signatures (verified live)

### Secondary (MEDIUM confidence)
- `paper/run_fixed_arch_ablation.py` — subprocess driver pattern
- `paper/fixed_architecture_ablation.json` — confirmed D1 < D3 < D2 under all shared architectures
- `tests/conftest.py` — confirmed fixture patterns and PUA encoding logic

---

## Metadata

**Confidence breakdown:**
- BPBL implementation: HIGH — code is a direct extension of existing `evaluate_bpb`; all hooks exist
- Embedding extraction: HIGH — MLX `wte.weight` access verified; numpy conversion straightforward
- Iso-data scaling: HIGH — arithmetic verified; step counts consistent with CONTEXT estimate
- Checkpoint existence: HIGH — confirmed by filesystem scan (no .npz or .safetensors anywhere)
- Visualization stack: MEDIUM — matplotlib/sklearn not yet installed; standard APIs, low risk

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable codebase, no fast-moving deps)
