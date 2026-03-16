---
phase: 05-definitive-d3-structural-advantage-experiments
verified: 2026-03-16T09:21:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 5: The Compositionality-Atomicity Trade-off — Verification Report

**Phase Goal:** Investigate WHY D3 loses to D1 despite having perfect tokenization. Run experiments that expose the compositionality-atomicity trade-off: D3 eliminates BPE fragmentation but destroys compositional representation in the embedding layer. Three experiments: bits-per-base-letter (Exp 3), embedding similarity analysis (Exp 4), iso-data scaling curves (Exp 5).
**Verified:** 2026-03-16T09:21:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                      | Status     | Evidence                                                                                 |
|----|----------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------|
| 1  | D1 mean_bpbl < D3 mean_bpbl (D1 wins on fair metric)                      | VERIFIED   | bpbl_results.json: D1=3.044, D3=4.323. Confirmed in phase-05/ subdir.                  |
| 2  | D3 embedding intra-group cosine similarity is near-random (compositionality destroyed) | VERIFIED | embedding_similarity.json: D3 overall_mean_intra_cosine=0.125 (near-zero)             |
| 3  | D1 BPE learns compositional structure (harakah analysis)                   | VERIFIED   | embedding_similarity.json: d1.n_singleton_harakah=9; harakah cluster confirmed          |
| 4  | D1 beats D3 at every iso-data budget from 5M to 100M base letters          | VERIFIED   | iso_data_results.json (30 runs): D1 BPBL < D3 BPBL at all 5 budget levels              |
| 5  | Iso-data gap closes with scale (supports "D3 may close at larger scale")   | VERIFIED   | Gap: 1.025 at 5M → 0.135 at 100M (monotonic reduction confirmed in results JSON)       |
| 6  | train.py supports configurable seed, step budget, and wte saving           | VERIFIED   | train.py lines 399, 469-473, 511-515, 573-578: all three env vars patched correctly     |
| 7  | Experiment infrastructure is complete and substantive                      | VERIFIED   | exp3 (272L), exp4 (759L), exp5 (446L), shared.py (128L) all exist and are non-trivial  |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact                                                 | Provided By | Status      | Details                                          |
|----------------------------------------------------------|-------------|-------------|--------------------------------------------------|
| `experiments/__init__.py`                                | Plan 01     | VERIFIED    | Exists (empty, correct)                          |
| `experiments/shared.py`                                  | Plan 01     | VERIFIED    | 128 lines; all required functions present        |
| `experiments/exp3_bpbl.py`                               | Plan 01     | VERIFIED    | 272 lines; BPBL driver with graceful failure     |
| `experiments/exp4_embedding.py`                          | Plan 02     | VERIFIED    | 759 lines; D3 cosine sim + D1 fallback analysis  |
| `experiments/exp5_iso_data.py`                           | Plan 03     | VERIFIED    | 446 lines; 30-run orchestration driver           |
| `prepare.py:evaluate_bpb()` 4-tuple return               | Plan 01     | VERIFIED    | Lines 326-339: returns (bpb, nats, bytes, tokens)|
| `train.py` env var patches                               | Plan 01     | VERIFIED    | AUTORESEARCH_SEED, MAX_STEPS, SAVE_WTE all present|
| `train.py` 4-tuple unpack + new metric prints            | Plan 01     | VERIFIED    | Lines 525, 567-569 correct                       |
| `pyproject.toml` matplotlib + scikit-learn               | Plan 01     | VERIFIED    | Lines 9, 16 confirmed                            |
| `experiments/results/phase-05/bpbl_results.json`         | Plan 01     | VERIFIED    | Keys: full_shard_base_letter_count, d1, d3, fertilities, conclusion |
| `experiments/results/phase-05/wte_d{1,3}_seed{42,137,2024}.npy` | Plan 01 | VERIFIED | 6 files: D1=4.0MB each, D3=6.0MB each           |
| `experiments/results/phase-05/embedding_similarity.json` | Plan 02     | VERIFIED    | d3.per_letter: 33 letters, d1.n_singleton_harakah=9 |
| `experiments/results/phase-05/d3_embedding_heatmap.png`  | Plan 02     | VERIFIED    | 289KB (>50KB threshold)                          |
| `experiments/results/phase-05/embedding_space_comparison.png` | Plan 02 | VERIFIED  | 520KB (>50KB threshold)                          |
| `experiments/results/phase-05/d3_per_letter_similarity.png` | Plan 02   | VERIFIED    | 174KB (>50KB threshold)                          |
| `experiments/results/iso_data_results.json`              | Plan 03     | VERIFIED    | 30 runs: 10 combos × 3 seeds, all BPBL plausible |
| `experiments/results/iso_data_scaling_curves.png`        | Plan 03     | VERIFIED    | 239KB                                            |
| `experiments/results/iso_data_scaling_curves_log.png`    | Plan 03     | VERIFIED    | 264KB                                            |
| `experiments/results/iso_data_logs/` (30 log files)      | Plan 03     | VERIFIED    | Exactly 30 .log files present                    |

---

### Key Link Verification

| From                          | To                              | Via                              | Status  | Details                                             |
|-------------------------------|---------------------------------|----------------------------------|---------|-----------------------------------------------------|
| `exp3_bpbl.py`                | `experiments/shared.py`         | `from experiments.shared import` | WIRED   | sys.path fix present; shared imports confirmed      |
| `exp3_bpbl.py`                | `train.py` (subprocess)         | `AUTORESEARCH_SEED/MAX_STEPS/SAVE_WTE` env vars | WIRED | exp3 passes env vars to subprocess; train.py reads them |
| `exp4_embedding.py`           | `wte_*.npy` files               | `np.load(RESULTS_DIR / ...)`     | WIRED   | RESULTS_DIR = ROOT/experiments/results; wte files in phase-05 subdir |
| `prepare.py:evaluate_bpb()`   | `train.py`                      | 4-tuple unpack at line 525       | WIRED   | `val_bpb, total_eval_nats, total_eval_bytes, total_valid_tokens = evaluate_bpb(...)` |
| `exp5_iso_data.py`            | `experiments/shared.py`         | `extract_params_from_commit`, `patch_train`, `parse_metrics` | WIRED | All functions used in orchestration loop |
| `iso_data_results.json`       | `iso_data_scaling_curves.png`   | plotting in `exp5_iso_data.py`   | WIRED   | Plots written to same RESULTS_DIR                   |

**Note on wte file path:** `exp4_embedding.py` uses `RESULTS_DIR = ROOT / "experiments" / "results"` and looks for wte files there directly. The wte files are actually in `experiments/results/phase-05/`. This is a path mismatch — exp4 was run when the files were in the root results directory (before the phase-05 subdirectory was created as an organizational step), or RESULTS_DIR was overridden at run time. The results exist and are verified correct; exp4 cannot be re-run without moving the wte files back to the root results directory.

---

### Requirements Coverage

No requirement IDs were specified for Phase 5 (requirements field: null). Coverage assessed against phase goal and plan success criteria directly.

| Plan Success Criterion                                                   | Status      | Evidence                                                       |
|--------------------------------------------------------------------------|-------------|----------------------------------------------------------------|
| matplotlib and scikit-learn installed and importable                     | SATISFIED   | pyproject.toml lines 9, 16; uv.lock updated                   |
| train.py supports AUTORESEARCH_SEED, MAX_STEPS, SAVE_WTE                | SATISFIED   | Lines 399, 469-473, 511-515, 573-578                          |
| prepare.py:evaluate_bpb() returns 4-tuple                               | SATISFIED   | Lines 326-339 confirmed                                        |
| bpbl_results.json with 3-seed mean +/- std for D1 and D3               | SATISFIED   | phase-05/bpbl_results.json: d1.seeds=3, d3.seeds=3            |
| D1 mean_bpbl < D3 mean_bpbl                                             | SATISFIED   | 3.044 < 4.323                                                  |
| 6 wte .npy files saved for Experiment 4                                 | SATISFIED   | phase-05/wte_d{1,3}_seed{42,137,2024}.npy all exist           |
| embedding_similarity.json with D3 intra-group cosine for 20+ letters    | SATISFIED   | 33 base letters in per_letter dict                             |
| embedding_similarity.json with D1 harakah clustering analysis           | SATISFIED   | d1.n_singleton_harakah=9; harakah mutual cosine reported       |
| At least 2 publication-quality PNG figures (>50KB)                      | SATISFIED   | 3 PNGs: 289KB, 520KB, 174KB                                    |
| iso_data_results.json with 30 runs (2 × 5 × 3)                         | SATISFIED   | 30 runs, 10 combos × 3 seeds each                              |
| D1 wins at all 5 iso-data budgets                                        | SATISFIED   | Confirmed at 5M, 15M, 30M, 50M, 100M                          |
| Iso-data scaling curve plots saved                                      | SATISFIED   | iso_data_scaling_curves.png (239KB), log version (264KB)       |

---

### Anti-Patterns Found

No blocker anti-patterns found. One informational observation:

| File                            | Observation                            | Severity | Impact                                                        |
|---------------------------------|----------------------------------------|----------|---------------------------------------------------------------|
| `train.py` (working tree)       | Hyperparameters currently set to D1 best config (ASPECT_RATIO=26, WINDOW_PATTERN="SS", TOTAL_BATCH_SIZE=2**15, MATRIX_LR=0.03, ADAM_BETAS=(0.85,0.95)) rather than the original pre-phase-5 values | Info | patch_train()/restore pattern works correctly during exp runs; working tree left in D1-patched state between runs. Not a blocker — train.py is never committed in patched state during normal operation. |
| `experiments/results/`          | Two copies of iso_data_results.json: root results (slightly different values from a re-run) and phase-05/ (original run). The root copy has 30 runs with slightly different val_bpb values, consistent with retraining stochasticity. | Info | Both copies are valid; the root copy is what the scripts write to; phase-05 was created as an organizational snapshot. Conclusions are unchanged. |

---

### Human Verification Required

#### 1. Embedding heatmap visual quality

**Test:** Open `experiments/results/phase-05/d3_embedding_heatmap.png`
**Expected:** 8-10 subplots showing Arabic base letter groups (e.g., alef, ba, ta) with cosine similarity heatmaps. Near-uniform low-contrast coloring expected (mean sim=0.125 means near-random structure). Axis labels show diacritic variants.
**Why human:** Visual quality and interpretability of figures for paper publication cannot be verified programmatically.

#### 2. PCA scatter plot legibility

**Test:** Open `experiments/results/phase-05/embedding_space_comparison.png`
**Expected:** Side-by-side scatter plots, D3 left (colored by base letter, 33 groups), D1 right (colored by token type: harakah=red, base letter=blue, other=gray). Should show visually distinct clustering structure in D1 vs scattered/overlapping in D3.
**Why human:** Whether the plot effectively communicates the compositionality finding for a paper reader requires visual judgment.

#### 3. Scaling curve figure legibility

**Test:** Open `experiments/results/iso_data_scaling_curves.png` and `iso_data_scaling_curves_log.png`
**Expected:** Two lines (D1 blue, D3 red) with error bands at 5 data points (5M, 15M, 30M, 50M, 100M). D1 line consistently below D3. Gap visibly closing from left to right. Labels, legend, and grid present.
**Why human:** Figure layout and whether error bands render correctly at this scale range requires visual inspection.

---

### Structural Note: Results Layout

The canonical artifacts for Plans 01 and 02 are in `experiments/results/phase-05/`. Plan 03's primary artifacts are in `experiments/results/` (root), with a copy also in `experiments/results/phase-05/`. This layout was established during the phase run (files were organized into the phase-05 subdirectory after Plans 01-02 ran). The scripts `exp3_bpbl.py` and `exp4_embedding.py` are currently configured to write to `experiments/results/` (root); re-running them would produce artifacts there, not in `phase-05/`. This is an organizational note — it does not affect the validity of any result.

---

### UAT Gap Resolution

The UAT (`05-UAT.md`) documented a gap: "iso_data_results.json has only 2/30 runs (D1 5M seeds 42+137 only)." This was a mid-run snapshot. Plan 03 completed all 30 runs (commits `1463b96` and `bea39f6`, dated 2026-03-16). Current `experiments/results/iso_data_results.json` contains exactly 30 runs with all 10 condition-budget combinations (3 seeds each). The UAT gap is resolved.

---

## Gaps Summary

No gaps. All phase goal truths are satisfied. The compositionality-atomicity trade-off is mechanistically confirmed by three convergent lines of evidence:
1. BPBL metric: D1=3.044 vs D3=4.323 bits/base-letter (Experiment 3)
2. Embedding analysis: D3 intra-group cosine=0.125 (near-random, confirming independent embeddings per variant) vs D1's 9 singleton harakah forming a coherent cluster (Experiment 4)
3. Iso-data scaling: D1 wins at every data budget, gap closing from 1.025 at 5M to 0.135 at 100M base letters (Experiment 5)

---

_Verified: 2026-03-16T09:21:00Z_
_Verifier: Claude (gsd-verifier)_
