# Phase 1: Data Pipeline - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Download Abdou/arabic-tashkeel-dataset, produce three validated parquet shard conditions (D1/D2/D3), and compute homograph collision statistics that quantify the D2 disambiguation tax. Training tokenizers and running baselines are Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Download fallback strategy
- HuggingFace is the primary source — no hard fallback to Kaggle in code
- Add tqdm progress bar to the HF download so the user can see if it's stuck
- No timeout: user manually Ctrl+C if stalled; script must print clear manual download instructions on exit
- Auto-detect local HF cache first (`load_dataset` will reuse it); only attempts network download if cache is cold
- HF's built-in caching handles re-run optimization — no extra raw cache layer needed

### Collision statistics scope
- Compute both word-level stats (already implemented) AND 128-token context-window collision probability
- Context-window metric: for a random 128-token window, what fraction of tokens are homographically ambiguous?
- Output: `collision_stats.txt` (human-readable, already exists) + `collision_stats.json` (machine-readable sidecar)
- JSON must include: aggregate stats + top-50 most ambiguous words with all diacritized variants (Phase 4 pulls this into paper tables)

### Validation suite
- Validation runs **inline after each condition** in `build_dataset.py` (auto, no flag needed)
- Validation also available as **standalone `validate_dataset.py`** for re-checking without rebuilding
- Output: print pass/fail per check to stdout + write `~/.cache/autoresearch-arabic/validation_report.json`
- **Mandatory checks (all must pass before Phase 2):**
  1. All shards load without exception
  2. Row count matches `metadata.txt` (train_docs + val_docs)
  3. Zero empty or null texts in any shard
  4. Character distribution sample (top-20 most frequent chars per condition, printed — visual assertion that D1 has harakat, D2 doesn't, D3 has PUA)
- **D3 PUA coverage**: hard-fail if atomic mapping size < 252 entries; this is a data integrity check, not a warning

### Claude's Discretion
- Exact tqdm integration approach and progress bar format
- JSON schema for validation_report.json (as long as it has per-condition pass/fail and the 4 check results)
- How to handle HF `load_dataset` network errors vs. stall (both print the manual instructions)
- Character distribution display format

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_dataset.py`: D1/D2/D3 parquet writing already complete — `process_condition()`, `strip_harakat()`, `apply_atomic_encoding()`, `build_atomic_mapping()` are implemented and tested
- `compute_collision_stats()` in `build_dataset.py`: word-level collision stats already written, needs context-window extension
- `BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"` — established cache convention
- `metadata.txt` per condition — already written with `train_docs`, `val_docs`, `val_shard` fields

### Established Patterns
- `uv run build_dataset.py --condition d1` — CLI pattern for all scripts
- Parquet schema: single `text` column per shard
- Shard naming: `shard_{idx:05d}.parquet`, last shard is val
- Atomic mapping built at runtime (not read from file), saved to `atomic_mapping.json`

### Integration Points
- `prepare.py` reads from `~/.cache/autoresearch-arabic/<condition>/data/` — must exist before Phase 2
- `prepare.py` reads `metadata.txt` to find `val_shard` index — must be correct
- Phase 4 paper generation will read `collision_stats.json` — must exist with top-50 ambiguous words
- `validation_report.json` will be checked by Phase 2 before proceeding

</code_context>

<specifics>
## Specific Ideas

- The 128-token window for context-window collision probability was specifically chosen to match a typical small-model attention window and be explainable in the paper
- top-50 (not top-20) most ambiguous words in JSON — Phase 4 needs examples to put in paper tables
- "Hard-fail below 252" for D3 PUA coverage is an integrity gate, not a metric

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-data-pipeline*
*Context gathered: 2026-03-12*
