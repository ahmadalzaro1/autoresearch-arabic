# Roadmap: autoresearch-arabic

## Overview

Four phases take the experiment from raw dataset through publishable paper. Phase 1 builds the three dataset conditions and quantifies the homograph disambiguation tax. Phase 2 trains condition-specific tokenizers and establishes baselines. Phase 3 runs the overnight autoresearch loop across all three conditions. Phase 4 extracts the comparative story and writes the paper.

## Phases

- [x] **Phase 1: Data Pipeline** - Download dataset, produce D1/D2/D3 parquet shards, compute homograph collision statistics (completed 2026-03-12)
- [x] **Phase 2: Tokenizer & Baseline** - Train BPE tokenizers per condition, measure fertility, run baseline val_bpb training (completed 2026-03-12)
- [x] **Phase 3: Architecture Search** - Run overnight autoresearch agent loop for D1, D2, D3 (70+ experiments each) (completed 2026-03-13)
- [x] **Phase 4: Analysis & Paper** - Compare winning architectures, run ablations, write paper draft (completed 2026-03-13)
- [x] **Phase 5: Definitive D3 Structural Advantage Experiments** - BPBL metric, embedding similarity analysis, iso-data scaling curves (completed 2026-03-16)

## Phase Details

### Phase 1: Data Pipeline
**Goal**: All three dataset conditions exist as validated parquet shards and homograph collision statistics quantify the disambiguation tax
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. D1, D2, and D3 parquet shards are on disk and load without error
  2. Token counts and character distributions are verified for each shard (no silent truncation or encoding corruption)
  3. Homograph collision rate is computed and logged — ألم-style ambiguities counted at corpus scale
  4. D3 PUA mapping table is complete (all letter+harakah combinations covered, ~252 combos)
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Test scaffold: install pytest, create tests/ with conftest fixtures and six stub tests
- [ ] 01-02-PLAN.md — Extend build_dataset.py: tqdm download wrapper + context-window collision metric + JSON sidecar
- [x] 01-03-PLAN.md — Add inline validation to build_dataset.py + create validate_dataset.py standalone validator

### Phase 2: Tokenizer & Baseline
**Goal**: Each condition has a trained BPE tokenizer and a measured baseline val_bpb on identical architecture, establishing the benchmark the search will beat
**Depends on**: Phase 1
**Requirements**: TOK-01, TOK-02, TOK-03, TOK-04, BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Three BPE tokenizers trained (one per condition), each at multiple vocab sizes
  2. Fertility table (tokens/word × condition × vocab size) is computed and shows measurable difference between D1/D2/D3
  3. Baseline val_bpb recorded for D1, D2, D3 on fixed depth=4 architecture
  4. D2 baseline is lower than D1 baseline (stripping reduces surface complexity — expected sanity check)
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Wave 0 test stubs: conftest tiny_corpus_parquet fixture + test_tokenizer.py (RED) + test_baseline.py (RED)
- [ ] 02-02-PLAN.md — Extend prepare.py (--vocab-size flag + write_fertility_report) + extend train.py (baseline_results.json writer)
- [ ] 02-03-PLAN.md — Run tokenizer sweep (9 runs) + baseline training (3 x 300s) + human verify fertility and D2 < D1 sanity check

### Phase 3: Architecture Search
**Goal**: Autoresearch has run 70+ experiments per condition overnight and best-performing architecture configs per condition are identified
**Depends on**: Phase 2
**Requirements**: SRCH-01, SRCH-02, SRCH-03
**Success Criteria** (what must be TRUE):
  1. D1, D2, and D3 overnight runs each complete with 70+ experiments logged
  2. Best val_bpb per condition is below the Phase 2 baseline (search found improvements)
  3. Winning architecture configs (depth, heads, window) are recorded per condition
  4. Search results are stored in condition-labeled output directories for reproducibility
**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Wave 0 test scaffold: tests/test_search.py with four smoke tests (skips pre-run, passes post-run)
- [x] 03-02-PLAN.md — D3 overnight run: branch autoresearch/arabic-d3, 70+ experiments, results_d3.tsv, d3 entry in search_results.json
- [x] 03-03-PLAN.md — D1 overnight run: branch autoresearch/arabic-d1, 70+ experiments, results_d1.tsv, d1 entry in search_results.json
- [x] 03-04-PLAN.md — D2 overnight run: branch autoresearch/arabic-d2, 70+ experiments, results_d2.tsv, complete search_results.json

### Phase 4: Analysis & Paper
**Goal**: A first paper draft exists that explains the D1 < D3 < D2 result quantitatively, with comparison tables, a fixed-architecture robustness ablation, and a revised narrative around the ambiguity tax of stripping harakat
**Depends on**: Phase 3
**Requirements**: ANLZ-01, ANLZ-02, ANLZ-03
**Success Criteria** (what must be TRUE):
  1. Cross-condition comparison table shows D1, D3, and D2 best val_bpb under the same 5-minute budget, plus baseline deltas and winning architectures
  2. Analysis includes winning-architecture behavior across conditions plus a fixed-architecture transfer matrix showing whether the ordering survives when architecture is held constant
  3. Paper draft (LaTeX or Markdown) contains abstract, introduction, method, results, limitations, and conclusion
  4. Paper uses the Phase 1 collision statistic and Phase 2 fertility table to support the ambiguity-tax argument, while explicitly discussing why D1 beats D3 in this setup
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md — Synthesize Phase 1-3 evidence into comparison tables, revised paper narrative, and paper outline
- [x] 04-02-PLAN.md — Run targeted Phase 4 ablations to test robustness of the D1 < D3 < D2 ordering
- [x] 04-03-PLAN.md — Draft the paper in Markdown/LaTeX with figures, tables, and limitations

### Phase 5: Definitive D3 Structural Advantage Experiments
**Goal:** Investigate WHY D3 loses to D1 despite having perfect tokenization. Run experiments that expose the compositionality--atomicity trade-off: D3 eliminates BPE fragmentation but destroys compositional representation in the embedding layer.
**Depends on:** Phase 4
**Requirements**: EXP3-01, EXP3-02, EXP4-01, EXP4-02, EXP5-01, EXP5-02
**Success Criteria** (what must be TRUE):
  1. BPBL metric computed for D1 and D3 (3 seeds each) confirms D1 beats D3 on the fair, un-gameable metric
  2. Embedding similarity analysis shows D3 fails to learn compositional representations (low intra-group cosine similarity for PUA embeddings)
  3. D1 harakah tokens cluster separately from base letters, demonstrating compositional structure
  4. Iso-data scaling curves at 5 data budgets show how the D1 vs D3 gap evolves with data scale
  5. Publication-quality figures (heatmaps, scatter plots, scaling curves) ready for paper revision
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — Setup + BPBL metric: install deps, patch train.py, run 6 training jobs, compute bits-per-base-letter
- [ ] 05-02-PLAN.md — Embedding similarity analysis: cosine similarity heatmaps, harakah clustering, t-SNE/PCA scatter plots
- [ ] 05-03-PLAN.md — Iso-data scaling curves: train at 5 data budgets, plot BPBL vs base-letters-processed

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 3/3 | Complete   | 2026-03-12 |
| 2. Tokenizer & Baseline | 3/3 | Complete   | 2026-03-12 |
| 3. Architecture Search | 4/4 | Complete   | 2026-03-13 |
| 4. Analysis & Paper | 3/3 | Complete   | 2026-03-13 |
| 5. D3 Structural Experiments | 3/3 | Complete   | 2026-03-16 |
