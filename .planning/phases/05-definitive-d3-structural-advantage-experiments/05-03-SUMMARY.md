---
phase: 05-definitive-d3-structural-advantage-experiments
plan: "03"
subsystem: experiments
tags: [python, training, bpbl, scaling, iso-data, d1, d3, arabic-nlp]

# Dependency graph
requires:
  - phase: 05-01
    provides: "train.py AUTORESEARCH_MAX_STEPS + AUTORESEARCH_SEED env vars, experiments/shared.py, base_letters_per_token calibration"
  - phase: 03-architecture-search
    provides: "D1 best config (commit ab075a6) and D3 best config (commit 532b0d1)"
provides:
  - "30-run iso-data experiment: 2 conditions x 5 budgets x 3 seeds"
  - "experiments/results/iso_data_results.json with calibration, per-run stats, and budget-level summary"
  - "experiments/results/iso_data_scaling_curves.png and iso_data_scaling_curves_log.png"
  - "30 per-run training logs in experiments/results/iso_data_logs/"
affects:
  - "paper — scaling curves are a primary figure in the paper"
  - "04-analysis-paper — gap-closing trend contradicts static 'D3 wins' framing"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Iso-data budget calibration: convert letter budget to max_steps via base_letters_per_token derived from fertility and val-shard statistics"
    - "Read-update-write JSON for incremental result preservation across 30 runs"
    - "baseline_results.json backup/restore in try/finally to protect Phase 2 baseline records"

key-files:
  created:
    - "experiments/exp5_iso_data.py"
    - "experiments/results/iso_data_results.json"
    - "experiments/results/iso_data_scaling_curves.png"
    - "experiments/results/iso_data_scaling_curves_log.png"
    - "experiments/results/iso_data_logs/ (30 per-run training logs)"
  modified: []

key-decisions:
  - "D1 wins at every iso-data budget from 5M to 100M base letters — the compositional representation advantage holds across data scale"
  - "Gap closes with data: D1-D3 BPBL gap 1.025 at 5M, narrows to 0.135 at 100M — suggests D3 may close further with more data"
  - "Each condition uses its own optimal architecture (not a shared reference), per prior user decision in Plan 01"

patterns-established:
  - "Iso-data calibration: base_letters_per_step = TOTAL_BATCH_SIZE * base_letters_per_token; max_steps = int(budget / base_letters_per_step)"
  - "Graceful failure handling for D3 crash rate: non-zero subprocess exit -> warning + continue, not raise"

requirements-completed: []

# Metrics
duration: 10min (execution verification + commit; overnight runs pre-completed)
completed: 2026-03-16
---

# Phase 5 Plan 03: Iso-data Scaling Curves (Experiment 5) Summary

**30-run iso-data experiment confirms D1 BPBL advantage over D3 across all five data budgets (5M-100M base letters), with gap closing from 1.025 at 5M to 0.135 at 100M**

## Performance

- **Duration:** 10 min (verification + commits; 30 training runs ran overnight)
- **Started:** 2026-03-16T00:00:00Z
- **Completed:** 2026-03-16T11:30:00Z
- **Tasks:** 2 of 2
- **Files modified:** 33 (1 script + 1 JSON + 2 PNGs + 30 logs)

## Accomplishments

- Verified all 30 training runs complete (2 conditions x 5 budgets x 3 seeds), all BPBL values plausible
- Confirmed plan verification commands pass: 30 runs present, 10 condition-budget combos, 3 seeds each
- Committed experiment driver, results JSON, 30 training logs, and both scaling curve plots

## Results Summary

| Budget (BL) | D1 BPBL (mean +/- std) | D3 BPBL (mean +/- std) | Gap   | Winner |
|-------------|------------------------|------------------------|-------|--------|
| 5M          | 4.0599 +/- 0.0570      | 5.0846 +/- 0.1153      | 1.025 | D1     |
| 15M         | 3.2325 +/- 0.0234      | 3.8456 +/- 0.0701      | 0.613 | D1     |
| 30M         | 2.9282 +/- 0.0109      | 3.1539 +/- 0.0368      | 0.226 | D1     |
| 50M         | 2.7473 +/- 0.0683      | 2.9139 +/- 0.0099      | 0.167 | D1     |
| 100M        | 2.5883 +/- 0.0336      | 2.7233 +/- 0.0184      | 0.135 | D1     |

Calibration: base_letters_per_word=3.7897, D1 BL/token=1.5045, D3 BL/token=1.7278

## Task Commits

Each task was committed atomically:

1. **Task 1: Iso-data experiment driver + 30-run results** - `1463b96` (feat)
2. **Task 2: Scaling curve visualizations** - `bea39f6` (feat)

## Files Created/Modified

- `experiments/exp5_iso_data.py` — Full 30-run orchestration driver with step calibration, patching loop, graceful failure, incremental JSON writes, plotting
- `experiments/results/iso_data_results.json` — All 30 data points: calibration, per-run stats (condition, budget, seed, bpbl, actual_base_letters, etc.), and budget-level summary
- `experiments/results/iso_data_scaling_curves.png` — BPBL vs data budget (linear x-axis, 239KB)
- `experiments/results/iso_data_scaling_curves_log.png` — BPBL vs data budget (log x-axis, 264KB)
- `experiments/results/iso_data_logs/` — 30 training logs (d1/d3 x 5M/15M/30M/50M/100M x seed42/seed137/seed2024)

## Decisions Made

- D1 compositional advantage confirmed at all data scales up to 100M base letters
- Gap-closing trend (1.025 -> 0.135) is the most important new finding: the paper must frame this as "D1 advantage is real but may diminish at large scale" rather than a static ordering
- Results support Phase 5 hypothesis that structural advantage exists; the scaling behavior adds nuance about when it matters most

## Deviations from Plan

None — plan executed exactly as written. Experiment ran overnight and all artifacts were present for verification. Both plan verification commands passed on first run.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 5 is now complete (Plans 01, 02, 03 all done)
- All three mechanistic experiments are finished: BPBL metric (01), embedding analysis (02), iso-data scaling (03)
- The iso-data gap-closing trend is a critical paper figure — the paper draft should be updated to discuss convergence at scale
- No blockers for final paper revision

---
*Phase: 05-definitive-d3-structural-advantage-experiments*
*Completed: 2026-03-16*
