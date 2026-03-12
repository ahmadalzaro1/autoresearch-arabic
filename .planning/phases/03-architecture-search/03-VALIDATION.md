---
phase: 03
slug: architecture-search
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 (installed) |
| **Config file** | none — uses pyproject.toml project root |
| **Quick run command** | `uv run pytest tests/test_search.py -x -q` |
| **Full suite command** | `uv run pytest tests/test_search.py -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** N/A — no code changes during the search loop; tests run only after each condition's overnight run completes
- **After every plan wave:** `uv run pytest tests/test_search.py -q` after the post-run summary task
- **Before `/gsd:verify-work`:** All four test_search tests green
- **Max feedback latency:** ~5 seconds (post-run, not during loop)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SRCH-01,02,03 | unit stub | `uv run pytest tests/test_search.py -q` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 2 | SRCH-03 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d3 -x` | ✅ W1 | ⬜ pending |
| 03-03-01 | 03 | 3 | SRCH-01 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d1 -x` | ✅ W1 | ⬜ pending |
| 03-04-01 | 04 | 4 | SRCH-02 | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_d2 -x` | ✅ W1 | ⬜ pending |
| 03-04-02 | 04 | 4 | All | smoke (post-run) | `uv run pytest tests/test_search.py::test_search_results_json -x` | ✅ W1 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_search.py` — stubs for SRCH-01, SRCH-02, SRCH-03 + search_results.json schema test; reads TSV files at project root and JSON at project root; skips with helpful message if TSV not yet created (pre-run state)

*Existing infrastructure (`tests/test_baseline.py`, `tests/test_tokenizer.py`, `tests/test_pipeline.py`, `tests/conftest.py`) covers Phases 1–2 only; none cover Phase 3 requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Search loop runs autonomously without pausing | SRCH-01,02,03 | Agent behavior, not file output | Observe that agent commits without prompting during overnight run |
| Best val_bpb is below Phase 2 baseline | SRCH-01,02,03 | Empirical — search may not always beat baseline | Read search_results.json, compare to baseline_results.json |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
