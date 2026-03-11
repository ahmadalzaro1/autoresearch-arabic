---
phase: 1
slug: data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-12
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (not yet installed — Wave 0 installs) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `uv run pytest tests/test_pipeline.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~30 seconds (cache-hit mode with fixtures) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_pipeline.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | DATA-01 | smoke | `uv run pytest tests/test_pipeline.py::test_dataset_columns -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 0 | DATA-02 | unit | `uv run pytest tests/test_pipeline.py::test_d1_shards -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 0 | DATA-03 | unit | `uv run pytest tests/test_pipeline.py::test_d2_stripped -x` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 0 | DATA-04 | unit | `uv run pytest tests/test_pipeline.py::test_d3_pua -x` | ❌ W0 | ⬜ pending |
| 1-01-05 | 01 | 1 | DATA-05 | unit | `uv run pytest tests/test_pipeline.py::test_collision_stats_json -x` | ❌ W0 | ⬜ pending |
| 1-01-06 | 01 | 1 | DATA-01..05 | integration | `uv run pytest tests/test_pipeline.py::test_validation_report -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline.py` — stubs for DATA-01 through DATA-05 + validation report
- [ ] `tests/conftest.py` — shared fixtures: tiny parquet shard builder (100 rows of real Arabic text), path helpers
- [ ] `tests/__init__.py` — empty package marker
- [ ] `uv add --dev pytest` — pytest not in pyproject.toml dependencies

*Note: Tests must use a small fixture built from a 100-row subset of D1/D2/D3 parquets (not the full cache) to keep CI independent of the HF download.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| tqdm heartbeat visible during HF download | DATA-01 | Requires live network; cannot automate in fixture tests | Run `uv run python build_dataset.py` on cold cache and verify tqdm progress bar renders |
| Manual download instructions printed on Ctrl+C | DATA-01 | Requires network + keyboard interrupt | Run script, press Ctrl+C mid-download, verify MANUAL_INSTRUCTIONS text appears |
| Character distribution visually shows D1 has harakat, D2 doesn't, D3 has PUA | DATA-02..04 | Visual assertion — no programmatic regex to verify "looks right" | Run validation after build; eyeball top-20 char distribution per condition |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
