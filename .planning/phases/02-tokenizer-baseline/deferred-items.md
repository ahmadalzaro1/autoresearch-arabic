## Deferred Items (Out of Scope)

### Pre-existing test failure: test_load_dataset_with_progress_keyboard_interrupt

**Discovered during:** Phase 02-02 execution
**Location:** tests/test_pipeline.py::test_load_dataset_with_progress_keyboard_interrupt
**Root cause:** The test's `fake_load2` monkeypatches `datasets.load_dataset` without the `data_files` keyword argument, but `build_dataset.load_dataset_with_progress` now has a local-dataset shortcut path (added post-Plan 01) that calls `_load("parquet", data_files=...)` — this path is hit first because `arabic-tashkeel-dataset/` directory exists on disk.
**Status:** Pre-existing. Confirmed failing before 02-02 changes.
**Fix:** Update test to either remove the local dataset directory during the test or monkeypatch to also accept `data_files` keyword argument.
