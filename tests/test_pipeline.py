"""
Pipeline tests for autoresearch-arabic Phase 1.
Tests DATA-01 through DATA-05 + validation report.

Tests using ~/.cache data are skipped until build_dataset.py has run.
Fixture-based tests (D1/D2/D3 shard shape/content) pass immediately.
"""
import json
import pytest
import pyarrow.parquet as pq
from pathlib import Path

BASE_CACHE = Path.home() / ".cache" / "autoresearch-arabic"
HARAKAT_RANGE = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')
PUA_START = 0xE000
PUA_END = 0xF8FF


# ---------------------------------------------------------------------------
# DATA-01: Dataset download (smoke test — requires cache or live network)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires HF cache from build_dataset.py run; run manually")
def test_dataset_columns():
    """Abdou/arabic-tashkeel-dataset has vocalized and non_vocalized columns."""
    from datasets import load_dataset
    ds = load_dataset("Abdou/arabic-tashkeel-dataset", split="train[:10]")
    assert "vocalized" in ds.column_names
    assert "non_vocalized" in ds.column_names
    assert len(ds) == 10


# ---------------------------------------------------------------------------
# DATA-02: D1 shards contain harakat characters
# ---------------------------------------------------------------------------

def test_d1_shards(d1_shard_path):
    """D1 parquet shard loads without error and text column contains harakat."""
    table = pq.read_table(d1_shard_path)
    assert "text" in table.schema.names
    texts = table.column("text").to_pylist()
    assert len(texts) == 100
    assert all(t is not None and t != "" for t in texts)
    # D1 must contain at least one harakat character somewhere
    combined = "".join(texts)
    assert any(c in HARAKAT_RANGE for c in combined), "D1 texts should contain harakat"


# ---------------------------------------------------------------------------
# DATA-03: D2 shards have zero harakat codepoints
# ---------------------------------------------------------------------------

def test_d2_stripped(d2_shard_path):
    """D2 parquet shard has zero harakat codepoints (U+064B-U+0652, U+0670)."""
    table = pq.read_table(d2_shard_path)
    texts = table.column("text").to_pylist()
    assert all(t is not None and t != "" for t in texts)
    combined = "".join(texts)
    assert not any(c in HARAKAT_RANGE for c in combined), "D2 texts must have all harakat stripped"


# ---------------------------------------------------------------------------
# DATA-04: D3 shards contain PUA codepoints; atomic_mapping.json >= 252 entries
# ---------------------------------------------------------------------------

def test_d3_pua(d3_shard_path, cache_base, tmp_path):
    """D3 parquet shard contains PUA codepoints; atomic_mapping.json has >= 252 entries."""
    table = pq.read_table(d3_shard_path)
    texts = table.column("text").to_pylist()
    assert all(t is not None and t != "" for t in texts)
    combined = "".join(texts)
    assert any(PUA_START <= ord(c) <= PUA_END for c in combined), "D3 texts should contain PUA codepoints"

    # Check atomic_mapping.json (requires build_dataset.py to have run)
    mapping_path = BASE_CACHE / "atomic_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        assert len(mapping) >= 252, f"atomic_mapping.json has {len(mapping)} entries, need >= 252"


# ---------------------------------------------------------------------------
# DATA-05: collision_stats.json has top_50_ambiguous and context_window_ambiguous_pct
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires build_dataset.py to run with Wave 2 JSON sidecar")
def test_collision_stats_json():
    """collision_stats.json exists with top_50 ambiguous words and context_window metric."""
    json_path = BASE_CACHE / "collision_stats.json"
    assert json_path.exists(), f"collision_stats.json not found at {json_path}"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "context_window_ambiguous_pct" in data, "Missing context_window_ambiguous_pct field"
    assert "top_50_ambiguous" in data, "Missing top_50_ambiguous field"
    assert len(data["top_50_ambiguous"]) >= 50, "top_50_ambiguous must have >= 50 entries"
    assert "context_window_tokens" in data
    assert data["context_window_tokens"] == 128
    # Verify Arabic characters survive JSON round-trip (ensure_ascii=False)
    for entry in data["top_50_ambiguous"][:3]:
        assert "word" in entry
        assert "variants" in entry
        # Word should contain Arabic characters, not only escape sequences
        assert any('\u0600' <= c <= '\u06FF' for c in entry["word"]), \
            "top_50 words should be Arabic chars, not escape sequences"


# ---------------------------------------------------------------------------
# Integration: validation_report.json all conditions pass
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires build_dataset.py + validate_dataset.py to run (Wave 3)")
def test_validation_report():
    """validation_report.json exists and all per-condition checks pass."""
    report_path = BASE_CACHE / "validation_report.json"
    assert report_path.exists(), f"validation_report.json not found at {report_path}"
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    for condition in ["d1", "d2", "d3"]:
        assert condition in report, f"Missing condition {condition} in report"
        cond_data = report[condition]
        assert cond_data.get("shards_loadable") is True, f"{condition}: shards_loadable check failed"
        assert cond_data.get("row_count_matches") is True, f"{condition}: row_count_matches check failed"
        assert cond_data.get("no_empty_texts") is True, f"{condition}: no_empty_texts check failed"
        assert "char_distribution" in cond_data, f"{condition}: missing char_distribution"
