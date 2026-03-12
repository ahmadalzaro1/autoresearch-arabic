# autoresearch-arabic

## What This Is

A mechanistic research experiment studying how Arabic diacritical marks (harakat) should be encoded for small language model pretraining. Using Karpathy's autoresearch framework on Apple Silicon MLX, it runs autonomous overnight architecture searches across three dataset conditions to produce a publishable ML paper targeting ArabicNLP/EMNLP/COLM.

## Core Value

Prove that stripping harakat from Arabic text forces models to waste capacity on implicit disambiguation — and that atomic diacritical encoding (D3) closes this gap for small models without the information loss of stripping (D2).

## Requirements

### Validated

- ✓ Dataset pipeline: D1/D2/D3 parquet shards produced — Phase 1
- ✓ Homograph collision statistics at corpus scale — Phase 1
- ✓ Tokenizer fertility analysis: D1=2.519, D2=1.467, D3=2.193 tokens/word @ 8192 vocab — Phase 2
- ✓ Baseline val_bpb per condition: D3=1.075, D1=1.191, D2=1.597 (depth=4, SSSL) — Phase 2

### Active

- [ ] Overnight autoresearch run per condition (70+ experiments each)
- [ ] Optimal architecture comparison across D1 vs D2 vs D3
- [ ] Paper draft targeting ArabicNLP Workshop or EMNLP

### Out of Scope

- Production Arabic LLM — this is a research experiment, not a deployable model
- Large models (>100M params) — 5-minute fixed budget means small models only
- Dialect Arabic — experiment uses classical Arabic (Tashkeela corpus)
- Fine-tuning / RLHF — pretraining only

## Context

- **Framework**: Fork of trevin-creator/autoresearch-mlx (Karpathy's autoresearch on Apple Silicon)
- **Hardware**: Apple Silicon Mac with MLX (unified memory, no PyTorch)
- **English baseline**: val_bpb 1.815 at depth=4, 11.5M params (autoresearch-mlx, confirmed working)
- **Dataset**: Abdou/arabic-tashkeel-dataset — 1.5M examples, MIT, paired vocalized/non_vocalized columns
- **Dataset blocker**: HuggingFace downloads stall on current network — needs retry or VPN
- **Three conditions**:
  - D1: Fully diacritized (raw Unicode combining chars) — what nobody trains on
  - D2: Harakat stripped — what every Arabic LLM does today (control)
  - D3: Atomic encoding — letter+harakah pairs mapped to PUA codepoints (novel contribution)
- **Key insight**: Arabic without harakat is ambiguous. ألم = 4 different words (pain/befell/did not/be informed). Models recover meaning from context — wasting capacity. D3 eliminates this tax.
- **Career goal**: Use publications from this work to land a position at TII/G42/Booking.com Amsterdam in Arabic AI

## Constraints

- **Hardware**: Single Mac, Apple Silicon — MLX only, no CUDA
- **Time budget**: 5 min per training run (fixed by autoresearch framework)
- **Network**: HuggingFace downloads currently unreliable — may need Kaggle/SourceForge fallback for Tashkeela
- **Scope**: This milestone = working pipeline + baselines + one overnight run. Paper writing is next milestone.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| MLX fork over PyTorch/MPS fork | Proven results (1.808 baseline), 13x faster eval, no PyTorch | ✓ Working |
| D3 atomic PUA encoding | BPE never splits harakah from letter; ~252 combos fit in PUA range | ✓ D3 bpb=1.075 (best) |
| Abdou/arabic-tashkeel-dataset | MIT license, 1.5M examples, paired columns ready for D1/D2 | ✓ 1.43M train rows loaded |
| EVAL_TOKENS = 3×524288 | MLX fork default — 13x faster than original, good for iteration | ✓ Working |
| Coarse phases | Research experiment, not a product — 3-5 broad phases is right | ✓ On track |
| D1 < D2 on val_bpb | Diacritical marks disambiguate word forms, reducing predictive difficulty | ✓ Confirmed Phase 2 |
| D3 best baseline (1.075) | Atomic PUA encoding packs letter+diacritic into single codepoint — compact, predictable sequences | ✓ Confirmed Phase 2 |
| DEFAULT_VOCAB_SIZE stable constant | Mutable global VOCAB_SIZE corrupted get_dirs() path routing — fixed with immutable constant | ✓ Fixed Phase 2 |

---
*Last updated: 2026-03-12 after Phase 2 (tokenizer-baseline)*
