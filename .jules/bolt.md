## 2025-02-20 - [O(n) hash map lookup for MSA seq deduplication]
**Learning:** O(n²) bottlenecks easily creep in during list manipulation in tight loops. In `src/boltz/data/msa/mmseqs2.py`, list membership checks `item not in list` coupled with `.index()` lookups were causing an O(N²) slowdown.
**Action:** Replace `list` based deduplication and indexing with `list(dict.fromkeys(seqs))` and a precomputed dictionary `{seq: i for i, seq in enumerate(seqs_unique)}`. This brings the operation down to O(n) and maintains deterministic ordering.
