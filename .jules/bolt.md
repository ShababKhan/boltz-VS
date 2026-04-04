## 2024-03-24 - [MSA Unique Sequences deduplication Bottleneck]
**Learning:** Found a major performance bottleneck in `src/boltz/data/msa/mmseqs2.py` where O(N²) list membership checks (`x not in seqs_unique`) and `.index(seq)` lookups were used for finding unique sequences and their indexes. This can be extremely slow for large inputs, like MSA alignments.
**Action:** Always avoid O(N²) lookups for deduplication and indexing, especially for potentially large lists. Use `list(dict.fromkeys(seqs))` for ordered deduplication and a hash map (dictionary) for index lookups which are O(N) instead.
