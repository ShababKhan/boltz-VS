## 2025-02-23 - [O(N²) Array Index Lookups During Feature Symmetries Computation]
**Learning:** In the Boltz architecture, extracting chain symmetries involves iterating over `cropped.tokens` and checking `chain_asym_id.index(token["asym_id"])` to get an index. This introduces an O(N^2) complexity bottleneck.
**Action:** Always precompute a dictionary mapping (e.g. `chain_asym_id_to_idx = {asym_id: i for i, asym_id in enumerate(chain_asym_id)}`) before iterating over tokens to maintain O(1) lookups for list matching.
