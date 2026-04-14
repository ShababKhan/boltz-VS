## 2024-05-18 - Optimized feature combination for HTVS mode
**Learning:** In the `combine_feats` logic, allocating `torch.zeros()` with dimensions size `N_max` and then copying `N_p` and `N_l` fragments into it creates severe CPU / memory overhead when high-dimensional tensors (like MSA features `[B, N, L, C]`) are heavily padded. This padding operation was happening redundantly for almost all keys even when `N_p == N_max` and `N_l == N_max` (the most common case, since usually ligand `N_l` is 1 and `N_p` is `N_max`). By unconditionally running `torch.zeros`, execution time per molecule grew to ~0.3s.
**Action:** Always condition sequence block zero-allocations and `F.pad()` logic on `if N < N_max`, falling back directly to `torch.cat([fp, fl], dim=2)` when unnecessary padding can be skipped. This speeds up HTVS processing significantly (by ~80%).

## 2025-01-01 - Avoid O(N²) index lookups in MSA parsing and Symmetry computation
**Learning:** O(N²) list `.index()` lookups during MSA sequence deduplication (`seqs_unique.index(seq)`) and feature symmetric calculation lookups (`chain_asym_id.index(token["asym_id"])`) cause significant bottlenecks as sequence and token counts grow.
**Action:** Always precompute a dictionary mapping (`{key: i for i, key in enumerate(keys)}`) before iterating over sequences or tokens to maintain O(1) lookups.
