## 2024-05-18 - Optimized feature combination for HTVS mode
**Learning:** In the `combine_feats` logic, allocating `torch.zeros()` with dimensions size `N_max` and then copying `N_p` and `N_l` fragments into it creates severe CPU / memory overhead when high-dimensional tensors (like MSA features `[B, N, L, C]`) are heavily padded. This padding operation was happening redundantly for almost all keys even when `N_p == N_max` and `N_l == N_max` (the most common case, since usually ligand `N_l` is 1 and `N_p` is `N_max`). By unconditionally running `torch.zeros`, execution time per molecule grew to ~0.3s.
**Action:** Always condition sequence block zero-allocations and `F.pad()` logic on `if N < N_max`, falling back directly to `torch.cat([fp, fl], dim=2)` when unnecessary padding can be skipped. This speeds up HTVS processing significantly (by ~80%).

## 2025-02-28 - Avoided O(N^2) list lookups in MSA processing
**Learning:** Deduplicating and tracking index positions for large Multiple Sequence Alignments (MSA) using list membership (`item not in list`) and `.index()` methods causes significant O(N^2) performance bottlenecks.
**Action:** Always precompute a dictionary mapping (`{seq: i for i, seq in enumerate(seqs_unique)}`) or use `list(dict.fromkeys(seqs))` for deduplication to maintain O(1) lookups and O(N) processing time in high-frequency loops.
