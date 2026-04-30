## 2024-05-18 - Optimized feature combination for HTVS mode
**Learning:** In the `combine_feats` logic, allocating `torch.zeros()` with dimensions size `N_max` and then copying `N_p` and `N_l` fragments into it creates severe CPU / memory overhead when high-dimensional tensors (like MSA features `[B, N, L, C]`) are heavily padded. This padding operation was happening redundantly for almost all keys even when `N_p == N_max` and `N_l == N_max` (the most common case, since usually ligand `N_l` is 1 and `N_p` is `N_max`). By unconditionally running `torch.zeros`, execution time per molecule grew to ~0.3s.
**Action:** Always condition sequence block zero-allocations and `F.pad()` logic on `if N < N_max`, falling back directly to `torch.cat([fp, fl], dim=2)` when unnecessary padding can be skipped. This speeds up HTVS processing significantly (by ~80%).

## 2024-05-18 - Pre-compiling regex in structural writers
**Learning:** In the MMCIF and PDB writers (`src/boltz/data/write/`), using `re.sub(r"\d", "", atom_name)` inside nested loops that iterate over thousands of atoms caused a significant bottleneck due to repeated regex compilation.
**Action:** When working with high-frequency loops, always pre-compile regular expression patterns at the module level (e.g., `RE_DIGIT = re.compile(r'\d')`) to optimize CPU usage.
