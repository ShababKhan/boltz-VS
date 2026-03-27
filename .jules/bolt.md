
## 2024-03-24 - [Avoid List Membership/Index in Loops]
**Learning:** Using `item not in list` and `list.index(item)` inside loops for deduplication and indexing creates an O(N²) performance bottleneck, particularly noticeable in processing large sets like sequences for MSAs.
**Action:** Use an O(N) hash map approach with a dictionary to store seen items and their indices to achieve significant speedups for large collections.
