
## 2024-03-01 - Exception Suppression in Gemmi Parsing is a Bottleneck
**Learning:** Using `contextlib.suppress(Exception)` with `gemmi.cif.Block.find([key])[0]` to handle missing keys in MMCIF parsing is extremely slow due to the overhead of creating and catching exceptions (specifically `IndexError` when the key doesn't exist). In tight loops or large structures, this becomes a significant bottleneck.
**Action:** When querying `gemmi.cif.Block` for metadata or attributes, always use explicit bounds checking (`len(block.find([key])) > 0`) before accessing the first element. Use `try...except ValueError` strictly for safe type conversion, never for control flow on missing keys.
