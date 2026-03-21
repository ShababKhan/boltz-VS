
## 2026-03-21 - Gemmi CIF table lookups with exceptions are extremely slow
**Learning:** Checking for the presence of elements in gemmi.cif.Block.find() using exceptions (with `contextlib.suppress(Exception)`) is 11x slower than explicitly checking `len(table) > 0`. This significantly impacts performance given the large number of attributes extracted from MMCIF files where many fields can be legitimately missing.
**Action:** When working with `gemmi.cif.Block.find(keys)`, assign the resulting table to a variable and explicitly check `if len(table) > 0:` to handle missing elements safely and efficiently instead of catching `IndexError` via `contextlib.suppress`.
