## 2024-05-24 - Exception Handling Overhead in CIF Metadata Parsing
**Learning:** Parsing missing metadata fields from `gemmi.cif.Block.find` in Python is significantly slower when relying on `contextlib.suppress(Exception)` for control flow due to exception overhead.
**Action:** Always use explicit bounds checking (e.g. `len(block.find([key])) > 0`) when looking for optional fields instead of try/except blocks to avoid high runtime penalties on empty cases.
