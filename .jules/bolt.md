
## 2025-02-28 - Optimize MMCIF metadata parsing
**Learning:** Using `contextlib.suppress(Exception)` to handle missing MMCIF metadata keys or values (e.g. `_refine.ls_d_res_high`) during parsing with `gemmi.cif.Block.find()` introduces significant performance overhead, as exceptions are used for flow control. Missing data is highly frequent in these tables.
**Action:** Replace `contextlib.suppress(Exception)` with explicit bound checks (`len(table) > 0`) before accessing elements, and use localized `try...except ValueError` exclusively for casting strings to floats, accounting for empty strings and missing value markers like `?` or `.`.
