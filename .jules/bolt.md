
## 2025-02-28 - Fast dict/table lookups instead of contextlib.suppress in gemmi parser
**Learning:** Using `contextlib.suppress(Exception)` to handle missing MMCIF fields during parsing with `gemmi.cif.Block.find` is extremely slow in Python. Catching an exception for missing elements acts as a severe bottleneck.
**Action:** When working with `gemmi.cif.Block.find(list_of_keys)`, always use explicit bounds checking on the resulting table (`len(table) > 0`) instead of wrapping the element access in a try-except/suppress block. Only use `try...except ValueError` specifically when casting the extracted string value to a type like `float`, since MMCIF missing values (e.g. `?` or `.`) will correctly raise this.
