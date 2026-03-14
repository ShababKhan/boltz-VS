## 2024-03-14 - Exception handling in parsing mmcif
**Learning:** Using `contextlib.suppress(Exception)` with `block.find(key)[0][0]` on missing keys throws `IndexError`, which is about 50x slower to handle than checking `if len(block.find(key)) > 0:`. Moreover, wrapping an entire `for key in keys:` loop in a `contextlib.suppress` causes the loop to prematurely exit on the first missing key, silently skipping fallback keys.
**Action:** Replace `contextlib.suppress(Exception)` with explicit length checks when parsing mmcif structures to improve speed and fix fallback logic.
