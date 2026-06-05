# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CLI tool that enumerates and prints every representable value of low-bit floating-point quantization formats (the `E<exp>M<mantissa>` layouts used in ML, e.g. FP8 E4M3, NVFP4). Output is rendered as `rich` tables.

## Commands

```bash
uv run main.py              # print common NVIDIA tensor-core formats (NVFP4, FP8 E4M3, FP8 E5M2)
uv run main.py --bits 4     # enumerate all E?M? layouts for a given bit width (2..8)
uv sync                     # install/resolve dependencies from uv.lock
```

Python 3.13+ is required. Dependencies are managed with `uv` (`uv.lock`); the only runtime dependency is `rich`.

## Architecture

Everything lives in `main.py`. The data flow is a pipeline keyed on the triple `(total_bits, exp_bits, man_bits)`:

- `decode_fp_fields` splits a raw integer bit pattern into `(sign, exponent, mantissa)`.
- `decode_fp_value` converts those fields to a float, implementing IEEE-754-style semantics by hand: bias is `2**(exp_bits-1) - 1`, exponent `0` means subnormal (no implicit leading 1), and the `exp_bits == 0` / `man_bits == 0` edge cases are handled explicitly. This is the core numeric logic — changes here affect correctness of every printed value.
- `print_format_table` iterates all `2**total_bits` patterns for one layout and builds a `rich.Table`.
- `print_all_formats` (for `--bits`) sweeps every `E?M?` split of a width; `print_nvidia_formats` prints the hardcoded `NVIDIA_FORMATS` list. Both group output with `console.rule` headers.

When adding a new named format, append to `NVIDIA_FORMATS` as `(total_bits, exp_bits, man_bits)`.

## Notes

- There is no test suite yet. Per the user's working style, add tests alongside any logic change — `decode_fp_value` is the natural unit to cover (known values, subnormals, signed zero, the zero-exp/zero-mantissa edge cases).
- `README.md` is currently empty.
