# fp-quant-formats

A small CLI that enumerates **every representable value** of low-bit floating-point
quantization formats — the `E<exp>M<mantissa>` layouts used in ML, such as FP8 E4M3
and NVFP4. For each bit pattern it shows the sign / exponent / mantissa fields and the
decoded value, rendered as a table.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (used for running and dependency management)

## Install

```bash
uv sync
```

The only runtime dependency is [`rich`](https://github.com/Textualize/rich).

## Usage

Print the common NVIDIA tensor-core formats (NVFP4, FP8 E4M3, FP8 E5M2):

```bash
uv run main.py
```

Enumerate **all** `E?M?` layouts for a given total bit width (2–8):

```bash
uv run main.py --bits 4
```

### Options

| Option         | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| `--bits`, `-b` | Total bit width (2–8). Prints every `E?M?` split. Omit to print NVIDIA formats. |

## Example

```
$ uv run main.py --bits 4

FP4 formats: ===================================================================
...
E2M1 values: ───────────────────────────────────────────────────────────────────
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ bits ┃ sign ┃ mantissa ┃ exponent ┃ value ┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ 0000 │  0   │        0 │       00 │     0 │
│ 0001 │  0   │        1 │       00 │   0.5 │
│ 0010 │  0   │        0 │       01 │     1 │
│ 0011 │  0   │        1 │       01 │   1.5 │
│ 0100 │  0   │        0 │       10 │     2 │
│ 0101 │  0   │        1 │       10 │     3 │
│ 0110 │  0   │        0 │       11 │     4 │
│ 0111 │  0   │        1 │       11 │     6 │
│ 1000 │  1   │        0 │       00 │    -0 │
│ 1001 │  1   │        1 │       00 │  -0.5 │
│ 1010 │  1   │        0 │       01 │    -1 │
│ 1011 │  1   │        1 │       01 │  -1.5 │
│ 1100 │  1   │        0 │       10 │    -2 │
│ 1101 │  1   │        1 │       10 │    -3 │
│ 1110 │  1   │        0 │       11 │    -4 │
│ 1111 │  1   │        1 │       11 │    -6 │
└──────┴──────┴──────────┴──────────┴───────┘
```

The decoder follows IEEE-754-style semantics: the exponent bias is `2**(exp_bits-1) - 1`,
an all-zero exponent denotes subnormals (no implicit leading 1), and signed zero is shown
as `-0`.
