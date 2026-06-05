import argparse

from rich.console import Console
from rich.table import Table


def print_header(text: str, double_underline: bool = False) -> None:
    print(text)
    underline_char = "=" if double_underline else "-"
    print(underline_char * len(text))


def format_bits(value: int, width: int) -> str:
    return format(value, f"0{width}b")


def decode_fp_fields(bits: int, total_bits: int, exp_bits: int, man_bits: int) -> tuple[int, int, int]:
    sign = (bits >> (total_bits - 1)) & 1
    exp_mask = (1 << exp_bits) - 1
    man_mask = (1 << man_bits) - 1
    exponent = (bits >> man_bits) & exp_mask if exp_bits > 0 else 0
    mantissa = bits & man_mask
    return sign, exponent, mantissa


def decode_fp_value(bits: int, total_bits: int, exp_bits: int, man_bits: int) -> float:
    sign, exponent, mantissa = decode_fp_fields(bits, total_bits, exp_bits, man_bits)
    sign_factor = -1.0 if sign else 1.0

    if exp_bits == 0:
        if man_bits == 0:
            return sign_factor
        return sign_factor * mantissa / (2**man_bits)

    bias = (1 << (exp_bits - 1)) - 1

    if exponent == 0:
        if mantissa == 0:
            return sign_factor * 0.0
        if man_bits == 0:
            return sign_factor * 0.0
        return sign_factor * (mantissa / (2**man_bits)) * (2 ** (1 - bias))

    if man_bits == 0:
        significand = 1.0
    else:
        significand = 1.0 + mantissa / (2**man_bits)

    return sign_factor * significand * (2 ** (exponent - bias))


def format_value(value: float, sign: int) -> str:
    if value == 0.0 and sign:
        return "-0"
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:g}"


def print_format_table(
    console: Console,
    total_bits: int,
    exp_bits: int,
    man_bits: int,
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("bits", justify="right")
    table.add_column("sign", justify="center")
    table.add_column("mantissa", justify="right")
    table.add_column("exponent", justify="right")
    table.add_column("value", justify="right")

    for bits in range(2**total_bits):
        sign, exponent, mantissa = decode_fp_fields(bits, total_bits, exp_bits, man_bits)
        value = decode_fp_value(bits, total_bits, exp_bits, man_bits)

        table.add_row(
            format_bits(bits, total_bits),
            str(sign),
            format_bits(mantissa, man_bits) if man_bits > 0 else "-",
            format_bits(exponent, exp_bits) if exp_bits > 0 else "-",
            format_value(value, sign),
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fp-quant-formats",
        description="Shows info about different floating point quantization formats",
    )
    parser.add_argument("--bits", "-b", type=int, default=4, choices=range(2, 9))

    args = parser.parse_args()
    console = Console()

    print_header(f"FP{args.bits} formats:", double_underline=True)

    for exp_bits in range(args.bits):
        man_bits = args.bits - exp_bits - 1
        print_header(f"E{exp_bits}M{man_bits} values:")
        print_format_table(console, args.bits, exp_bits, man_bits)
        print()


if __name__ == "__main__":
    main()
