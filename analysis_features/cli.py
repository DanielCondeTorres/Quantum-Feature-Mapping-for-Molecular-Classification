"""Command-line entry points.

    python -m analysis_features clean
    python -m analysis_features embed --smiles "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    python -m analysis_features summary
    python -m analysis_features benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cleaning import clean_dataset
from .config import DEFAULT_OUTDIR, DEFAULT_PREVIOUS, DEFAULT_RAW
from .embedding import ORDER_CHOICES, embed
from .io import display_path, load_previous_smiles, load_raw, write_outputs


def _clean(args: argparse.Namespace) -> None:
    frame = load_raw(args.raw)
    previous = load_previous_smiles(args.previous)
    result = clean_dataset(
        frame,
        previous,
        cleanup=args.cleanup,
        source=display_path(args.raw),
    )
    paths = write_outputs(result, args.outdir)
    if not args.quiet:
        print("\n".join(result.report.log_lines()))
        print("\nSaved:")
        for name, path in paths.items():
            print(f"  {name:10s} {path}")


def _embed(args: argparse.Namespace) -> None:
    result = embed(
        args.smiles,
        order=args.order,
        property=args.property,
        tail_mode=args.tail,
        readout=args.readout,
        strip_salts=not args.keep_salts,
        random_seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return

    print(f"Input:         {result.input_smiles}")
    print(f"Salt stripped: {result.salt_stripped}")
    print(f"Parent:        {result.parent_smiles}")
    print(
        f"Order: {result.order}   property: {result.property}   "
        f"tail: {result.tail_mode}   readout: {result.readout}"
    )
    print("\nQubit  Atom  Element  Angle (rad)")
    for qubit in result.qubits:
        print(
            f"q{qubit.qubit:<5d} {qubit.atom_index:<5d} {qubit.element:<7s} "
            f"{qubit.angle:.6f}"
        )
    print(f"q10 (tail A) angle: {result.tail_angles[0]:.6f}")
    print(f"q11 (tail B) angle: {result.tail_angles[1]:.6f}")
    print(f"\nBond gates applied: {len(result.bond_gates)}")
    print(f"Feature vector: {result.features.size} values")
    print("First 8:", " ".join(f"{value:+.4f}" for value in result.features[:8]))


def _summary(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.clean)
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    print("Labels:", frame["label"].value_counts().sort_index().to_dict())
    for column in (
        "in_previous_250",
        "stereo_label_conflict",
        "tautomer_label_conflict",
    ):
        if column in frame.columns:
            print(f"{column}: {int(frame[column].sum())} rows")
    print(f"Charged structures: {int((frame['formal_charge'] != 0).sum())}")
    print(f"Molecules beyond ten heavy atoms: {int((frame['atoms_beyond_10'] > 0).sum())}")
    if "source_num" in frame.columns:
        print("WARNING: source_num is present; never use it as a model feature.")


def _benchmark(args: argparse.Namespace) -> None:
    from .benchmark import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved: {args.outdir}")


def _verify(args: argparse.Namespace) -> None:
    from .verify import main as verify_main

    raise SystemExit(verify_main(args.outdir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis_features",
        description="Clean full BBBP and inspect the 10+2 embedding.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="run the cleaning pipeline")
    clean.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    clean.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    clean.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    clean.add_argument(
        "--cleanup",
        action="store_true",
        help="apply rdMolStandardize.Cleanup (off by default: it rewrites charges)",
    )
    clean.add_argument("--quiet", action="store_true")
    clean.set_defaults(func=_clean)

    embedding = subparsers.add_parser("embed", help="print one molecule's circuit")
    embedding.add_argument("--smiles", required=True)
    embedding.add_argument("--order", choices=ORDER_CHOICES, default="canonical")
    embedding.add_argument("--property", choices=("pauling", "tpsa"), default="pauling")
    embedding.add_argument("--tail", choices=("paper", "global"), default="paper")
    embedding.add_argument("--readout", choices=("z_zz", "full"), default="z_zz")
    embedding.add_argument("--seed", type=int, default=0, help="seed for random orders")
    embedding.add_argument(
        "--keep-salts",
        action="store_true",
        help="do not strip counter-ions before assigning qubits",
    )
    embedding.add_argument("--json", action="store_true")
    embedding.set_defaults(func=_embed)

    summary = subparsers.add_parser("summary", help="summarise a cleaned CSV")
    summary.add_argument("--clean", type=Path, default=DEFAULT_OUTDIR / "BBBP_clean_full.csv")
    summary.set_defaults(func=_summary)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare atom-ordering rules on scaffold folds"
    )
    benchmark.add_argument("--outdir", type=Path, default=Path("results"))
    benchmark.set_defaults(func=_benchmark)

    verify = subparsers.add_parser("verify", help="check every cleaning requirement")
    verify.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    verify.set_defaults(func=_verify)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
