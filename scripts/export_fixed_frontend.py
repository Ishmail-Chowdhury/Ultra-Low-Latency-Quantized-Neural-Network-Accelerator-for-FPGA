"""Compile the existing trained preprocessing into integer constants and ROMs."""

import argparse
from pathlib import Path

from models.lob.fixed_features import export_frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("build/prediction-quality"))
    parser.add_argument("--output", type=Path, default=Path("build/fixed-features/frontend"))
    args = parser.parse_args()
    document = export_frontend(args.model, args.output)
    print(f"Exported {len(document['feature_names'])} normalization coefficients and integer log LUT to {args.output}")


if __name__ == "__main__":
    main()
