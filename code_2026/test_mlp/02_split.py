import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# Default paths (relative to this script, so it runs from any folder)
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TABLE = SCRIPT_DIR / "01_image_to_table" / "01_image_to_table.csv"   # made by 01_image_to_table.py
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "02_split"

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

# =========================================================
# Split settings: fractions of the number of rows
# =========================================================
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 0

# =========================================================
# Shuffle the rows, then cut by number of rows
# =========================================================
def split_by_rows(df, train_fraction, val_fraction, seed):
    n_rows = len(df)
    n_train = int(train_fraction * n_rows)
    n_val = int(val_fraction * n_rows)
    n_test = n_rows - n_train - n_val          # the remainder, so no row is lost

    # Shuffled row positions: the same seed always gives the same split
    order = np.random.default_rng(seed).permutation(n_rows)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )

# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Shuffle the rows of a (Row, Col, R, G, B, Class) table and split it by "
                    "number of rows into train.csv, val.csv and test.csv."
    )
    parser.add_argument("--table", type=str, default=str(DEFAULT_TABLE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train", type=float, default=TRAIN_FRACTION, help="Fraction of rows for train")
    parser.add_argument("--val", type=float, default=VAL_FRACTION, help="Fraction of rows for val")
    parser.add_argument("--test", type=float, default=TEST_FRACTION, help="Fraction of rows for test")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()

def main():
    args = parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1: train={args.train}, val={args.val}, test={args.test} "
            f"(sum = {args.train + args.val + args.test})"
        )

    table_path = Path(args.table)
    output_dir = Path(args.output_dir)

    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            "Run 01_image_to_table.py first."
        )
    df = pd.read_csv(table_path)
    print(f"Table: {table_path.name} | shape: {df.shape}")

    splits = dict(zip(
        [TRAIN_FILE, VAL_FILE, TEST_FILE],
        split_by_rows(df, args.train, args.val, args.seed),
    ))

    # =========================================================
    # QC
    # =========================================================
    print(f"\nSeed: {args.seed} | fractions: train={args.train}, val={args.val}, test={args.test}")
    for name, part in splits.items():
        print(f"\n===== {name} =====")
        print(f"Rows: {len(part):,} ({100 * len(part) / len(df):.2f}% of the table)")
        print("Class share (%):")
        print((100 * part["Class"].value_counts(normalize=True)).round(2).to_string())
        print("First 5 rows:")
        print(part.head(5))

    # Every pixel must be in exactly one table
    width = int(df["Col"].max()) + 1
    keys = [part["Row"].to_numpy() * width + part["Col"].to_numpy() for part in splits.values()]
    all_keys = np.concatenate(keys)
    print(f"\nRows in train + val + test: {len(all_keys):,} (table has {len(df):,})")
    print(f"Pixels appearing in more than one table: {len(all_keys) - len(np.unique(all_keys))}")

    # =========================================================
    # Save CSV
    # =========================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(output_dir / name, index=False)
        print(f"Saved CSV: {output_dir / name}")

if __name__ == "__main__":
    main()
