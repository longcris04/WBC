import argparse
import pandas as pd
from pathlib import Path

CLASSES = ["BA", "BL", "BNE", "EO", "LY", "MMY", "MO",
           "MY", "PC", "PLY", "PMY", "SNE", "VLY"]


def main():
    parser = argparse.ArgumentParser(
        description="Fill all labels in submission.csv with a specified class"
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        required=True,
        help=f"Class abbreviation. Options: {', '.join(CLASSES)}"
    )
    args = parser.parse_args()

    if args.class_name not in CLASSES:
        print(f"Error: '{args.class_name}' is not a valid class.")
        print(f"Valid classes are: {', '.join(CLASSES)}")
        exit(1)

    csv_path = Path(__file__).parent / "submission.csv"

    if not csv_path.exists():
        print(f"Error: submission.csv not found at {csv_path}")
        exit(1)

    df = pd.read_csv(csv_path)
    df["labels"] = args.class_name
    df.to_csv(csv_path, index=False)

    print(f"Successfully filled all {len(df)} labels with class '{args.class_name}'")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
