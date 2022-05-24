# This script converts MedNLI dataset, provided as JSONL files, into CoNLL tsv.

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    args = get_args()
    DIR = Path(args.input_dir)

    train_path = DIR / "mli_train_v1.jsonl"
    val_path = DIR / "mli_dev_v1.jsonl"
    test_path = DIR / "mli_test_v1.jsonl"

    convert_jsonl_to_tsv(train_path)
    convert_jsonl_to_tsv(val_path)
    convert_jsonl_to_tsv(test_path)


def convert_jsonl_to_tsv(input_path: str) -> None:
    input_path = Path(input_path)

    with open(input_path, encoding="utf-8-sig") as f:
        lines = f.readlines()
        print(f"Loaded {input_path}")
    records = [json.loads(line) for line in lines]

    df = pd.DataFrame(records).loc[:, ["sentence1", "sentence2", "gold_label"]]

    df.loc[:, "sentence1"] = df.loc[:, "sentence1"].apply(
        lambda x: x.replace("\ufeff", "")
    )
    df.loc[:, "sentence2"] = df.loc[:, "sentence2"].apply(
        lambda x: x.replace("\ufeff", "")
    )
    df.loc[:, "gold_label"] = df.loc[:, "gold_label"].apply(
        lambda x: x.replace("\ufeff", "")
    )

    output_path = input_path.with_suffix(".tsv")
    df.to_csv(output_path, index=False, header=False, sep="\t")
    print(f"Saved {output_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Specify the directory where MedNLI dataset is placed",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
