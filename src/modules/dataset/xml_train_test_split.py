import argparse
import copy
import os
import pathlib

import bs4
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    args = get_args()
    print(vars(args))

    input_path = pathlib.Path(args.input_path)

    train_out_path, test_out_path = map(pathlib.Path, args.out_paths)
    os.makedirs(train_out_path.parent, exist_ok=True)
    os.makedirs(test_out_path.parent, exist_ok=True)

    with open(input_path) as f:
        print(f"Loading {input_path} ...")
        train_soup = bs4.BeautifulSoup(f, "lxml")
        print("Loaded!")

    if train_soup.select(f"{args.document_tag_name} {args.document_tag_name}"):
        raise RuntimeError(
            f'Tag "{args.document_tag_name}" is overlapping in {input_path}'
        )

    test_soup = copy.copy(train_soup)
    train_doc_tags = train_soup.select(args.document_tag_name)
    test_doc_tags = test_soup.select(args.document_tag_name)
    ids = np.arange(len(train_doc_tags))

    train_ids, test_ids = train_test_split(
        ids, train_size=args.train_size, random_state=args.random_state
    )

    for idx in test_ids:
        train_doc_tags[idx].decompose()

    for idx in train_ids:
        test_doc_tags[idx].decompose()

    if train_soup.select(f"{args.document_tag_name} {args.document_tag_name}"):
        raise RuntimeError(
            f'Tag "{args.document_tag_name}" is overlapping in training data'
        )
    if test_soup.select(f"{args.document_tag_name} {args.document_tag_name}"):
        raise RuntimeError(
            f'Tag "{args.document_tag_name}" is overlapping in test data'
        )

    with open(train_out_path, "w") as f:
        f.write(str(train_soup))
        print(f"Saved training data (n={len(train_ids)}) -> {train_out_path}")
    with open(test_out_path, "w") as f:
        f.write(str(test_soup))
        print(f"Saved test data (n={len(test_ids)}) -> {test_out_path}")


def get_args():
    description = """
    This script divides an unsplitted XML file into training & test set.
    In this process, the tags with the same name as '--document-tag' option will be recognized records.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_path", type=str)
    parser.add_argument("out_paths", type=str, nargs=2)
    parser.add_argument(
        "--document-tag", type=str, dest="document_tag_name", default="document"
    )
    parser.add_argument("--random-state", type=int, dest="random_state", default=42)
    parser.add_argument("--train-size", type=float, dest="train_size", default=0.8)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
