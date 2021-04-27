import argparse
import pathlib

import bs4


def main():
    args = get_args()
    if (not args.overwrite) and (not args.out_path):
        raise RuntimeError("Specify out_path or set --overwrite.")
    if args.overwrite and args.out_path:
        raise RuntimeError("--overwrite cannot be set when out_path is specified.")

    input_path = pathlib.Path(args.input_path)
    out_path = pathlib.Path(args.out_path) if args.out_path else input_path

    with open(input_path) as f:
        soup = bs4.BeautifulSoup(f, "lxml")

    for text_tag in soup.select("text"):
        text_tag.unwrap()
    print("Unwrapped all <text> tags.")

    with open(out_path, "w") as f:
        f.write(str(soup))
        print(f"Saved! -> {out_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("out_path", type=str, nargs="?", help="Save destination.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
