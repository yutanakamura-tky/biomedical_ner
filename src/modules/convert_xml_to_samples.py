import argparse
import copy
import itertools
import os
import pathlib
from typing import List

import bs4
from nltk.tokenize import word_tokenize

# 0. Preparation

# 0-1. Preprocessing function


def main():
    args = get_args()
    input_path = pathlib.Path(args.input_path)
    out_dir = pathlib.Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load files
    # 1-1. XML file WITH tagging
    with open(input_path) as f:
        soup = bs4.BeautifulSoup(f, "xml")

    # 1-2. XML file WITHOUT medication tagging
    # CAUTION: this cannot process overlapping tags
    soup_notag = remove_tags(copy.copy(soup))

    for record_notag, record in zip(
        soup_notag.select("document"), soup.select("document")
    ):
        write_to_data(record_notag, record, out_dir)


def remove_tags(soup: bs4.BeautifulSoup) -> bs4.BeautifulSoup:
    for doc in soup.select("document"):
        for content in doc.contents:
            if type(content) is bs4.element.Tag:
                content.unwrap()
    return soup


def write_to_data(
    untagged_record: bs4.element.Tag, tagged_record: bs4.element.Tag, save_dir: str
) -> None:
    """
    This function takes a pair of bs4.element.Tag objects of the same XML record as input.

    input
    -----
    record (bs4.element.Tag):
        An XML record without NER tag.


    tagged_record (bs4.element.Tag):
        An XML record with NER tag.

        Example.
            Let the XML file be an XML file like below and
            suppose that the content of <TEXT> have some named entities tagged with <ENTITY>:
            Then, pass bs4.element.Tag object corresponding to <TEXT> tag to this function.

                <ROOT>
                    <TEXT> ... <ENTITY> ... </ENTITY> ... </TEXT>
                        ...
                    <TEXT> ... <ENTITY> ... </ENTITY> ... </TEXT>
                </ROOT>

    save_dir (str):
        The path of directory to save data.


    output
    ------
    None
    """

    def apply_tokenization_to_record(record):
        tokenized_objects = map(
            lambda obj: tokenize(obj)
            if type(obj) is bs4.element.NavigableString
            else tokenize(obj.contents[0]),
            record.contents,
        )
        return list(itertools.chain(*tokenized_objects))

    # Here we suppose the record tags have "id" property and use its value for filename prefix.
    # <document id="foo">
    #     ...
    # </document>

    file_name_prefix = untagged_record["id"]
    print(f"Processing record ID {file_name_prefix} ...")

    # Caution:
    #    Do not create tokens_notag by tokenizing untagged record.
    #    This is because it can change tokenization result.
    #
    #    For example,
    #        record_without_tag = <document>I took vitamin E.</document>
    #        record_with_tag = <document>I took <m>vitamin E</m>.</document>
    #
    #    In this case,
    #        xml_to_iob(record_with_tag) returns ['O', 'O', 'B', 'I', 'O'] (length=5).
    #        apply_tokenization_to_record(record_with_tag) returns ['I', 'took', 'vitamin', 'E', '.'] (length=5).
    #
    #    But,
    #        tokenize(record_without_tag) returns ['I', 'took', 'vitamin', 'E.'] (length=4(different length)).

    text_notag = preprocessing(untagged_record.contents[0])
    tokens_notag = apply_tokenization_to_record(tagged_record)
    iob = xml_to_iob(tagged_record)
    assert len(tokens_notag) == len(
        iob
    ), "The number of tokens and IOB tags are different."

    tokens_notag_str = " ".join(tokens_notag)
    iob_str = ",".join(iob)

    os.makedirs(save_dir, exist_ok=True)

    with open(f"{str(save_dir)}/{file_name_prefix}.text", "w") as f:
        f.write(text_notag)
        print(f"Wrote text to {str(save_dir)}/{file_name_prefix}.text")

    with open(f"{str(save_dir)}/{file_name_prefix}.tokens", "w") as f:
        f.write(tokens_notag_str)
        print(f"Wrote tokens to {str(save_dir)}/{file_name_prefix}.tokens")

    with open(f"{str(save_dir)}/{file_name_prefix}.ann", "w") as f:
        f.write(iob_str)
        print(f"Wrote IOB tag to {str(save_dir)}/{file_name_prefix}.ann")


def preprocessing(text: str) -> str:
    result = text.replace("&gt;", ">")
    result = result.replace("&lt;", "<")
    result = result.replace("&quot;", '"')
    result = result.replace("&apos;", "'")
    return result


def tokenize(text: str) -> List[str]:
    return word_tokenize(preprocessing(text))


def xml_to_iob(tagged_record: bs4.element.Tag) -> str:
    """
    input
    -----
    tagged_record (bs4.element.Tag):
        An XML record with NER tag.

        Example.
            Let the XML file be an XML file like below and
            suppose that the content of <TEXT> have some named entities tagged with <ENTITY>:
            Then, pass bs4.element.Tag object corresponding to <TEXT> tag to this function.

                <ROOT>
                    <TEXT> ... <ENTITY> ... </ENTITY> ... </TEXT>
                        ...
                    <TEXT> ... <ENTITY> ... </ENTITY> ... </TEXT>
                </ROOT>


    output
    ------
    result (str): IOB tagging for each token of the text in the record.
            Example.

    """
    parts = tagged_record.contents
    iob = []

    for part in parts:
        if type(part) is bs4.element.Tag:
            # if the part is in tag
            length = len(tokenize(part.contents[0]))
            iob += ["B"]
            iob += ["I"] * (length - 1)

        elif type(part) is bs4.element.NavigableString:
            # if the part is out of tag
            length = len(tokenize(part))
            iob += ["O"] * length

    return iob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
