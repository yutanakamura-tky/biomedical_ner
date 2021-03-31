# This script can be executed from anywhere.

import dataclasses
import pathlib
import re
from typing import List

from tqdm import tqdm


def main():
    # 'modules' dir
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    REPO_DIR = (BASE_DIR / "../..").resolve()
    CORPUS_DIR = REPO_DIR / "corpus"
    NCBI_DIR = CORPUS_DIR / "ncbi_disease_corpus"

    convert_brat_txt_to_xml(NCBI_DIR / "NCBItrainset_corpus.txt")
    convert_brat_txt_to_xml(NCBI_DIR / "NCBIdevelopset_corpus.txt")
    convert_brat_txt_to_xml(NCBI_DIR / "NCBItestset_corpus.txt")


def convert_brat_txt_to_xml(input_path: str):
    with open(input_path) as f:
        print(f"Opening {input_path} ...")
        lines = clean_txt_file(f.readlines())

    # split records (separated with a brank line)
    raw_documents = "".join(lines).split("\n\n")

    documents = [convert_text_to_document_obj(doc) for doc in tqdm(raw_documents)]
    xml_text = compose_xml(documents)
    output_path = pathlib.Path(input_path).with_suffix(".xml")

    with open(output_path, "w") as f:
        f.write(xml_text)

    print(f"XML written to {output_path}")


def clean_txt_file(lines: List[str]) -> List[str]:
    # If a file starts with a brank line, remove it
    lines = lines[1:] if lines[0] == "\n" else lines

    # If a file ends with a brank line, remove it
    lines = lines[:-1] if lines[-1] == "\n" else lines

    return lines


@dataclasses.dataclass
class Entity:
    start: int
    end: int
    value: str
    entity_type: str
    mesh: str

    @property
    def opening_tag(self):
        return f'<disease type="{self.entity_type}" mesh="{self.mesh}">'

    @property
    def closing_tag(self):
        return "</disease>"


@dataclasses.dataclass
class Document:
    pmid: int
    content: str
    entities: List[Entity]


def convert_text_to_document_obj(document: str) -> Document:
    doc_lines = document.split("\n")
    # Extract PMID from the beginning of each document
    # e.g., '12345678|t|...' -> '12345678'
    pmid = re.match(r"^\d+", doc_lines[0]).group(0)

    # '12345678|t|foo' -> 'foo'
    title = re.match(r"^\d+\|t\|(.+)", doc_lines[0]).group(1).strip()

    # '12345678|a|foo' -> 'foo'
    article = re.match(r"^\d+\|a\|(.+)", doc_lines[1]).group(1).strip()

    content = f"{title} {article}"

    raw_annotations = doc_lines[2:]
    annotations = []

    for annotation in raw_annotations:
        annotation_values = annotation.split("\t")
        if len(annotation_values) == 6:
            _pmid, start, end, value, entity_type, mesh = annotation_values
            entity = Entity(
                start=int(start),
                end=int(end),
                value=value,
                entity_type=entity_type,
                mesh=mesh,
            )
            annotations.append(entity)
        else:
            # Skip lines without entity in BC5CDR corpus
            # e.g., '12345678 CID Dxxxxxx Dyyyyyy ...'
            continue

    result = Document(pmid=int(pmid), content=content, entities=annotations)
    return result


def compose_xml(documents: List[Document]) -> str:
    header = f'<xml version="1.0" encoding="UTF-8"?>\n<dataset n_documents="{len(documents)}">\n'
    footer = "</dataset>"
    xml_text = header

    for document in tqdm(documents):
        xml_text += compose_document_tag(document)

    xml_text += footer
    return xml_text


def compose_document_tag(document: Document) -> str:
    disease_tags = {}
    for entity in document.entities:
        if entity.start not in disease_tags.keys():
            disease_tags[entity.start] = {
                "opening_tags": [entity.opening_tag],
                "closing_tags": [],
            }

        else:
            disease_tags[entity.start]["opening_tags"].append(entity.opening_tag)

        if entity.end not in disease_tags.keys():
            disease_tags[entity.end] = {
                "opening_tags": [],
                "closing_tags": [entity.closing_tag],
            }
        else:
            disease_tags[entity.end]["closing_tags"].append(entity.closing_tag)

    header = f'<document id="{document.pmid}">\n'
    footer = "\n</document>\n"

    xml_text = header

    for i in range(len(document.content)):
        # Add letters to XML text one by one
        if i in disease_tags.keys():
            # When comming to entity position, add tags to XML text

            # Closing tags have to be added in inverted order to avoid tag crossing
            # e.g.,
            #     (text)
            #     12345678 10 22 entity_1 type_1 mesh_1
            #     12345678 15 22 entity_2 type_2 mesh_2
            #
            #     (disease_tags)
            #     {10:{'opening_tags':['opening_tag_1'], 'closing_tags':[]},
            #      15:{'opening_tags':['opening_tag_2'], 'closing_tags':[]},
            #      22:{'opening_tags':[], 'closing_tags':['closing_tag_1', 'closing_tag_2']}}
            #
            #     -> 'closing_tag_2' must be come before 'closing_tag_1'

            xml_text += "".join(disease_tags[i]["closing_tags"][::-1])

            # Opening tags have to be added in inverted order to avoid tag crossing
            # e.g.,
            #     (text)
            #     12345678 40 45 entity_1 type_1 mesh_1
            #     12345678 40 53 entity_2 type_2 mesh_2
            #
            #     (disease_tags)
            #     {40:{'opening_tags':['opening_tag_1', 'opening_tag_2'], 'closing_tags':[]},
            #      45:{'opening_tags':[], 'closing_tags':['closing_tag_1']},
            #      53:{'opening_tags':[], 'closing_tags':['closing_tag_2']}}
            #
            #     -> 'opening_tag_2' must be come before 'opening_tag_1'

            xml_text += "".join(disease_tags[i]["opening_tags"][::-1])

        # Convert letters <, >, ', " to avoid XML parsing errors
        xml_text += convert_letters(document.content[i])
    xml_text += footer
    return xml_text


def convert_letters(letter: str) -> str:
    if letter not in ("'", '"', "<", ">"):
        return letter
    else:
        mapping = {
            "'": "&apos;",
            '"': "&quot;",
            "<": "&lt;",
            ">": "&gt;",
        }
        return mapping[letter]


if __name__ == "__main__":
    main()
