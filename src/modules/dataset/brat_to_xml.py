# This script can be executed from anywhere.

import dataclasses
import pathlib
import re
from typing import List

from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = (BASE_DIR / "../../..").resolve()
CORPUS_DIR = REPO_DIR / "corpus"


def main():
    # 'dataset' dir
    convert_ncbi_disease_corpus()
    convert_bc5cdr_corpus()


def convert_ncbi_disease_corpus():
    for corpus_name in ["train", "develop", "test"]:
        NCBI_DIR = CORPUS_DIR / "ncbi_disease_corpus"
        input_path = NCBI_DIR / f"NCBI{corpus_name}set_corpus.txt"
        output_path = NCBI_DIR / f"{corpus_name}.xml"

        with open(input_path) as f:
            brat_text = f.read()

        xml_text = BratToXmlConverter.convert_brat_text_to_xml_text(brat_text)
        with open(output_path, "w") as f:
            f.write(xml_text)
        print(f"XML written to {output_path}")


def convert_bc5cdr_corpus():
    corpus_name_mapping = {
        "Training": "train",
        "Development": "develop",
        "Test": "test",
    }

    for in_name, out_name in corpus_name_mapping.items():
        BC5CDR_DIR = CORPUS_DIR / "bc5cdr"
        input_path = BC5CDR_DIR / f"CDR_{in_name}Set.PubTator.txt"
        output_path_disease = BC5CDR_DIR / f"disease_{out_name}.xml"
        output_path_chemical = BC5CDR_DIR / f"chemical_{out_name}.xml"

        with open(input_path) as f:
            brat_text = f.read()

        regexp_disease_entity = r"(\d+\t){3}.+?\tDisease\tD\d+\n"
        regexp_chemical_entity = r"(\d+\t){3}.+?\tChemical\tD\d+\n"
        brat_text_disease = re.sub(regexp_chemical_entity, "", brat_text)
        brat_text_chemical = re.sub(regexp_disease_entity, "", brat_text)

        xml_text_disease = BratToXmlConverter.convert_brat_text_to_xml_text(
            brat_text_disease
        )
        xml_text_chemical = BratToXmlConverter.convert_brat_text_to_xml_text(
            brat_text_chemical
        )

        with open(output_path_disease, "w") as f:
            f.write(xml_text_disease)
        print(f"XML written to {output_path_disease}")

        with open(output_path_chemical, "w") as f:
            f.write(xml_text_chemical)
        print(f"XML written to {output_path_chemical}")


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


class BratToXmlConverter:
    @classmethod
    def convert_brat_text_to_xml_text(cls, text: str) -> str:
        """
        Opens convert NER annotation in Brat style and convert it into XML text
        Input file must satisfy the following:

            1. Starts with zero or one brank line
            2. Starts with zero or one brank line
            3. Documents are separated with a brank line
            4. The first line of each document is like below:
                f"{ID}|t|{TEXT}"
            5. The second line of each document is like below:
                f"{ID}|a|{TEXT}"
            6. The rest lines of each document are like below:
                f"{ID}\t{BEGIN_POSITION}\t{END_POSITION}\t{ENTITY}\t{ENTITY_TYPE}\t{MESH_TAG}"
        """
        # If the Brat text file starts with brank lines, remove them
        # If the Brat text file ends with brank lines, remove them
        text = text.strip()

        # split records (separated with a brank line)
        raw_documents = text.split("\n\n")

        documents = [
            cls.convert_text_to_document_obj(doc) for doc in tqdm(raw_documents)
        ]
        xml_text = cls.compose_xml(documents)

        return xml_text

    @staticmethod
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
                # Skip lines for relation extraction labels in BC5CDR corpus
                # e.g., '12345678 CID Dxxxxxx Dyyyyyy ...'
                continue

        result = Document(pmid=int(pmid), content=content, entities=annotations)
        return result

    @classmethod
    def compose_xml(cls, documents: List[Document]) -> str:
        xml_texts = []
        n_entity = []

        for document in tqdm(documents):
            xml_texts.append(cls.compose_xml_document_tag(document))
            n_entity.append(cls.count_entities(document))

        header = f'<xml version="1.0" encoding="UTF-8"?>\n<dataset n_documents="{len(documents)}" n_entity="{sum(n_entity)}">\n'
        footer = "</dataset>"
        xml_text = header + "".join(xml_texts) + footer

        return xml_text

    @classmethod
    def compose_xml_document_tag(cls, document: Document) -> str:
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

        n_entity = cls.count_entities(document)
        header = f'<document id="{document.pmid}" n_entity="{n_entity}">\n'
        footer = "\n</document>\n"

        xml_text = header

        for i in range(len(document.content) + 1):
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
            if i < len(document.content):
                xml_text += cls.convert_letters(document.content[i])
        xml_text += footer
        return xml_text

    @staticmethod
    def count_entities(document: Document) -> int:
        return len(document.entities)

    @staticmethod
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
