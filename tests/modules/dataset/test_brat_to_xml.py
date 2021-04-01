import pathlib

import bs4

from biomedical_ner.src.modules import brat_to_xml

BASE_DIR = pathlib.Path(__file__).resolve().parent


def test_compose_xml_works_correctly():
    with open(BASE_DIR / "sample_ncbi_disease_corpus.txt") as f:
        text_sample = f.read()
    with open(BASE_DIR / "sample_ncbi_disease_corpus.xml") as f:
        expected_xml_text = f.read()

    document = brat_to_xml.convert_text_to_document_obj(text_sample)
    xml_text = brat_to_xml.compose_xml([document])
    xml = bs4.BeautifulSoup(xml_text, "xml").prettify()
    expected_xml = bs4.BeautifulSoup(expected_xml_text, "xml").prettify()

    assert xml == expected_xml
