import pathlib

import bs4

from biomedical_ner.src.modules.dataset.brat_to_xml import BratToXmlConverter

BASE_DIR = pathlib.Path(__file__).resolve().parent


def test_compose_xml_works_correctly():
    text_sample_path = BASE_DIR / "sample_ncbi_disease_corpus.txt"
    xml_sample_path = BASE_DIR / "sample_ncbi_disease_corpus.xml"
    with open(xml_sample_path) as f:
        expected_xml_text = f.read()

    xml_text = BratToXmlConverter.convert_brat_txt_to_xml_text(text_sample_path)
    xml = bs4.BeautifulSoup(xml_text, "xml").prettify()
    expected_xml = bs4.BeautifulSoup(expected_xml_text, "xml").prettify()

    assert xml == expected_xml
