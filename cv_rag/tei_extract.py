from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass(slots=True)
class Section:
    title: str
    text: str


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return _normalize_text("".join(node.itertext()))


def extract_sections(tei_xml: str) -> list[Section]:
    if not tei_xml.strip():
        return []

    root = ET.fromstring(tei_xml)
    sections: list[Section] = []

    abstract_paragraphs = [
        _node_text(p)
        for p in root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS)
        if _node_text(p)
    ]
    if abstract_paragraphs:
        sections.append(Section(title="Abstract", text="\n\n".join(abstract_paragraphs)))

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return sections

    for idx, div in enumerate(body.findall("./tei:div", TEI_NS), start=1):
        title = _node_text(div.find("./tei:head", TEI_NS)) or f"Section {idx}"

        paragraphs = [_node_text(p) for p in div.findall(".//tei:p", TEI_NS)]
        paragraphs = [p for p in paragraphs if p]
        text = "\n\n".join(paragraphs) if paragraphs else _node_text(div)

        if text:
            sections.append(Section(title=title, text=text))

    if sections:
        return sections

    body_paragraphs = [_node_text(p) for p in body.findall(".//tei:p", TEI_NS)]
    body_paragraphs = [p for p in body_paragraphs if p]
    if body_paragraphs:
        sections.append(Section(title="Body", text="\n\n".join(body_paragraphs)))

    return sections
