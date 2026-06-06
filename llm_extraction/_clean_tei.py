#!/usr/bin/env python3
"""Clean a GROBID TEI XML for inclusion in an LLM prompt.

Drops content that is bulky and not referenced by any schema field used by
extract_claude/ or extract_claude_claim/:

  * teiHeader/encodingDesc      (GROBID's own toolchain metadata)
  * back/div[type=references]   (the bibliography — bibliometrics come from
                                 OpenAlex/Crossref, not the inline ref list)
  * back/div[type=acknowledgement]
  * back/listOrg[type=infrastructure]

Kept (because at least one schema field cares about them):

  * teiHeader/fileDesc          (title, authors, affiliations, journal)
  * teiHeader/profileDesc       (abstract, keywords / textClass)
  * text/body                   (the paper itself)
  * back/div[type=annex]        (appendix material — may carry power analysis)
  * back/div[type=availability] (data-availability statement → open_data)
  * back/div[type=funding]      (funding statement → funding_disclosure)
  * back/listOrg[type=funding]  (structured funder list)

Survey on the 1040 TEIs in data/tei_xml/ confirmed those are the only
back-matter types GROBID emits in practice.

Usage:
    python3 _clean_tei.py path/to/paper.tei.xml > cleaned.xml
    from _clean_tei import clean_tei_file       # returns a UTF-8 str
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = f"{{{TEI_NS}}}"

DROP_BACK_DIV_TYPES = {"references", "acknowledgement"}
DROP_BACK_LISTORG_TYPES = {"infrastructure"}


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _strip(parent: ET.Element, child: ET.Element) -> None:
    parent.remove(child)


def clean_tree(root: ET.Element) -> ET.Element:
    """Mutates `root` in place. Returns it for convenience."""
    # Drop GROBID's encodingDesc inside teiHeader.
    for header in root.iter(NS + "teiHeader"):
        for enc in list(header.findall(NS + "encodingDesc")):
            _strip(header, enc)

    # Drop unwanted children of <back>. Iterate over a snapshot since we mutate.
    for back in root.iter(NS + "back"):
        for child in list(back):
            tag = _local(child.tag)
            typ = child.attrib.get("type", "")
            if tag == "div" and typ in DROP_BACK_DIV_TYPES:
                _strip(back, child)
            elif tag == "listOrg" and typ in DROP_BACK_LISTORG_TYPES:
                _strip(back, child)

    return root


def clean_tei_file(path: str | Path) -> str:
    """Read a TEI file, return the cleaned XML as a UTF-8 string."""
    ET.register_namespace("", TEI_NS)
    tree = ET.parse(str(path))
    root = clean_tree(tree.getroot())
    return ET.tostring(root, encoding="unicode")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _clean_tei.py path/to/paper.tei.xml", file=sys.stderr)
        return 2
    sys.stdout.write(clean_tei_file(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
