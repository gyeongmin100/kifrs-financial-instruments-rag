from __future__ import annotations

import re
import unicodedata
import zipfile
from math import ceil
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


PARAGRAPH_ID_PATTERN = (
    r"(?:"
    r"한\d+[A-Z]?(?:\.\d+[A-Z]?)*"
    r"|[A-Z]{1,5}\.?\d+[A-Z]?(?:\.\d+[A-Z]?)*"
    r"|\d+[A-Z]?(?:\.\d+[A-Z]?)*"
    r")"
)
PARAGRAPH_NUMBER_RE = re.compile(
    rf"^(?P<number>{PARAGRAPH_ID_PATTERN})(?=\s|$)"
)
LIST_MARKER_RE = re.compile(r"^(?P<marker>[⑴-⒇㈎-㈘①-⑳])\s*")
STANDARD_REF_RE = re.compile(
    r"(?:기업회계기준서\s*)?제(?P<kifrs>\d{4})호"
    r"|기업회계기준서\s*(?P<kifrs_without_je>\d{4})호"
    r"|(?P<ifrs>IFRS)\s*(?P<ifrs_number>[7７9９])"
    r"|(?P<ias>IAS)\s*(?P<ias_number>[3３][2２9９])",
    re.IGNORECASE,
)
PARAGRAPH_MARKER_RE = re.compile(rf"문단\s*(?={PARAGRAPH_ID_PATTERN})")
PARAGRAPH_ID_RE = re.compile(PARAGRAPH_ID_PATTERN)
SUBITEM_PATTERN = r"(?:[⑴-⒇㈎-㈘①-⑳]|\(\d+\)|\([A-Za-z]\))"
SUBITEM_RE = re.compile(SUBITEM_PATTERN)
RANGE_DELIMITER_RE = re.compile(r"[~～∼〜\-‐‑‒–—]")
REFERENCE_CONNECTOR_RE = re.compile(
    r"\s*(?P<separator>,|，|·|․|ㆍ|과|와|및|또는|이나|나)"
    r"\s*(?:문단\s*)?"
)

SKIP_TEXT_ANCESTORS = {"tbl", "footNote", "header", "footer"}


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _mixed_text(text_element: ET.Element) -> str:
    """Read current HWPX text, excluding deleted revision text.

    HWPX stores revision markers as children of a text element and the edited
    characters as XML tails. A plain ``itertext`` therefore mixes deleted and
    inserted wording.
    """

    parts: list[str] = []
    delete_depth = 0
    if text_element.text:
        parts.append(text_element.text)

    for child in text_element:
        tag = local_name(child)
        if tag == "deleteBegin":
            delete_depth += 1
        elif tag == "deleteEnd":
            delete_depth = max(0, delete_depth - 1)
        elif delete_depth == 0:
            if tag == "tab":
                parts.append(" ")
            elif tag in {"lineBreak", "br"}:
                parts.append("\n")
            elif tag in {"nbSpace", "fwSpace"}:
                parts.append(" ")

        if child.tail and delete_depth == 0:
            parts.append(child.tail)

    return "".join(parts)


def element_text(element: ET.Element, excluded_ancestors: set[str] | None = None) -> str:
    excluded = excluded_ancestors or set()
    parts: list[str] = []

    def visit(node: ET.Element, blocked: bool = False) -> None:
        name = local_name(node)
        now_blocked = blocked or name in excluded
        if name == "t" and not now_blocked:
            parts.append(_mixed_text(node))
            return
        for child in node:
            visit(child, now_blocked)

    visit(element)
    return normalize_space("".join(parts))


@dataclass
class Reference:
    kind: str
    raw: str
    target_standard: str | None = None
    target_paragraph_start: str | None = None
    target_paragraph_end: str | None = None
    target_subitem_start: str | None = None
    target_subitem_end: str | None = None
    group_index: int | None = None
    group_raw: str | None = None
    separator: str | None = None
    range_delimiter: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    method: str = "explicit_reference"
    confidence: float = 1.0


@dataclass
class Footnote:
    footnote_id: str
    parent_paragraph_id: str | None
    number: str | None
    text: str


@dataclass
class TableCell:
    row: int
    column: int
    row_span: int
    column_span: int
    text: str


@dataclass
class Table:
    table_id: str
    parent_paragraph_id: str | None
    rows: int
    columns: int
    repeat_header: bool
    cells: list[TableCell]


@dataclass
class Subitem:
    marker: str
    text: str
    xml_index: int


@dataclass
class Paragraph:
    paragraph_id: str
    number: str
    zone: str
    section_path: list[str]
    text: str
    source_section: str
    xml_index: int
    document_order: int = 0
    occurrence: int = 1
    block_ids: list[str] = field(default_factory=list)
    subitems: list[Subitem] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    footnotes: list[Footnote] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    page_match_confidence: float = 0.0


@dataclass
class DocumentBlock:
    block_id: str
    block_type: str
    zone: str
    text: str
    source_section: str
    xml_index: int
    document_order: int
    searchable: bool
    search_priority: int
    section_path: list[str]
    parent_paragraph_id: str | None = None
    table_ids: list[str] = field(default_factory=list)
    footnote_ids: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


def _iter_descendants(element: ET.Element, tag: str) -> Iterable[ET.Element]:
    for candidate in element.iter():
        if local_name(candidate) == tag:
            yield candidate


def _standard_id(match: re.Match[str]) -> str:
    if match.group("kifrs"):
        return match.group("kifrs")
    if match.group("kifrs_without_je"):
        return match.group("kifrs_without_je")
    if match.group("ifrs"):
        number = unicodedata.normalize("NFKC", match.group("ifrs_number"))
        return {"7": "1107", "9": "1109"}[number]
    number = unicodedata.normalize("NFKC", match.group("ias_number"))
    return {"32": "1032", "39": "1039"}[number]


def _target_standard(text: str, marker_start: int, source_standard: str) -> str:
    prefix = text[max(0, marker_start - 500) : marker_start]
    prefix = re.split(r"(?<=[.!?])\s+", prefix)[-1]
    standards = list(STANDARD_REF_RE.finditer(prefix))
    if not standards:
        return source_standard
    latest = standards[-1]
    between = prefix[latest.end() :]
    direct_qualifier = re.fullmatch(
        r"\s*(?:의\s*)?(?:[‘'\"“][^’'\"”]{1,100}[’'\"”]\s*)?"
        r"(?:(?:부록|적용지침|결론도출근거)\s*[A-Z]?(?:의)?\s*)?",
        between,
    )
    anaphoric = re.search(r"(?:그|해당|동)\s*기준서(?:의)?\s*$", between)
    if direct_qualifier or anaphoric:
        return _standard_id(latest)
    return source_standard


def _parse_reference_atom(text: str, position: int) -> tuple[dict, int] | None:
    position += len(text[position:]) - len(text[position:].lstrip())
    paragraph_match = PARAGRAPH_ID_RE.match(text, position)
    if not paragraph_match:
        return None

    start = paragraph_match.group(0)
    cursor = paragraph_match.end()
    subitem_match = SUBITEM_RE.match(text, cursor)
    subitem_start = subitem_match.group(0) if subitem_match else None
    if subitem_match:
        cursor = subitem_match.end()

    end = None
    subitem_end = None
    delimiter = None
    whitespace_end = cursor + len(text[cursor:]) - len(text[cursor:].lstrip())
    delimiter_match = RANGE_DELIMITER_RE.match(text, whitespace_end)
    if delimiter_match:
        after_delimiter = delimiter_match.end()
        after_delimiter += len(text[after_delimiter:]) - len(
            text[after_delimiter:].lstrip()
        )
        end_match = PARAGRAPH_ID_RE.match(text, after_delimiter)
        end_subitem_match = SUBITEM_RE.match(text, after_delimiter)
        if end_match:
            delimiter = delimiter_match.group(0)
            end = end_match.group(0)
            cursor = end_match.end()
            trailing_subitem = SUBITEM_RE.match(text, cursor)
            if trailing_subitem:
                subitem_end = trailing_subitem.group(0)
                cursor = trailing_subitem.end()
        elif subitem_start and end_subitem_match:
            delimiter = delimiter_match.group(0)
            subitem_end = end_subitem_match.group(0)
            cursor = end_subitem_match.end()

    if subitem_end:
        kind = "subitem_range"
    elif end:
        kind = "paragraph_range"
    elif subitem_start:
        kind = "subitem"
    else:
        kind = "paragraph"

    return (
        {
            "kind": kind,
            "target_paragraph_start": start,
            "target_paragraph_end": end,
            "target_subitem_start": subitem_start,
            "target_subitem_end": subitem_end,
            "range_delimiter": delimiter,
            "char_start": position,
            "char_end": cursor,
        },
        cursor,
    )


def parse_references(text: str, source_standard: str) -> list[Reference]:
    """Extract explicit references without treating ordinary numeric ranges as citations."""

    references: list[Reference] = []
    for match in STANDARD_REF_RE.finditer(text):
        references.append(
            Reference(
                kind="standard",
                raw=match.group(0),
                target_standard=_standard_id(match),
                char_start=match.start(),
                char_end=match.end(),
            )
        )

    cursor = 0
    group_index = 0
    while marker_match := PARAGRAPH_MARKER_RE.search(text, cursor):
        group_index += 1
        group_start = marker_match.start()
        target_standard = _target_standard(text, group_start, source_standard)
        atom_position = marker_match.end()
        atoms: list[tuple[dict, str | None]] = []
        first = _parse_reference_atom(text, atom_position)
        if not first:
            cursor = marker_match.end()
            continue
        atom, group_end = first
        atoms.append((atom, None))

        while connector_match := REFERENCE_CONNECTOR_RE.match(text, group_end):
            next_atom = _parse_reference_atom(text, connector_match.end())
            if not next_atom:
                break
            atom, group_end = next_atom
            atoms.append((atom, connector_match.group("separator")))

        group_raw = text[group_start:group_end]
        for atom, separator in atoms:
            atom_start = atom["char_start"]
            atom_end = atom["char_end"]
            references.append(
                Reference(
                    kind=atom["kind"],
                    raw=text[atom_start:atom_end],
                    target_standard=target_standard,
                    target_paragraph_start=atom["target_paragraph_start"],
                    target_paragraph_end=atom["target_paragraph_end"],
                    target_subitem_start=atom["target_subitem_start"],
                    target_subitem_end=atom["target_subitem_end"],
                    group_index=group_index,
                    group_raw=group_raw,
                    separator=separator,
                    range_delimiter=atom["range_delimiter"],
                    char_start=atom_start,
                    char_end=atom_end,
                )
            )
        cursor = group_end

    return references


def _parse_table(
    table: ET.Element, table_id: str, parent_paragraph_id: str | None = None
) -> Table:
    cells: list[TableCell] = []
    for row_element in (child for child in table if local_name(child) == "tr"):
        for cell in (child for child in row_element if local_name(child) == "tc"):
            address = next(_iter_descendants(cell, "cellAddr"), None)
            span = next(_iter_descendants(cell, "cellSpan"), None)
            cells.append(
                TableCell(
                    row=int(address.get("rowAddr", "0")) if address is not None else 0,
                    column=int(address.get("colAddr", "0")) if address is not None else 0,
                    row_span=int(span.get("rowSpan", "1")) if span is not None else 1,
                    column_span=int(span.get("colSpan", "1")) if span is not None else 1,
                    text=element_text(cell, {"tbl"}),
                )
            )

    return Table(
        table_id=table_id,
        parent_paragraph_id=parent_paragraph_id,
        rows=int(table.get("rowCnt", "0")),
        columns=int(table.get("colCnt", "0")),
        repeat_header=table.get("repeatHeader") == "1",
        cells=cells,
    )


def _zone_transition(text: str, current_zone: str, standard_id: str) -> str:
    normalized = normalize_space(text)
    if normalized == f"기업회계기준서 제{standard_id}호":
        return "standard_body"
    if normalized.startswith("부록 A") and "용어의 정의" in normalized:
        return "appendix_definitions"
    if (
        normalized.startswith("부록 A") or normalized.startswith("부록 B")
    ) and "적용지침" in normalized:
        return "application_guidance"
    if "회계기준위원회의 의결" in normalized and len(normalized) <= 160:
        return "committee_resolution"
    if normalized == "적용사례":
        return "application_examples"
    if normalized == "실무적용지침" or normalized.startswith("실무적용지침 목차"):
        return "implementation_guidance"
    if normalized == "결론도출근거" or (
        len(normalized) <= 120 and normalized.endswith("의 결론도출근거")
    ):
        return "basis_for_conclusions"
    if "소수의견" in normalized and len(normalized) <= 180:
        return "dissenting_opinion"
    if normalized.replace("⋅", "·").replace("․", "·") == "제·개정 경과":
        return "amendment_history"
    if (
        normalized.startswith(f"기업회계기준서 제{standard_id}호와 ")
        and "국제재무보고기준" in normalized
    ):
        return "ifrs_comparison"
    return current_zone


def _search_priority(zone: str) -> int:
    if zone in {"standard_body", "appendix_definitions", "application_guidance"}:
        return 3
    if zone in {"application_examples", "implementation_guidance"}:
        return 2
    if zone in {"basis_for_conclusions", "dissenting_opinion"}:
        return 1
    return 0


def _is_heading(text: str, paragraph_number: str | None) -> bool:
    if not text or paragraph_number:
        return False
    if LIST_MARKER_RE.match(text):
        return False
    if re.fullmatch(r".{1,80}\(제\d+(?:\.\d+)*절\)", text):
        return True
    if re.match(r"^제\d+(?:\.\d+)?(?:장|절)\s+", text):
        return True
    return len(text) <= 45 and not text.endswith((".", "다.", ")"))


class HwpxParser:
    def __init__(self, standard_id: str = "1109") -> None:
        self.standard_id = standard_id
        self._table_counter = 0
        self._footnote_counter = 0
        self._block_counter = 0

    def parse(self, path: str | Path) -> dict:
        source = Path(path)
        paragraphs: list[Paragraph] = []
        blocks: list[DocumentBlock] = []
        orphan_tables: list[dict] = []
        orphan_footnotes: list[dict] = []
        paragraph_occurrences: dict[str, int] = {}
        document_order = 0
        zone = "front_matter"
        section_path: list[str] = []
        current: Paragraph | None = None
        self._table_counter = 0
        self._footnote_counter = 0
        self._block_counter = 0

        with zipfile.ZipFile(source) as archive:
            section_names = sorted(
                name
                for name in archive.namelist()
                if re.fullmatch(r"Contents/section\d+\.xml", name)
            )
            for section_name in section_names:
                root = ET.fromstring(archive.read(section_name))

                top_level_paragraphs = [
                    child for child in root if local_name(child) == "p"
                ]
                for xml_index, paragraph_element in enumerate(top_level_paragraphs):
                    text = element_text(paragraph_element, SKIP_TEXT_ANCESTORS)
                    if not text and not list(_iter_descendants(paragraph_element, "tbl")):
                        continue
                    document_order += 1
                    next_zone = _zone_transition(text, zone, self.standard_id)
                    if next_zone != zone:
                        zone = next_zone
                        current = None
                        section_path = []

                    number_match = PARAGRAPH_NUMBER_RE.match(text)
                    number = (
                        number_match.group("number")
                        if number_match and zone != "front_matter"
                        else None
                    )
                    is_heading = _is_heading(text, number)

                    if is_heading:
                        if not section_path or section_path[-1] != text:
                            if re.match(r"^제\d+장", text):
                                section_path = [text]
                            elif re.match(r"^제\d+(?:\.\d+)?절", text):
                                section_path = section_path[:1] + [text]
                            elif section_path and section_path[0].startswith("부록"):
                                section_path = section_path[:1] + [text]
                            else:
                                section_path = section_path[:2] + [text]
                        current = None

                    if number:
                        body = normalize_space(text[number_match.end() :])
                        occurrence = paragraph_occurrences.get(number, 0) + 1
                        paragraph_occurrences[number] = occurrence
                        paragraph_id = f"KIFRS{self.standard_id}-{number}"
                        if occurrence > 1:
                            paragraph_id = f"{paragraph_id}__{occurrence}"
                        current = Paragraph(
                            paragraph_id=paragraph_id,
                            number=number,
                            zone=zone,
                            section_path=list(section_path),
                            text=body,
                            source_section=section_name,
                            xml_index=xml_index,
                            document_order=document_order,
                            occurrence=occurrence,
                        )
                        paragraphs.append(current)
                        block_type = "paragraph"
                    else:
                        list_match = LIST_MARKER_RE.match(text)
                        if list_match and current:
                            current.subitems.append(
                                Subitem(
                                    marker=list_match.group("marker"),
                                    text=normalize_space(text[list_match.end() :]),
                                    xml_index=xml_index,
                                )
                            )
                            block_type = "subitem"
                        elif is_heading:
                            block_type = "heading"
                        elif zone == "front_matter":
                            block_type = "front_matter"
                        elif current:
                            block_type = "continuation"
                        else:
                            block_type = "unnumbered"

                    footnotes = list(_iter_descendants(paragraph_element, "footNote"))
                    parsed_footnotes: list[Footnote] = []
                    for footnote_element in footnotes:
                        self._footnote_counter += 1
                        footnote = Footnote(
                            footnote_id=(
                                f"KIFRS{self.standard_id}-F-{self._footnote_counter:04d}"
                            ),
                            parent_paragraph_id=(current.paragraph_id if current else None),
                            number=footnote_element.get("number"),
                            text=element_text(footnote_element),
                        )
                        parsed_footnotes.append(footnote)
                        if current:
                            current.footnotes.append(footnote)
                        else:
                            orphan_footnotes.append(
                                {
                                    "zone": zone,
                                    "section_path": list(section_path),
                                    "source_section": section_name,
                                    "xml_index": xml_index,
                                    "footnote": asdict(footnote),
                                }
                            )

                    tables = list(_iter_descendants(paragraph_element, "tbl"))
                    parsed_tables: list[Table] = []
                    for table_element in tables:
                        self._table_counter += 1
                        table = _parse_table(
                            table_element,
                            f"KIFRS{self.standard_id}-T-{self._table_counter:04d}",
                            current.paragraph_id if current else None,
                        )
                        parsed_tables.append(table)
                        if current:
                            current.tables.append(table)
                        else:
                            orphan_tables.append(
                                {
                                    "zone": zone,
                                    "section_path": list(section_path),
                                    "source_section": section_name,
                                    "xml_index": xml_index,
                                    "table": asdict(table),
                                }
                            )

                    self._block_counter += 1
                    block = DocumentBlock(
                        block_id=f"KIFRS{self.standard_id}-BLK-{self._block_counter:05d}",
                        block_type=block_type,
                        zone=zone,
                        text=text,
                        source_section=section_name,
                        xml_index=xml_index,
                        document_order=document_order,
                        searchable=_search_priority(zone) > 0,
                        search_priority=_search_priority(zone),
                        section_path=list(section_path),
                        parent_paragraph_id=current.paragraph_id if current else None,
                        table_ids=[table.table_id for table in parsed_tables],
                        footnote_ids=[footnote.footnote_id for footnote in parsed_footnotes],
                        references=(
                            parse_references(text, source_standard=self.standard_id)
                            if zone != "front_matter"
                            else []
                        ),
                    )
                    blocks.append(block)
                    if current:
                        current.block_ids.append(block.block_id)

        block_lookup = {block.block_id: block for block in blocks}
        for paragraph in paragraphs:
            paragraph.references = [
                reference
                for block_id in paragraph.block_ids
                for reference in block_lookup[block_id].references
            ]

        return {
            "document": {
                "standard_id": self.standard_id,
                "title": {
                    "1032": "K-IFRS 제1032호 금융상품: 표시",
                    "1039": "K-IFRS 제1039호 금융상품: 인식과 측정",
                    "1107": "K-IFRS 제1107호 금융상품: 공시",
                    "1109": "K-IFRS 제1109호 금융상품",
                }.get(self.standard_id, f"K-IFRS 제{self.standard_id}호"),
                "source_file": source.name,
                "structure_source": "HWPX",
                "revision_policy": "exclude_deleted_include_inserted",
            },
            "paragraphs": [asdict(paragraph) for paragraph in paragraphs],
            "blocks": [asdict(block) for block in blocks],
            "orphan_tables": orphan_tables,
            "orphan_footnotes": orphan_footnotes,
        }


def search_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def _anchors(text: str, size: int = 32) -> list[str]:
    normalized = search_normalize(text)
    if len(normalized) <= size:
        return [normalized] if normalized else []
    positions = [0, len(normalized) // 3, (2 * len(normalized)) // 3, len(normalized) - size]
    return list(dict.fromkeys(normalized[position : position + size] for position in positions))


def map_pdf_pages(paragraphs: list[dict], pdf_path: str | Path) -> None:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [search_normalize(page.extract_text() or "") for page in reader.pages]

    for paragraph in paragraphs:
        table_text = [
            cell["text"]
            for table in paragraph["tables"]
            for cell in table["cells"]
        ]
        content = " ".join(
            [paragraph["number"], paragraph["text"]]
            + [item["text"] for item in paragraph["subitems"]]
            + [footnote["text"] for footnote in paragraph["footnotes"]]
            + table_text
        )
        anchors = _anchors(content)
        if not anchors:
            continue

        matches: list[tuple[int, int]] = []
        for page_index, page_text in enumerate(pages):
            hits = sum(anchor in page_text for anchor in anchors)
            if hits:
                matches.append((page_index + 1, hits))
        if not matches:
            continue

        seed_page, _ = max(matches, key=lambda item: (item[1], -item[0]))
        radius = max(3, min(12, ceil(len(search_normalize(content)) / 1000) + 2))
        candidate_pages = range(
            max(1, seed_page - radius), min(len(pages), seed_page + radius) + 1
        )
        anchor_pages: dict[str, int] = {}
        for anchor in anchors:
            exact_pages = [page for page in candidate_pages if anchor in pages[page - 1]]
            if exact_pages:
                anchor_pages[anchor] = min(exact_pages, key=lambda page: abs(page - seed_page))
                continue

            best_page = seed_page
            best_ratio = 0.0
            for page in candidate_pages:
                match = SequenceMatcher(
                    None, anchor, pages[page - 1], autojunk=False
                ).find_longest_match()
                ratio = match.size / len(anchor)
                if ratio > best_ratio:
                    best_page, best_ratio = page, ratio
            if best_ratio >= 0.78:
                anchor_pages[anchor] = best_page

        relevant_pages = list(anchor_pages.values()) or [seed_page]
        paragraph["pdf_page_start"] = min(relevant_pages)
        paragraph["pdf_page_end"] = max(relevant_pages)
        paragraph["page_match_confidence"] = round(
            len(anchor_pages) / len(anchors), 3
        )


def select_paragraphs(document: dict, numbers: Iterable[str]) -> list[dict]:
    wanted = set(numbers)
    return [paragraph for paragraph in document["paragraphs"] if paragraph["number"] in wanted]
