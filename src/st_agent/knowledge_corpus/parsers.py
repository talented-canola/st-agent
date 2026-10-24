"""Deterministic extraction only. Never execute code, translate or summarize sources."""

from __future__ import annotations

import ast
import csv
import io
import re

from .contracts import CorpusError, ParsedBlock, ParsedDocument, SourceSpec

PARSER_VERSION = "external-original-1"


def _python(spec, text):
    tree = ast.parse(text)
    blocks = {}

    def visit(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                if name in spec.selectors:
                    doc = ast.get_docstring(child, clean=False)
                    if doc:
                        literal = child.body[0]
                        blocks[name] = ParsedBlock(
                            text=doc,
                            kind="api_docstring",
                            locator={
                                "symbol": name,
                                "line_start": literal.lineno,
                                "line_end": literal.end_lineno,
                            },
                        )
                visit(child, name + ".")

    visit(tree)
    missing = set(spec.selectors) - blocks.keys()
    if missing or not spec.selectors:
        raise CorpusError("parse.symbol_missing:" + ",".join(sorted(missing)))
    return list(blocks.values()), {}


def _markdown(text):
    lines = text.splitlines(keepends=True)
    blocks, start, section, fenced = [], 0, "", False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        if not fenced and re.match(r"^#{1,6}\s+", line):
            content = "".join(lines[start:i]).strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        text=content,
                        kind="section",
                        locator={"section": section, "line_start": start + 1, "line_end": i},
                    )
                )
            start, section = i, line.strip()
    content = "".join(lines[start:]).strip()
    if content:
        blocks.append(
            ParsedBlock(
                text=content,
                kind="section",
                locator={"section": section, "line_start": start + 1, "line_end": len(lines)},
            )
        )
    return blocks, {}


def _jats(spec, raw):
    from defusedxml import ElementTree

    root = ElementTree.fromstring(raw)
    meta = root.find("./front/article-meta")
    if meta is None:
        raise CorpusError("parse.article_metadata_missing")
    identifiers = {
        n.get("pub-id-type"): "".join(n.itertext()).strip() for n in meta.findall("article-id")
    }
    doi = identifiers.get("doi", "").lower()
    pmc = identifiers.get("pmcid") or identifiers.get("pmc", "")
    if pmc and not pmc.startswith("PMC"):
        pmc = "PMC" + pmc
    if (spec.doi and doi != spec.doi.lower()) or (spec.pmcid and pmc != spec.pmcid):
        raise CorpusError("source.identity_mismatch")
    permission = meta.find("permissions")
    permission_text = " ".join(permission.itertext()) if permission is not None else ""
    permission_links = (
        " ".join(value for node in permission.iter() for value in node.attrib.values())
        if permission is not None
        else ""
    )
    if spec.license_marker not in permission_text + " " + permission_links or re.search(
        r"duration of|pandemic|permission.*required", permission_text, re.I
    ):
        raise CorpusError("source.license_mismatch")
    blocks = []

    def walk(node, path, section=()):
        if node.tag == "sec":
            heading = node.find("title")
            if heading is not None:
                section = (*section, "".join(heading.itertext()).strip())
        if node.tag in {"p", "table-wrap", "fig"}:
            # Preserve caption and table cell order; no generated narrative.
            if node.tag == "table-wrap":
                caption = node.find("caption")
                label = node.find("label")
                parts = [
                    "".join(item.itertext()).strip()
                    for item in (label, caption)
                    if item is not None
                ]
                for row in node.findall(".//tr"):
                    parts.append(
                        "\t".join(
                            "".join(cell.itertext()).strip()
                            for cell in row
                            if cell.tag in {"td", "th"}
                        )
                    )
                foot = node.find("table-wrap-foot")
                if foot is not None:
                    parts.append("".join(foot.itertext()).strip())
                content = "\n".join(parts).strip()
            else:
                content = "".join(node.itertext()).strip()
            if content:
                blocks.append(
                    ParsedBlock(
                        text=content,
                        kind={"p": "paragraph", "table-wrap": "table", "fig": "figure_caption"}[
                            node.tag
                        ],
                        locator={
                            "xml_path": path,
                            "element_id": node.get("id"),
                            "section": list(section),
                        },
                    )
                )
            return
        counts = {}
        for child in node:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            walk(child, f"{path}/{child.tag}[{counts[child.tag]}]", section)

    body = root.find("body")
    if body is not None:
        walk(body, "/article/body[1]")
    title = meta.find("title-group/article-title")
    return blocks, {
        "doi": doi,
        "pmcid": pmc,
        "title": "".join(title.itertext()) if title is not None else "",
        "permission_text": permission_text,
    }


def _html(spec, text):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    main = soup.select_one(spec.html_selector)
    if main is None:
        raise CorpusError("parse.main_content_missing")
    for node in main.select("script, style, nav"):
        node.decompose()
    blocks = []
    for node in main.find_all(["p", "pre", "table", "dt", "dd"]):
        if node.find_parent(["pre", "table", "dd"]) is not None:
            continue
        content = node.get_text(" ", strip=True)
        if content:
            blocks.append(
                ParsedBlock(
                    text=content,
                    kind=node.name,
                    locator={
                        "selector": spec.html_selector,
                        "ordinal": len(blocks),
                        "anchor": node.get("id"),
                    },
                )
            )
    return blocks, {}


def _tsv(text):
    lines = text.splitlines()
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if not rows or len(rows) != len(lines):
        raise CorpusError("parse.multiline_tsv_not_supported")
    if any(len(row) != len(rows[0]) for row in rows):
        raise CorpusError("parse.tsv_schema_mismatch")
    return [
        ParsedBlock(
            text=lines[0] + "\n" + line,
            kind="record",
            locator={"line_start": i + 2, "header_line": 1},
        )
        for i, line in enumerate(lines[1:])
        if line.strip()
    ], {"columns": rows[0]}


def parse_source(spec: SourceSpec, raw: bytes) -> ParsedDocument:
    try:
        text = raw.decode("utf-8-sig")
        if spec.format == "python_docstrings":
            blocks, metadata = _python(spec, text)
        elif spec.format == "jats":
            blocks, metadata = _jats(spec, raw)
        elif spec.format in {"markdown", "obo"}:
            # OBO remains external-original stanzas; graph interpretation is out of scope.
            blocks, metadata = (
                _markdown(text)
                if spec.format == "markdown"
                else (
                    [
                        ParsedBlock(
                            text=text.strip(),
                            kind="ontology_original",
                            locator={"line_start": 1, "line_end": len(text.splitlines())},
                        )
                    ]
                    if text.strip()
                    else [],
                    {},
                )
            )
        elif spec.format == "html":
            blocks, metadata = _html(spec, text)
        else:
            blocks, metadata = _tsv(text)
        if not blocks:
            raise CorpusError("parse.empty_content")
        return ParsedDocument(
            parser_version=PARSER_VERSION, blocks=tuple(blocks), metadata=metadata
        )
    except CorpusError:
        raise
    except Exception as error:
        raise CorpusError("parse.invalid_source:" + type(error).__name__) from error
