"""Document-structure detection & structure-aware sentence segmentation.

Real documents — especially financial reports, contracts and technical specs —
are NOT plain prose.  They contain headings, numbered sections, tables, and
bullet lists.  Naive sentence splitting destroys that structure, which is the
single biggest cause of bad chunks on hard documents.

This module fixes the *segmentation* (the foundation of the whole pipeline):

  * It is LINE-AWARE: headings and table/list rows are kept as their own atomic
    units instead of being glued into the surrounding prose by punkt.
  * It DETECTS HEADINGS broadly (markdown #, numbered "1.2 …", SEC "Item 7",
    "Note 5", ALL-CAPS lines, legal Article/Section/Chapitre, short title lines).
  * Consecutive table/list rows are grouped into a single block so a table stays
    one retrievable unit instead of exploding into dozens of tiny fragments.

The chunkers then treat the gap *before every heading* as a HARD boundary that
is always opened — so chunks never straddle a section break.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_FR_WORDS = {
    "le", "la", "les", "des", "de", "du", "et", "à", "pour", "dans", "qui",
    "que", "une", "un", "est", "sont", "aux", "par", "sur", "ce", "cette",
    "au", "en", "ne", "pas", "plus", "ou", "se", "son", "ses", "leur",
}

# Abbreviations that must NOT end a sentence.
_PROTECT = [
    (re.compile(r"\bArt\.\s"), "Art<DOT> "),
    (re.compile(r"\bart\.\s"), "art<DOT> "),
    (re.compile(r"\bn°\s*"), "n° "),
    (re.compile(r"\bM\.\s"), "M<DOT> "),
    (re.compile(r"\bMM\.\s"), "MM<DOT> "),
    (re.compile(r"\bp\.\s"), "p<DOT> "),
    (re.compile(r"\bal\.\s"), "al<DOT> "),
    (re.compile(r"\bparag\.\s"), "parag<DOT> "),
    (re.compile(r"\bNo\.\s"), "No<DOT> "),
    (re.compile(r"\bInc\.\s"), "Inc<DOT> "),
    (re.compile(r"\bCorp\.\s"), "Corp<DOT> "),
    (re.compile(r"\bvs\.\s"), "vs<DOT> "),
    (re.compile(r"\be\.g\.\s"), "e<DOT>g<DOT> "),
    (re.compile(r"\bi\.e\.\s"), "i<DOT>e<DOT> "),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# Explicit heading vocabulary (legal + financial + generic section words).
# Division words (titre/chapitre/section/chapter/part) REQUIRE a number or
# ordinal after them: "Titre II" is a heading, but a wrapped line starting with
# "titre des frais…" is ordinary prose and must not match.
_HEADING_WORD_RE = re.compile(
    r"^\s*("
    r"art\.\s*\d|article\s+(premier|\d|[ivxlcm]+)|"
    r"(chapitre|titre|chapter)\s+(premier|premi[eè]re|[ivxlcm]+\b|\d+)|"
    r"section\s+(premi[eè]re|[ivxlcm]+\b|\d+)|"
    r"annexe\b|appendix\b|schedule\b|exhibit\b|"
    r"item\s+\d+[a-z]?\b|note\s+\d+\b|notes?\s+to\b|part\s+[ivxlcm\d]+\b|"
    r"(premi[eè]re|deuxi[eè]me|troisi[eè]me|quatri[eè]me|cinqui[eè]me|sixi[eè]me|"
    r"septi[eè]me|huiti[eè]me|neuvi[eè]me|dixi[eè]me|onzi[eè]me)\s+(section|partie)"
    r")",
    re.IGNORECASE,
)
# Numbered section like "1." / "1.2" / "2.3.4 Title" / "A. Title" / "B) Title".
# A single letter MUST be followed by '.' or ')' so the article "A"/"I" at the
# start of an ordinary sentence is not mistaken for a heading marker.
_NUM_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+){0,3}[\).]?|[A-Z][\).])\s+\S")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def detect_language(text: str) -> str:
    """Detect english / french / arabic / cjk so we can pick the right sentence
    splitter (punkt has no Arabic/CJK model, so those use a script-aware regex)."""
    sample = text[:4000]
    total = max(len(sample), 1)
    arabic = len(re.findall(r"[؀-ۿ]", sample))
    cjk = len(re.findall(r"[一-鿿぀-ヿ가-힯]", sample))
    if arabic / total > 0.10:
        return "arabic"
    if cjk / total > 0.10:
        return "cjk"
    toks = re.findall(r"[A-Za-zÀ-ÿ]+", sample.lower())
    if not toks:
        return "english"
    fr = sum(1 for t in toks if t in _FR_WORDS)
    return "french" if fr / max(len(toks), 1) > 0.07 else "english"


def _generic_split(blob: str, lang: str) -> List[str]:
    """Script-aware sentence split for languages punkt doesn't support."""
    if lang == "cjk":
        parts = re.split(r"(?<=[。！？\.!?])", blob)
    else:  # arabic (and any other fallback): split on Arabic + Latin enders
        parts = re.split(r"(?<=[\.!\?؟۔])\s+", blob)
    return [p.strip() for p in parts if p.strip()]


def is_heading(line: str) -> bool:
    """True if a standalone line looks like a section heading."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _MD_HEADING_RE.match(s):
        return True
    # Real headings never start with a lowercase letter; PDF line-wrap
    # fragments ("titre des frais…", "section des produits…") do, and they
    # must not become hard boundaries or breadcrumbs.
    first_alpha = next((c for c in s if c.isalpha()), "")
    if first_alpha.islower():
        return False
    if _HEADING_WORD_RE.match(s):
        return True
    words = s.split()
    if len(words) <= 12 and _NUM_HEADING_RE.match(s):
        return True
    # ALL-CAPS short line (e.g. "BALANCE SHEET", "RISK FACTORS").
    letters = [c for c in s if c.isalpha()]
    if letters and len(words) <= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return True
    # Short Title-Case line with no terminal punctuation ("Cash Flow Summary").
    if (len(words) <= 9 and s[-1] not in ".!?:;,"
            and sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.6
            and any(c.isalpha() for c in s)):
        return True
    return False


def is_table_or_list_row(line: str) -> bool:
    """True for table rows / list items that should stay atomic, not be parsed
    as prose (e.g. 'Revenue    1,234   1,180', '| A | B |', '- bullet')."""
    s = line.strip()
    if not s:
        return False
    if s.count("|") >= 2:
        return True
    if re.match(r"^\s*([-*•·]|\(?[a-z0-9]{1,3}[\.\)])\s+\S", s):  # bullets / (a) (1)
        return True
    if "\t" in s or re.search(r"\S {2,}\S", s):  # column gaps via runs of spaces/tabs
        # ...but only if it carries a number (typical of financial line items)
        if re.search(r"\d", s):
            return True
    return False


def _punkt_split(blob: str, lang: str) -> List[str]:
    # Arabic / CJK: punkt has no model, route to the script-aware splitter.
    if lang in ("arabic", "cjk"):
        return _generic_split(blob, lang)
    protected = blob
    for pat, repl in _PROTECT:
        protected = pat.sub(repl, protected)
    try:
        import nltk

        for res in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(res)
            except LookupError:
                try:
                    nltk.download(res.split("/")[-1], quiet=True)
                except Exception:
                    pass
        try:
            sents = nltk.sent_tokenize(protected, language=lang)
        except Exception:
            sents = nltk.sent_tokenize(protected)
    except Exception:
        sents = _SENT_SPLIT.split(protected)

    out: List[str] = []
    for s in sents:
        s = s.replace("<DOT>", ".").replace(" ", " ").strip()
        if not s:
            continue
        if len(s) > 600:  # runaway "sentence" — break on clause punctuation
            out.extend(p.strip() for p in re.split(r"(?<=[,;:])\s+", s) if p.strip())
        else:
            out.append(s)
    return out


def segment(text: str, language: Optional[str] = None) -> Tuple[List[str], List[bool]]:
    """Structure-aware segmentation.

    Returns (sentences, is_heading_flags) where headings and table/list blocks
    are kept as their own atomic units and prose is punkt-split.
    """
    text = text.strip()
    if not text:
        return [], []
    lang = language or detect_language(text)

    sents: List[str] = []
    flags: List[bool] = []
    prose: List[str] = []
    table: List[str] = []

    def flush_prose():
        if prose:
            for s in _punkt_split(" ".join(prose), lang):
                sents.append(s)
                flags.append(False)
            prose.clear()

    def flush_table():
        if table:
            sents.append(" \n".join(table).strip())
            flags.append(False)
            table.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            flush_table(); flush_prose()
            continue
        if is_heading(line):
            flush_table(); flush_prose()
            sents.append(line); flags.append(True)
        elif is_table_or_list_row(line):
            flush_prose()
            table.append(line)
        else:
            flush_table()
            prose.append(line)
    flush_table(); flush_prose()

    if not sents:
        return [text], [False]
    return sents, flags


def split_sentences(text: str, language: Optional[str] = None) -> List[str]:
    """Backwards-compatible wrapper returning just the sentence list."""
    return segment(text, language)[0]


def heading_gaps(sentences: List[str]) -> set:
    """Gap indices i (split after sentence i) that precede a heading sentence —
    i.e. HARD boundaries that the chunkers must always open."""
    gaps = set()
    for i in range(1, len(sentences)):
        if is_heading(sentences[i]):
            gaps.add(i - 1)
    return gaps


def heading_level(line: str) -> int:
    """Approximate hierarchy depth of a heading (1 = top).  Markdown depth is
    explicit; numbered headings use their dotted depth; document-level words
    (ITEM/PART/CHAPTER/ALL-CAPS) rank above note/article/title-case lines."""
    s = line.strip()
    m = re.match(r"^\s{0,3}(#{1,6})\s+\S", s)
    if m:
        return len(m.group(1))
    m = re.match(r"^\s*(\d+(?:\.\d+){0,3})[\).]?\s+\S", s)
    if m:
        return 1 + m.group(1).count(".")
    if re.match(r"^\s*(item\s+\d|part\s+[ivxlcm\d]|chapitre|chapter|titre\s)", s, re.IGNORECASE):
        return 1
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return 1
    if re.match(r"^\s*(art\.|article|section|annexe|appendix|schedule|exhibit|note\s+\d)", s, re.IGNORECASE):
        return 2
    return 3


# "Art. 18 - Est créé un mécanisme…" -> breadcrumb component "Art. 18".
_ARTICLE_SHORT_RE = re.compile(
    r"^\s*((?:art\.|article)\s*(?:premier|\d+(?:\s*(?:bis|ter|quater))?|[ivxlcm]+))\b[\s\.\-–:]*",
    re.IGNORECASE,
)
_CRUMB_MAX = 60


def _crumb(heading: str) -> str:
    """Short breadcrumb component: article headings keep only their number;
    everything else is clipped so a breadcrumb never drags a whole sentence."""
    s = heading.strip()
    m = _ARTICLE_SHORT_RE.match(s)
    if m:
        return m.group(1)
    return s if len(s) <= _CRUMB_MAX else s[: _CRUMB_MAX - 1].rstrip() + "…"


# A bare division marker ("TITRE II", "Chapitre 3") whose caption sits on the
# NEXT line — the two lines form one heading and must share one breadcrumb.
_BARE_DIVISION_RE = re.compile(
    r"^\s*(titre|chapitre|chapter|section|partie|part|item)\s+\S{1,12}\s*$",
    re.IGNORECASE,
)


def section_paths(sentences: List[str]) -> List[str]:
    """Heading breadcrumb for every sentence: the path of enclosing headings,
    e.g. "ITEM 7. MD&A > Liquidity".  Attaching this to each chunk's metadata is
    one of the highest-impact known tricks for retrieval."""
    stack: List[Tuple[int, str]] = []  # (level, short heading text)
    out: List[str] = []
    prev_was_heading = False
    for s in sentences:
        if is_heading(s):
            # Only a plain caption line merges into a bare division above it;
            # structural markers (Art. N, Note N, 1.2 …) start their own level.
            is_caption = (not _HEADING_WORD_RE.match(s.strip())
                          and not _NUM_HEADING_RE.match(s.strip()))
            if (prev_was_heading and is_caption and stack
                    and _BARE_DIVISION_RE.match(stack[-1][1])):
                lvl, txt = stack[-1]  # caption line joins its bare division
                stack[-1] = (lvl, _crumb(txt + " — " + s.strip()))
            else:
                lvl = heading_level(s)
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, _crumb(s)))
            prev_was_heading = True
        else:
            prev_was_heading = False
        out.append(" > ".join(t for (_, t) in stack))
    return out
