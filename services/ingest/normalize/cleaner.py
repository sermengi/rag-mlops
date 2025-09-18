from typing import List, Protocol
import re
import unicodedata

class Normalizer(Protocol):
    def __call__(self, text: str) -> str: ...

# Map of zero-width code points to None for str.translate
ZERO_WIDTH: dict[int, None] = dict.fromkeys(
    [
        0x200B, 0x200C, 0x200D, 0xFEFF,  # zero-width space/marks/BOM
        0x2060,                          # word joiner
        0x00AD,                          # soft hyphen
    ],
    None,
)

def nfk_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)

def normalize_whitespace(text: str) -> str:
    # Keep paragraph breaks: convert any CRLF to LF, trim spaces at EOL
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def dehyphenate_linebreaks(text: str) -> str:
    # Join "hy-\nphen" → "hyphen" but NOT if next line starts with capital letter mid-sentence? keep simple first
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def unwrap_paragraphs(text: str) -> str:
    # Replace single newlines inside paragraphs with spaces, keep double newlines
    # Do not unwrap if line looks like list item or heading.
    lines: List[str] = text.split("\n")
    out: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            out.append(" ".join(s.strip() for s in buf))
            buf.clear()
    for line in lines:
        if not line.strip():  # blank line -> paragraph break
            flush()
            out.append("")
        elif re.match(r"^\s*([\-–•\*]|\d+[\.\)])\s+", line):  # list item
            flush()
            out.append(line.strip())
        else:
            buf.append(line)
    flush()
    return "\n".join(out)

def normalize_bullets_quotes(text: str) -> str:
    table = str.maketrans({
        "•": "-", "–": "-", "—": "-", "“": '"', "”": '"', "’": "'",
    })
    return text.translate(table)

def looks_like_code_or_table(text: str) -> bool:
    sample = text[:2000]
    signal = sum(sample.count(ch) for ch in ["|", "`", "{", "}", ";"])
    return signal > 40

DEFAULT_PIPELINE: List[Normalizer] = [
    nfk_normalize,
    normalize_whitespace,
    dehyphenate_linebreaks,
    unwrap_paragraphs,
    normalize_bullets_quotes,
]

def normalize_text(text: str, *, aggressive: bool = True) -> str:
    if not aggressive or looks_like_code_or_table(text):
        # Minimal normalization only
        for fn in [nfk_normalize, normalize_whitespace]:
            text = fn(text)
        return text
    for fn in DEFAULT_PIPELINE:
        text = fn(text)
    return text
