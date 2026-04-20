import re
from cs_abbreviations import CS_ABBREVIATIONS

# Keys are already sorted longest-first in the dict — preserve that order
_KEYS = list(CS_ABBREVIATIONS.keys())

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _KEYS) + r")\b(?!\s*\()",
    flags=re.IGNORECASE,
)


def expand_cs_abbreviations(text: str) -> str:
    """
    Expand known CS / ICSI abbreviations inline.
    E.g. "The NCLT passed the order" →
         "The NCLT (National Company Law Tribunal) passed the order"

    Already-expanded terms (followed by a parenthesis) are left untouched.
    """
    if not text:
        return text

    def replacer(match: re.Match) -> str:
        key      = match.group(0)
        full_form = CS_ABBREVIATIONS.get(key.upper())
        return f"{key} ({full_form})" if full_form else key

    return _PATTERN.sub(replacer, text)