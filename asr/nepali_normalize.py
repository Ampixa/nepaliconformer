"""Nepali ASR target normalisation — everything in Devanagari, verbatim.

Why this exists: the corpus mixes two conventions for the same spoken content.
207,895 rows write number WORDS (बीस) and 60,992 write NUMERALS (२०). The model
hears /bis/ either way and has no acoustic cue to choose, so 7.7% of the corpus
is an unlearnable coin flip. Someone ran inverse text normalisation into the
labels; ASR targets should be verbatim and ITN belongs downstream in the LM,
where there is context to decide whether "बीस रुपैयाँ" should render as "Rs 20".

Target convention:
    Devanagari letters + matras + danda ONLY
    numerals  -> spoken words     (२० -> बीस)
    currency  -> spoken words     (₹ -> रुपैयाँ)
    Latin     -> transliterated via the code-switch lexicon, else dropped
    junk      -> stripped (212 of 310 corpus characters are scraping noise:
                 Cyrillic, Latin diacritics, currency symbols, punctuation)

NUM_21_99 is written to standard Nepali orthography, not Hindi. Points where the
two diverge and this table follows Nepali:
  दश   not दस        (श, not स)
  अठार not अठारह     (no final ह)
  नasalisation uses CHANDRABINDU (ँ) throughout — चौँतीस, पैँतीस, सैँतीस,
  पैँतालीस, चौँसट्ठी, पैँसट्ठी — never anusvara (ं). Mixing the two is the most
  common error in Devanagari number tables and was present in the first draft
  of this one.
Irregular by construction: 39/49/59/69/79/89/99 are उन- forms (one-less-than the
next ten), not compositional. Verified against the corpus, where these forms
appear as spoken words in 207,895 rows.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- numbers

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

NUM_0_20 = [
    "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ", "दश",
    "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार", "उन्नाइस", "बीस",
]

NUM_21_99 = {
    21: "एक्काइस", 22: "बाइस", 23: "तेइस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस",
    27: "सत्ताइस", 28: "अट्ठाइस", 29: "उनन्तीस", 30: "तीस",
    31: "एकतीस", 32: "बत्तीस", 33: "तेत्तीस", 34: "चौँतीस", 35: "पैँतीस", 36: "छत्तीस",
    37: "सैँतीस", 38: "अठतीस", 39: "उनन्चालीस", 40: "चालीस",
    41: "एकचालीस", 42: "बयालीस", 43: "त्रिचालीस", 44: "चवालीस", 45: "पैँतालीस",
    46: "छयालीस", 47: "सतचालीस", 48: "अठचालीस", 49: "उनन्चास", 50: "पचास",
    51: "एकाउन्न", 52: "बाउन्न", 53: "त्रिपन्न", 54: "चवन्न", 55: "पचपन्न", 56: "छपन्न",
    57: "सन्ताउन्न", 58: "अन्ठाउन्न", 59: "उनन्साठी", 60: "साठी",
    61: "एकसट्ठी", 62: "बयसट्ठी", 63: "त्रिसट्ठी", 64: "चौँसट्ठी", 65: "पैँसट्ठी",
    66: "छयसट्ठी", 67: "सतसट्ठी", 68: "अठसट्ठी", 69: "उनन्सत्तरी", 70: "सत्तरी",
    71: "एकहत्तर", 72: "बहत्तर", 73: "त्रिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छयहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "असी",
    81: "एकासी", 82: "बयासी", 83: "त्रियासी", 84: "चौरासी", 85: "पचासी",
    86: "छयासी", 87: "सतासी", 88: "अठासी", 89: "उनान्नब्बे", 90: "नब्बे",
    91: "एकानब्बे", 92: "बयानब्बे", 93: "त्रियानब्बे", 94: "चौरानब्बे", 95: "पन्चानब्बे",
    96: "छयानब्बे", 97: "सन्तानब्बे", 98: "अन्ठानब्बे", 99: "उनान्सय",
}

# South Asian scale, not Western. लाख = 10^5 and करोड = 10^7 — a thousands-based
# grouping would verbalise large numbers wrongly.
SCALES = [(10_000_000, "करोड"), (100_000, "लाख"), (1_000, "हजार"), (100, "सय")]


def number_to_words(n: int) -> str:
    if n < 0:
        return "माइनस " + number_to_words(-n)
    if n <= 20:
        return NUM_0_20[n]
    if n < 100:
        return NUM_21_99[n]
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = f"{number_to_words(head)} {name}"
            return f"{out} {number_to_words(rest)}" if rest else out
    return NUM_0_20[n]


def digits_to_words(tok: str) -> str:
    """A digit string as spoken. Long strings (phone numbers, IDs) are read
    digit-by-digit, which is what speakers actually do — 'नौ आठ शून्य...' — rather
    than as a single enormous quantity."""
    tok = tok.translate(DEV_DIGITS)
    if not tok.isdigit():
        return tok
    if len(tok) > 6 or (len(tok) > 1 and tok[0] == "0"):
        return " ".join(NUM_0_20[int(c)] for c in tok)
    return number_to_words(int(tok))


# ---------------------------------------------------------------- edge cases

# Symbols spoken as words. Currency first: the corpus contains ₹ and Rs forms that
# would otherwise be stripped, silently deleting a word the speaker said.
# Symbols only. Devanagari abbreviations like रु are deliberately ABSENT: as bare
# substrings they match inside real words (रुपैया -> "रुपैयाँ पैया") and, worse,
# inside their own replacement, cascading रुपैयाँ -> रुपैयाँ पैयाँ. Anything that
# is already a Devanagari word is left alone — it is already verbatim.
SYMBOL_WORDS = {
    "₹": "रुपैयाँ", "$": "डलर", "£": "पाउन्ड", "€": "युरो", "¥": "येन",
    "%": "प्रतिशत", "&": "र", "+": "जोड", "=": "बराबर",
    "°": "डिग्री", "@": "एट",
}
# One alternation, one pass. Sequential str.replace re-scans its own output.
_SYMBOL_RE = re.compile("|".join(re.escape(k) for k in sorted(SYMBOL_WORDS, key=len, reverse=True)))

# The trailing बजे is CONSUMED, not left behind: "१०:११ बजे सम्म" must become
# "दश बजेर एघार मिनेट सम्म", not "... मिनेट बजे सम्म", which is ungrammatical.
_TIME = re.compile(r"(?<!\d)([0-9०-९]{1,2})[:.]([0-9०-९]{2})(?!\d)(\s*बजे)?")
_DECIMAL = re.compile(r"(?<!\d)([0-9०-९]+)[.]([0-9०-९]+)(?!\d)")
_NUMBER = re.compile(r"[0-9०-९]+")
# Thousands separators must be removed BEFORE number parsing. Without this, "20,000" splits
# into "20" and "000", and the leading-zero rule then reads the second group digit-by-digit:
# "बीस शून्य शून्य शून्य". Measured on Chirp 2 output, which writes grouped digits where our
# references use words. South Asian grouping ("2,00,000") is affected the same way. Only
# commas BETWEEN digits are removed, so a comma used as punctuation is untouched.
_THOUSANDS = re.compile(r"(?<=[0-9०-९]),(?=[0-9०-९])")
# Times written WITHOUT a separator: "१०११ बजे" is 10:11, not 1011. Seen in the
# corpus and verbalised as "एक हजार एघार बजे" before this rule existed. A 3-4
# digit run immediately followed by बजे is a clock time, so split it HH|MM.
_BARE_TIME = re.compile(r"(?<![0-9०-९])([0-9०-९]{3,4})\s*बजे")


def _expand_bare_time(m: re.Match) -> str:
    d = m.group(1).translate(DEV_DIGITS)
    h, mn = (d[:-2], d[-2:])
    if not (0 < int(h) <= 24 and int(mn) < 60):
        return m.group(0)          # not a plausible time; leave for the number rule
    return f"{digits_to_words(h)} बजेर {digits_to_words(mn)} मिनेट "
_KEEP = re.compile(r"[^ऀ-ॿ ]")          # Devanagari block + space
_DANDA_OK = re.compile(r"[^ऀ-ॿ ।]")
# The danda U+0964 and double danda U+0965 lie INSIDE the Devanagari block ऀ-ॿ (U+0900-U+097F),
# so _KEEP preserves them and _DANDA_OK only re-adds a character that was never removed. The two
# regexes are therefore identical in effect and `keep_danda=False` did nothing at all — verified
# 2026-08-04 by scoring FLEURS both ways and getting byte-identical WER. Stripping now happens
# explicitly below.
_DANDA_CHARS = re.compile(r"[।॥]")


def _expand_time(m: re.Match) -> str:
    h, mn = digits_to_words(m.group(1)), digits_to_words(m.group(2))
    return f"{h} बजेर {mn} मिनेट"


def _expand_decimal(m: re.Match) -> str:
    whole = digits_to_words(m.group(1))
    frac = " ".join(NUM_0_20[int(c)] for c in m.group(2).translate(DEV_DIGITS))
    return f"{whole} दशमलव {frac}"


def normalize(text: str, lexicon: dict[str, str] | None = None,
              keep_danda: bool = True) -> str:
    """Normalise one ASR target to verbatim Devanagari.

    `lexicon` maps Devanagari -> English (the project's 963-entry code-switch
    lexicon). It is inverted here to bring stray Latin BACK into Devanagari, so
    that the 665.6 h of English-as-Devanagari already in the corpus keeps a single
    consistent spelling instead of competing with Latin forms.
    """
    s = unicodedata.normalize("NFC", str(text))

    # Latin -> Devanagari where the lexicon knows the word; otherwise it is
    # dropped by the character filter below. Longest-first so multi-word entries
    # are not partially matched.
    if lexicon:
        rev = {v.lower(): k for k, v in lexicon.items()}
        def _lat(m: re.Match) -> str:
            return rev.get(m.group(0).lower(), " ")
        s = re.sub(r"[A-Za-z]+", _lat, s)

    s = _THOUSANDS.sub("", s)
    s = _SYMBOL_RE.sub(lambda m: f" {SYMBOL_WORDS[m.group(0)]} ", s)

    s = _TIME.sub(_expand_time, s)
    s = _BARE_TIME.sub(_expand_bare_time, s)
    s = _DECIMAL.sub(_expand_decimal, s)
    s = _NUMBER.sub(lambda m: f" {digits_to_words(m.group(0))} ", s)

    s = _DANDA_OK.sub(" ", s)
    if not keep_danda:
        s = _DANDA_CHARS.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    cases = [
        ("मानौ कसैले मलाई २० रुपैया माग्यो", "numeral -> word"),
        ("त्यो मात्रै १६ वर्षको छोरी", "teens"),
        ("सुत्यो १०:११ बजे सम्म", "time"),
        ("मूल्य ₹1450 हो", "currency + numeral"),
        ("९८४१२३४५६७ मा फोन गर्नुस्", "phone read digit-by-digit"),
        ("२.५ प्रतिशत बढ्यो", "decimal + percent"),
        ("१ लाख ५० हजार", "south asian scale"),
        ("यो online र account को कुरा हो", "latin -> devanagari via lexicon"),
        ("अनि ३३ लाख ३० लाख भन्दा बढी", "large numbers"),
        ("२०,००० वटा फिल्म", "thousands separator (was: बीस शून्य शून्य शून्य)"),
        ("2,00,000 रुपैयाँ", "south asian grouping"),
    ]
    lex = {"अनलाइन": "online", "अकाउन्ट": "account"}
    for t, why in cases:
        print(f"{why:<34} {t}\n{'':<34} -> {normalize(t, lex)}\n")
