"""The language an agent writes its messages in, decided from the user's own words instead of left to the model.

A model that is told "write in the user's language" often does not (members of a team answered in English to a request in
Chinese). So the worker looks at the text it was given once, and when it is sure, says the language outright in the system
prompt. When it is not sure it says nothing and the general rule stays.
"""

from __future__ import annotations

import re

# Code, links and paths say nothing about the language a person writes in.
_NOISE = re.compile(r"```.*?```|`[^`\n]*`|https?://\S+|(?:/[\w.@-]+){2,}", re.DOTALL)
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA_HANGUL = re.compile(r"[぀-ヿ가-힯]")
_LATIN = re.compile(r"[A-Za-z]")

# Fewer letters than this is too little to tell ("ok", "好").
MIN_LETTERS = 8
# Share of Han characters among the letters. Chinese text with many English product names is still well over the first; English
# text with a Chinese name or two is well under the second. Between them the text is mixed and nothing is decided.
CHINESE_FROM = 0.3
ENGLISH_UP_TO = 0.05

# The line for each language the worker is sure of.
PROMPT_LINES = {
    "zh": "Write every message (progress, replies, handovers, the final answer) in Simplified Chinese (简体中文). Code, "
    "commands and file names stay as they are.",
    "en": "Write every message (progress, replies, handovers, the final answer) in English. Code, commands and file names stay "
    "as they are.",
}


def detect_language(*texts: str) -> str:
    """`zh` or `en` when the texts, taken together, are clearly one of them; an empty string when they are too short, mixed, or
    in another script (Japanese kana, Korean)."""
    text = _NOISE.sub(" ", "\n".join(texts))
    han, latin = len(_HAN.findall(text)), len(_LATIN.findall(text))
    letters = han + latin
    if letters < MIN_LETTERS or len(_KANA_HANGUL.findall(text)) * 10 >= letters:
        return ""
    share = han / letters
    if share >= CHINESE_FROM:
        return "zh"
    if share <= ENGLISH_UP_TO:
        return "en"
    return ""


def language_line(language: str) -> str:
    """The line to add to the system prompt for `language`, or none when it is not one the worker is sure of."""
    return PROMPT_LINES.get(language, "")
