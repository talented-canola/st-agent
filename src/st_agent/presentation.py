"""Public prose is separate from the internally attributed answer."""

import re


def public_answer(answer: str) -> str:
    text = re.sub(r"\[(?:事实|解释|推断|待确认|假设)\]\s*", "", answer)
    text = re.sub(r"\[fact:[A-Za-z0-9_-]+\]", "", text)
    return re.sub(r"[ \t]+([，。；！？])", r"\1", text).strip()
