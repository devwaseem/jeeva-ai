"""Terminal-safe rendering for data-derived text."""

import unicodedata
from typing import Any


def terminal_text(value: Any) -> str:
    """Render control characters visibly instead of emitting terminal commands."""
    text = str(value)
    rendered: list[str] = []
    for character in text:
        codepoint = ord(character)
        if unicodedata.category(character).startswith("C"):
            if codepoint <= 0xFF:
                rendered.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                rendered.append(f"\\u{codepoint:04x}")
            else:
                rendered.append(f"\\U{codepoint:08x}")
        else:
            rendered.append(character)
    return "".join(rendered)
