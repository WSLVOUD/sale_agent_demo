"""Language detection and response-language policy."""
from __future__ import annotations

import re
from typing import Dict, Optional

_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")

_LANGUAGE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("de", (
        "ich brauche", "wir brauchen", "brauchen", "bildschirm", "leinwand",
        "innenbereich", "außenbereich", "abstand", "mieten", "metern",
    )),
    ("fr", (
        "je cherche", "nous avons besoin", "besoin d", "écran", "ecran",
        "extérieur", "exterieur", "salle de",
    )),
    ("es", ("necesito", "pantalla", "centro comercial", "alquiler", "distancia de")),
    ("pt", ("preciso", "painel", "distância", "aluguel")),
    ("it", ("cerco", "schermo", "distanza", "esterno")),
    ("ru", ("нам нужен", "нужен", "экран", "расстояние")),
)


def detect_language(text: str) -> str:
    """Detect the customer's language for keyword selection and response policy."""
    if not text:
        return "en"
    if _KANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _ARABIC_RE.search(text):
        return "ar"
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _CJK_RE.search(text):
        return "zh"
    lowered = text.lower()
    for lang, hints in _LANGUAGE_HINTS:
        if any(hint in lowered for hint in hints):
            return lang
    return "en"


LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
}


def language_name(code: Optional[str]) -> str:
    """Return the language name, or the unknown code unchanged."""
    if not code:
        return "English"
    return LANGUAGE_NAMES.get(str(code).lower(), str(code))


def response_language_rule(language: Optional[str]) -> str:
    """Build the response-language instruction from the configured policy."""
    try:
        from src.config import config

        policy = str(getattr(config, "RESPONSE_LANGUAGE_POLICY", "en")).lower()
    except Exception:  # pragma: no cover - defensive fallback
        policy = "en"

    if policy == "auto" and language and str(language).lower() != "en":
        return f"Reply in {language_name(language)} (the customer's language)."
    return "ALWAYS use English, regardless of the customer's language."


__all__ = [
    "LANGUAGE_NAMES",
    "detect_language",
    "language_name",
    "response_language_rule",
    "_ARABIC_RE",
    "_CJK_RE",
    "_CYRILLIC_RE",
    "_HANGUL_RE",
    "_KANA_RE",
    "_LANGUAGE_HINTS",
]
