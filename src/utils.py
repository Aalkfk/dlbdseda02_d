"""
utils.py
========
Zentrale Utilities für Projektpfade und Environment-Variablen.

- Lädt (falls vorhanden) eine `.env` per `python-dotenv`, überschreibt
  dabei **keine** bereits gesetzten Umgebungsvariablen.
- Stellt den Projekt-Datenpfad `DATA_DIR` bereit (../data relativ zu diesem Modul).
- Bietet `get_env()` zum fail-fast Auslesen von Env-Variablen.

Dieses Modul wird früh importiert; side effects (Anlegen von `DATA_DIR`)
sind bewusst und stabil.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# .env laden (überschreibt **nicht** bereits gesetzte Variablen).
# So können lokale .env-Werte verwendet werden, ohne CI/Prod zu beeinflussen.
load_dotenv(find_dotenv(), override=False)

# Pfad zum Datenverzeichnis (../data) und sicherstellen, dass es existiert.
DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)


def get_env(name: str, default: str = "") -> str:
    """
    Liefert den Wert einer Environment-Variable oder wirft einen Fehler,
    falls (nach Fallback) ein leerer Wert vorliegt.

    Args:
        name: Name der Env-Variable (z. B. "REDDIT_CLIENT_ID").
        default: Optionaler Fallback-Wert. Achtung: Ein leerer Default
                 führt weiterhin zu einem Fehler (fail-fast).

    Returns:
        Der nicht-leere Stringwert der Env-Variable (oder des Defaults).

    Raises:
        RuntimeError: Wenn kein Wert gefunden wurde **oder** der Wert leer ist.
    """
    val = os.getenv(name, default)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val


__all__ = ["DATA_DIR", "get_env"]
