import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# .env laden (überschreibt NICHT bereits gesetzte Variablen)
load_dotenv(find_dotenv(), override=False)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

def get_env(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val
