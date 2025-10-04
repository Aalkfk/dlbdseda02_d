#!/usr/bin/env bash
set -euo pipefail

# Python 3.12 prüfen
pyv=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$pyv" != "3.12" ]; then
  echo "Warnung: Python $pyv gefunden – empfohlen ist Python 3.12 auf Ubuntu 24.04."
fi

# venv anlegen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Dependencies
pip install -r requirements.txt
[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true

# NLTK + spaCy Modelle (klein, schnell einsatzfähig)
python - <<'PY'
import nltk, sys
pkgs = ["punkt","stopwords","wordnet"]
for p in pkgs:
    try:
        nltk.data.find(p)
    except LookupError:
        nltk.download(p)
PY

python - <<'PY'
import sys, subprocess
for model in ("en_core_web_sm","de_core_news_sm"):
    try:
        __import__(model.replace("-","_"))
    except Exception:
        subprocess.check_call([sys.executable,"-m","spacy","download",model])
PY

# Jupyter Kernel (falls Notebooks)
python -m ipykernel install --user --name reddit-topics --display-name "reddit-topics (venv)" || true

echo "✔ Setup fertig. Aktiviere mit: source .venv/bin/activate"
