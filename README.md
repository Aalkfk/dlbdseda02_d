<div id="readme-top"></div>

# Stuttgart Reddit Topic Explorer

Automatisches Sammeln und Auswerten von Posts aus **r/Stuttgart**: Preprocessing, Topic Modeling (LDA), intelligente Label-Generierung mit Duplikat-Vermeidung und ein moderner **HTML-Report** mit DE/EN-Vergleichen und Statistiken.  
Der Report zeigt u. a. die aktivsten Flairs und Nutzer:innen, pro Flair die Top-Topics (nebeneinander: DE/EN) sowie ganz unten den gesamten Konsolen-Log.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#environment-variables">Environment Variables</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#outputs">Outputs</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## About The Project

Dieses Projekt crawlt Posts und (optional) Kommentare aus **r/Stuttgart**, bereitet die Texte auf, erzeugt **pro Flair** eigene Themenmodelle (LDA), vergibt **aussagekräftige Labels** (inkl. Duplikat-Vermeidung) und baut daraus einen kompakten, modernen **HTML-Report**.  
Der Report stellt **DE/EN-Ergebnisse je Flair nebeneinander**, ergänzt um eine **Statistik-Karte** (Dokumentanzahl, Features, K, Kohärenz) sowie **Tabellen zu Flairs & Top-Usern**. Ganz unten findest du den **vollständigen Konsolen-Output**.

### Built With

- Python 3.10+
- pandas, numpy
- scikit-learn
- PRAW (Reddit API)
- python-dotenv
- tqdm

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- Optional: **Reddit API** (PRAW), um Live-Daten zu ziehen. Ohne API wird eine vorhandene CSV genutzt.

### Installation

```bash
# 1) Repo klonen
git clone https://github.com/Aalkfk/dlbdseda02_d.git
cd dlbdseda02_d

# 2) (optional) virtuelles Environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (Powershell):
# .venv\Scripts\Activate.ps1

# 3) Dependencies installieren
pip install -r requirements.txt
# Falls keine requirements.txt vorhanden:
# pip install pandas numpy scikit-learn praw python-dotenv tqdm
```

### Environment Variables

Für den Live-Fetch via Reddit-API lege eine `.env` im Projekt-Root an:

```env
REDDIT_CLIENT_ID=xxxxxxxx
REDDIT_CLIENT_SECRET=xxxxxxxx
REDDIT_USER_AGENT=stuttgart-topics/1.0
```

> Hinweis: Falls `data/raw_r_stuttgart.csv` existiert, nutzt das Programm standardmäßig **diese Datei** (kein API-Call nötig).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

Starte die komplette Pipeline inkl. HTML-Report:

```bash
python -m src.main
```

- Wenn keine Raw-CSV existiert, lädt das Skript Daten via Reddit-API und speichert sie unter `data/raw_r_stuttgart.csv`.
- Danach folgen Preprocessing, Topics je Flair und Ausgabe eines **HTML-Reports** nach `data/report.html`.  
- Der **gesamte Konsolen-Log** ist im Report unten eingebettet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration

Die wichtigsten Einstellungen liegen in `src/main.py`.

- **FLAIR_RUNS**: Liste der Modell-Runs pro Flair, inkl. Sprache (`lang`), Mindest-Token (`min_tokens`), `title_boost`, Auto-Stopwörter (`df_ratio_auto_sw`), Zusatz-Stopwörter (`extra_stopwords`), Feature-Schwelle (`feat_min_hard`) und **`candidate_k`** für die Topic-Anzahl.
- **Kandidatenauswahl (K)**: Per `candidate_k` (z. B. `[4,5,6]`) wird die Suche nach der besten Topic-Anzahl eingegrenzt.
- **Vectorizer-Fenster**: Pro Block können `(min_df, max_df)`-Paare via `pairs=[(...)]` vorgegeben werden (z. B. engere Fenster für News).
- **Fallbacks**: Bei sehr kleinen Korpora nutzt das Skript **TF-IDF-Keywords** statt LDA.
- **Label-Dedup**: Verhindert doppelte Topic-Labels; alternative Kandidaten basieren auf den Top-Termen.

**Beispiel (vereinfacht):**
```python
FLAIR_RUNS = [
  { "name": "News - DE", "flairs": ["News"], "lang": "de",
    "include_comments": False,
    "min_tokens": 5,
    "title_boost": 5,
    "feat_min_hard": 40,
    "candidate_k": [3],
    "pairs": [(3,0.60),(3,1.00),(2,0.60)]
  },
  # ...
]
```

**Hinweis zu Kohärenz**: Negative Kohärenzwerte sind bei dünnen, kurzen Reddit-Texten nicht unüblich. Entscheidend ist die **Vergleichbarkeit** innerhalb desselben Blocks (z. B. Auswahl von K und Seeds).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Outputs

Standard-Verzeichnis: `data/` (siehe `DATA_DIR` in `src/utils.py`).

- `raw_r_stuttgart.csv` – Rohdaten (Posts + ggf. Kommentare)  
- `top_flairs.csv` – Häufigste Flairs (Top 20)  
- `top_users.csv` – Aktivste Nutzer:innen (Top 20)  
- `samples_<block>_per_topic.csv` – Beispiel-Posts je Topic-ID & Label  
- `report.html` – Styling-starker HTML-Report inkl. Statistiken, DE/EN-Vergleiche und Konsolen-Log

**Report-Inhalte (Auszug):**
- **Top Flairs** und **Top User** (Tabellen)
- **Statistik-Karte** (Dokumente, EN-Filter-Quote, K, Kohärenz, Features)
- **Topics je Flair**: DE und EN **nebeneinander** mit Labels & Top-Termen
- **Roh-Log** (ganzer Konsolen-Output am Ende)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [ ] Interaktive Report-Ansicht (Filter/Collapse)
- [ ] Persistenz für Topic-Historie (Zeitreihen)
- [ ] Optionale Stemming/POS-Filter als Switch
- [ ] Export nach Markdown/PDF
- [ ] Unit-Tests für Preprocessing & Labeling

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

Pull Requests und Issues sind willkommen — kurze Beschreibung, reproduzierbare Beispiele und, wenn möglich, kleine Test-Snippets helfen sehr.

**Standard-Flow:**
1. Forken  
2. Feature-Branch erstellen (`git checkout -b feature/awesome`)  
3. Committen (`git commit -m "Add awesome"`)  
4. Pushen (`git push origin feature/awesome`)  
5. PR öffnen

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

Distributed under the **Unlicense**. Siehe `LICENSE.txt` für Details.  
*(Gerne anpassen, falls du eine andere Lizenz bevorzugst.)*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contact

Maintainer: **Joshua Wolf**  
Project Link: https://github.com/Aalkfk/dlbdseda02_d/tree/main

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Acknowledgments

- scikit-learn, pandas, numpy  
- PRAW (Python Reddit API Wrapper)  
- tqdm  
- python-dotenv  
- Struktur angelehnt an bekannte README-Templates

<p align="right">(<a href="#readme-top">back to top</a>)</p>
