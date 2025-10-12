from __future__ import annotations

"""
fetch.py — Reddit-Datenbeschaffung für r/Stuttgart

Dieses Modul kapselt:
- den Aufbau eines PRAW-Clients (reddit_client)
- das Abrufen von Posts (inkl. optionaler Top-Level-Kommentare) aus einem Subreddit
  und das Speichern als CSV in DATA_DIR/raw_r_stuttgart.csv

Umgebungsvariablen (via utils.get_env):
- REDDIT_CLIENT_ID
- REDDIT_CLIENT_SECRET
- REDDIT_USER_AGENT (optional; Default: "stuttgart-topics/1.0")
"""

import time
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import praw

from .utils import DATA_DIR, get_env


def reddit_client() -> praw.Reddit:
    """Erzeuge einen synchronen PRAW-Client.

    Returns:
        praw.Reddit: Konfigurierter Reddit-Client (check_for_async=False).

    Hinweise:
        Erwartet die Credentials in den Umgebungsvariablen (s. Modul-Docstring).
    """
    return praw.Reddit(
        client_id=get_env("REDDIT_CLIENT_ID"),
        client_secret=get_env("REDDIT_CLIENT_SECRET"),
        user_agent=get_env("REDDIT_USER_AGENT", "stuttgart-topics/1.0"),
        check_for_async=False,
    )


def fetch_subreddit_posts(
    subreddit_name: str = "Stuttgart",
    where: str = "top",
    time_filter: str = "year",
    limit: int = 600,
    with_comments: bool = True,
    max_comments_per_post: int = 50,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    """Lade Beiträge (und optional Top-Level-Kommentare) aus einem Subreddit.

    Args:
        subreddit_name: Name des Subreddits (ohne "r/").
        where: Feed-Quelle: "top", "new" oder "hot".
        time_filter: Nur bei "top" relevant (z. B. "day", "week", "month", "year", "all").
        limit: Maximale Anzahl geladener Posts.
        with_comments: Wenn True, lade je Post bis zu `max_comments_per_post` Top-Level-Kommentare.
        max_comments_per_post: Hartes Limit für geladene Kommentare pro Post.
        sleep_sec: Kurze Pause nach jedem Post zum Throttling (Default 0.2 s).

    Returns:
        pd.DataFrame: Tabelle mit Posts (und Kommentarblob), die zusätzlich als
        CSV unter DATA_DIR/raw_r_stuttgart.csv gespeichert wird.

    Notizen:
        - Kommentare werden als einfacher Text-Blob (Zeilen getrennt) in `comments_text` gespeichert.
        - Felder `flair_text` und `flair_css` sind „best-effort“ (können leer sein).
        - Der kombinierte freie Text des Posts liegt in `text` (title + selftext).
    """
    reddit = reddit_client()
    sub = reddit.subreddit(subreddit_name)

    # Quelle/Feed wählen
    if where == "new":
        generator = sub.new(limit=limit)
    elif where == "hot":
        generator = sub.hot(limit=limit)
    else:
        # Default: "top" mit Zeitfilter
        generator = sub.top(time_filter=time_filter, limit=limit)

    records: List[Dict[str, Any]] = []

    # tqdm: Fortschrittsbalken im Terminal
    for post in tqdm(generator, desc=f"Fetching r/{subreddit_name}"):
        rec: Dict[str, Any] = {
            "id": post.id,
            "author": str(post.author) if post.author else None,
            "created_utc": post.created_utc,
            "title": post.title or "",
            "selftext": post.selftext or "",
            "score": int(post.score or 0),
            "num_comments": int(post.num_comments or 0),
            "url": post.url or "",
            "permalink": f"https://reddit.com{post.permalink}",
            # Flair-Felder sind nicht immer gesetzt; defensiv abfragen.
            "flair_text": (post.link_flair_text or "").strip()
            if hasattr(post, "link_flair_text")
            else "",
            "flair_css": (post.link_flair_css_class or "").strip()
            if hasattr(post, "link_flair_css_class")
            else "",
        }

        # Kombinierter Content für spätere Vektorisierung
        rec["text"] = f"{rec['title']} {rec['selftext']}".strip()

        # Optional: Top-Level-Kommentare einsammeln
        comments_blob: List[str] = []
        if with_comments and post.num_comments:
            # Keine Placeholder („MoreComments“); nur Top-Level
            post.comments.replace_more(limit=0)
            for i, c in enumerate(post.comments[: max_comments_per_post]):
                comments_blob.append(c.body or "")
                # Korrigiere num_comments auf „tatsächlich gesammelt“
                rec["num_comments"] = max(rec["num_comments"], i + 1)
        rec["comments_text"] = "\n".join(comments_blob)

        records.append(rec)

        # Throttling: vorsichtig mit der Reddit-API umgehen
        if sleep_sec and sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame.from_records(records)

    # Persistiere Rohdaten (für reproduzierbare Läufe & Report)
    out_path = DATA_DIR / "raw_r_stuttgart.csv"
    df.to_csv(out_path, index=False)

    return df