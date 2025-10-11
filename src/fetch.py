from __future__ import annotations
import time
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import praw
from .utils import DATA_DIR, get_env

def reddit_client():
    # Nutze read-only App Credentials (Reddit: "script"-App)
    return praw.Reddit(
        client_id=get_env("REDDIT_CLIENT_ID"),
        client_secret=get_env("REDDIT_CLIENT_SECRET"),
        user_agent=get_env("REDDIT_USER_AGENT", "stuttgart-topics/1.0"),
        check_for_async=False
    )

def fetch_subreddit_posts(subreddit_name: str="Stuttgart",
                          where: str="top", time_filter: str="year",
                          limit: int=500, with_comments: bool=False,
                          max_comments_per_post: int=100) -> pd.DataFrame:
    """Ziehe Posts (und optional Top-Kommentare) aus einem Subreddit."""
    reddit = reddit_client()
    sub = reddit.subreddit(subreddit_name)

    if where == "new":
        generator = sub.new(limit=limit)
    elif where == "hot":
        generator = sub.hot(limit=limit)
    else:
        generator = sub.top(time_filter=time_filter, limit=limit)

    records: List[Dict] = []
    for post in tqdm(generator, desc=f"Fetching r/{subreddit_name}"):
        rec = {
            "id": post.id,
            "author": str(post.author) if post.author else None,
            "created_utc": post.created_utc,
            "title": post.title or "",
            "selftext": post.selftext or "",
            "score": int(post.score or 0),
            "num_comments": int(post.num_comments or 0),
            "url": post.url or "",
            "permalink": f"https://reddit.com{post.permalink}"
        }
        text = f"{rec['title']} {rec['selftext']}".strip()
        rec["text"] = text

        # Optional Kommentare einbeziehen (mehr „Diskussionstext“)
        comments_blob = []
        if with_comments and post.num_comments:
            post.comments.replace_more(limit=0)
            for i, c in enumerate(post.comments[:max_comments_per_post]):
                au = str(c.author) if c.author else None
                comments_blob.append(c.body or "")
                rec["num_comments"] = max(rec["num_comments"], i+1)
        rec["comments_text"] = "\n".join(comments_blob)

        records.append(rec)
        time.sleep(0.2)  # netter zur API

    df = pd.DataFrame.from_records(records)
    df.to_csv(DATA_DIR / "raw_r_stuttgart.csv", index=False)
    return df

