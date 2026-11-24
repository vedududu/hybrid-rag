import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

FEEDBACK_FILE = Path("data/feedback/feedback.jsonl")


def save_feedback(question, answer, vote, level=None, confidence=None, comment=None):
    """Store one thumbs-up/down vote: in Supabase if it is set up, otherwise in a local file."""
    row = {"question": question[:500], "answer": (answer or "")[:10000], "vote": vote,
           "level": level, "confidence": confidence, "comment": (comment or "")[:1000] or None}
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if url and key:
        reply = requests.post(f"{url.rstrip('/')}/rest/v1/feedback", json=row, timeout=10,
                              headers={"apikey": key, "Prefer": "return=minimal"})
        reply.raise_for_status()
        return "supabase"
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    row["created_at"] = datetime.now(timezone.utc).isoformat()
    with FEEDBACK_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")
    return "file"