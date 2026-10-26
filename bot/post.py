"""Send a prediction to Discord and/or Telegram. Always prints it too, so it shows in the run log."""
import requests

from .config import DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def post(text, dry_run=False):
    print(text, "\n")
    if dry_run:
        return []
    sent = []
    if DISCORD_WEBHOOK_URL:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=20).raise_for_status()
        sent.append("discord")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20).raise_for_status()
        sent.append("telegram")
    return sent
