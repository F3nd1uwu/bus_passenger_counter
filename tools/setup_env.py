"""Create a local Compose .env with a generated password, preserving existing config."""
from pathlib import Path
import secrets

from config.settings import ROOT


def setup():
    path = ROOT / ".env"
    if path.exists():
        print("Existing .env preserved")
        return
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    content = content.replace("replace-with-a-long-random-password", secrets.token_urlsafe(32))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print("Created .env with a generated database password; no secrets printed")


if __name__ == "__main__":
    setup()
