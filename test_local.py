"""Preview uses the production pipeline and never writes to PostgreSQL/state."""
from core.cli import cli


if __name__ == "__main__":
    cli(preview=True)
