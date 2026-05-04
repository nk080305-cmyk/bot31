"""Central configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# Optional explicit encryption key; falls back to bot token derivation when empty
ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Storage paths
DATA_DIR: str = os.getenv("DATA_DIR", "/data")
DB_PATH: str = os.path.join(DATA_DIR, "bot31.db")
CASES_DIR: str = os.path.join(DATA_DIR, "cases")

# Limits
MAX_FILE_SIZE: int = 15 * 1024 * 1024  # 15 MB
DATA_TTL_DAYS: int = 7
