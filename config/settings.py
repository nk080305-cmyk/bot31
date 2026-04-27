"""
Configuration loaded from environment variables / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Redis (FSM storage) ───────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

# ── 2Captcha (optional CAPTCHA solving) ──────────────────────────────────────
TWOCAPTCHA_API_KEY: str = os.getenv("TWOCAPTCHA_API_KEY", "")

# ── Appeal submission target ──────────────────────────────────────────────────
# Israeli police traffic violations appeal portal
APPEAL_URL: str = os.getenv(
    "APPEAL_URL",
    "https://www.gov.il/he/Departments/Guides/police_traffic_violations_appeal",
)

# ── Misc ──────────────────────────────────────────────────────────────────────
TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/bot31")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
