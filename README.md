# bot31 – Israel Traffic Fine Appeal Bot

A Dockerised Python Telegram bot that helps users submit appeals for Israeli traffic fines.  
Supports **Russian 🇷🇺 / Hebrew 🇮🇱 / English 🇬🇧** UI; appeal letters are always generated **in Hebrew**.

---

## Features

| Capability | Details |
|---|---|
| **Multi-user** | Each user has an independent session |
| **File upload** | JPG, PNG, PDF – max 15 MB |
| **OCR** | Tesseract (heb+eng) with OpenCV preprocessing; PSM 6/4/11 with best-result heuristics |
| **AI extraction** | OpenAI GPT-4o extracts structured fields with per-field confidence scores |
| **Data correction** | After extraction, inline Edit button lets users fix any field (fine number, violation code, date, amount, plate, location, deadline) before generating the appeal |
| **Appeal generation** | Formal Hebrew letter using confirmed facts only |
| **Multilingual UI** | JSON locale dictionaries (ru/he/en), switchable via `/language` |
| **Encrypted storage** | Personal data encrypted at rest with Fernet (PBKDF2-derived key); 7-day TTL |
| **Audit logs** | Technical logs (no PII) + encrypted audit log entries in SQLite |
| **Data deletion** | `/delete` command removes all user data immediately |

---

## Quick Start

```bash
# 1. Copy and fill in your secrets
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, optionally ENCRYPTION_KEY

# 2. Build and run
docker compose up --build -d

# 3. View logs
docker compose logs -f
```

---

## Configuration (`.env`)

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | Bot token from @BotFather |
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `OPENAI_MODEL` | | Model to use (default: `gpt-4o`) |
| `ENCRYPTION_KEY` | | Explicit encryption secret; falls back to bot token derivation |
| `LOG_LEVEL` | | Python log level (default: `INFO`) |

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message + language picker |
| `/language` | Change UI language |
| `/appeal` | Generate appeal for the most recent case |
| `/delete` | Permanently delete all personal data |
| `/help` | Show command list |

---

## Correcting Extracted Data

After the bot processes your fine notice it displays the extracted fields with confidence indicators and three inline buttons:

- **✅ All correct** – proceed directly to generating the appeal letter.
- **✏️ Edit** – open the field-selection menu where you can pick any field (fine number, violation code, date, amount, licence plate, location, or payment deadline) and type a corrected value.
- **↩️ Back** – return to the "send your fine notice" prompt.

Inside the edit menu:
- Tap a field to select it, then type the corrected value.
- **↩️ Back** from the value-prompt returns you to the field list.
- **↩️ Back** from the field list returns you to the summary with all three buttons.

Corrected fields are marked with ✏️ in the summary and are used when generating the appeal letter.

---

## Storage

- **SQLite** database at `/data/bot31.db` (Docker volume `bot_data`)
- Encrypted case JSON blobs stored in the DB
- Encrypted uploaded files stored under `/data/cases/`
- Both are encrypted with AES-128-CBC (Fernet) before being written

---

## Development (without Docker)

```bash
# Install system deps (Debian/Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng poppler-utils libgl1

# Python deps
pip install -r requirements.txt

# Run
export DATA_DIR=/tmp/bot31_data
python -m bot.main
```