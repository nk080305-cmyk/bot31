# bot31 – Israel Traffic Fine Appeal Bot

A Dockerised Python Telegram bot that helps users submit appeals for Israeli traffic fines.  
Supports **Russian 🇷🇺 / Hebrew 🇮🇱 / English 🇬🇧** UI; appeal letters are always generated **in Hebrew**.

---

## Features

| Capability | Details |
|---|---|
| **Multi-user** | Each user has an independent session |
| **File upload** | JPG, PNG, PDF – max 15 MB (for best OCR accuracy, send as Telegram document/file) |
| **OCR** | Tesseract (heb+eng) with OpenCV preprocessing + numeric-focused OCR fallback for fine numbers |
| **AI extraction** | OpenAI GPT-4o extracts structured fields with per-field confidence scores |
| **Data correction** | After extraction, ✅/❌ inline buttons let users confirm or fix any field (fine number, violation text, date, amount, licence plate, location, deadline) before generating the appeal |
| **Appeal reason selection** | After confirming data, choose from 5 localised reasons (sign not visible, data error, force majeure, permit, or free text); reason is embedded in the Hebrew letter |
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
| `OCR_MULTI_PREPROCESS` | | Enable improved OCR variants + multi-pass OCR (`1` default, set `0` to disable) |

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

## Correcting Extracted Data & Choosing an Appeal Reason

After the bot processes your fine notice it displays the extracted fields with confidence indicators and two inline buttons:

- **✅ Data is correct** – confirm the data and proceed to appeal-reason selection.
- **❌ Data is incorrect** – open the field-selection menu where you can pick any field (fine number, violation text, date, amount, licence plate, location, or payment deadline) and type a corrected value.

Inside the edit menu:
- Tap a field to select it, then type the corrected value.
- **↩️ Back** from the value-prompt returns you to the field list.
- **↩️ Back** from the field list returns you to the summary with the ✅/❌ buttons.

Corrected fields are marked with ✏️ in the summary (instead of the AI confidence label) and are used when generating the appeal letter.

### Appeal Reason Selection

After pressing **✅ Data is correct**, the bot asks you to choose a reason for the appeal (buttons are shown in your chosen language):

1. **Sign/markings not visible or unclear**
2. **Error in identification/data (plate/time/place)**
3. **Short stop due to necessity / force majeure**
4. **Permit/authorization to park/stop**
5. **Other** – type your own reason in any language

The selected reason is included in the formal Hebrew appeal letter.

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
