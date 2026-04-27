# bot31 — Israeli Traffic Violation Appeal Bot 🚔

Telegram-бот, который анализирует письмо о нарушении ПДД в Израиле и автоматически подаёт обжалование на официальном портале.

## Возможности

- 📄 Принимает **фото** или **PDF** письма о штрафе
- 🔍 Извлекает текст через **Tesseract OCR** (иврит + английский)
- 🤖 Анализирует данные и генерирует текст обжалования с помощью **GPT-4o**
- 🌐 Автоматически заполняет форму обжалования через **Playwright**
- ✅ Возвращает скриншот подтверждения пользователю

## Архитектура

```
[Пользователь] → [Telegram Bot] → [OCR] → [GPT-4o] → [Playwright] → [Сайт обжалования]
```

```
bot31/
├── bot/
│   ├── main.py        # Точка входа
│   ├── handlers.py    # FSM-хендлеры
│   └── keyboards.py   # Inline-кнопки
├── ocr/
│   └── extractor.py   # OCR (Tesseract + OpenCV)
├── ai/
│   └── analyzer.py    # GPT-4o анализ и генерация текста
├── scraper/
│   └── submitter.py   # Playwright автоматизация формы
├── config/
│   └── settings.py    # Настройки из .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Быстрый старт

### 1. Скопируйте файл конфигурации

```bash
cp .env.example .env
```

Заполните `.env`:

| Переменная | Описание |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Токен от [@BotFather](https://t.me/BotFather) |
| `OPENAI_API_KEY` | Ключ OpenAI API |
| `TWOCAPTCHA_API_KEY` | *(необязательно)* Ключ [2captcha](https://2captcha.com) для обхода CAPTCHA |
| `APPEAL_URL` | URL формы обжалования (по умолчанию: портал gov.il) |

### 2. Запустите через Docker Compose

```bash
docker compose up --build -d
```

### 3. Откройте бота в Telegram и отправьте `/start`

## Разработка (без Docker)

```bash
# Установить системные зависимости (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-heb poppler-utils

# Установить Python-зависимости
pip install -r requirements.txt

# Установить браузер Playwright
playwright install --with-deps chromium

# Запустить бота
python -m bot.main
```

## FSM-состояния

```
/start
  └─ WAITING_FILE
       └─ (получен файл) → PROCESSING → CONFIRM
            ├─ ✅ Подтвердить → SUBMITTING → done
            ├─ ✏️ Редактировать → CONFIRM
            └─ ❌ Отмена → done
```

## Важные замечания

- Бот помогает сформировать обжалование, но **ответственность за его содержание несёт пользователь**.
- Личные данные хранятся только в памяти сессии и не сохраняются в постоянную БД.
- Для корректной работы автозаполнения форм могут потребоваться обновления селекторов в `scraper/submitter.py` при изменении структуры сайта.