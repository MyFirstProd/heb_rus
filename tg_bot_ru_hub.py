from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import logging

# === Инициализация моделей ===
# Hebrew → English (NLLB)
nllb_path = "facebook/nllb-200-1.3B"
tokenizer_nllb = AutoTokenizer.from_pretrained(nllb_path, use_fast=False)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_path)

# English → Russian (MarianMT)
marian_en_ru_path = "Helsinki-NLP/opus-mt-en-ru"
tokenizer_en_ru = MarianTokenizer.from_pretrained(marian_en_ru_path)
model_en_ru = MarianMTModel.from_pretrained(marian_en_ru_path)

# Russian → English (MarianMT)
marian_ru_en_path = "Helsinki-NLP/opus-mt-ru-en"
tokenizer_ru_en = MarianTokenizer.from_pretrained(marian_ru_en_path)
model_ru_en = MarianMTModel.from_pretrained(marian_ru_en_path)


# === Определение языка текста ===
def is_hebrew(text):
    return any('֐' <= c <= 'ת' for c in text)


# === Переводчики ===
def hebrew_to_russian(text):
    tokenizer_nllb.src_lang = "heb"
    prompt = f">>eng_Latn<< {text}"
    inputs = tokenizer_nllb(prompt, return_tensors="pt")
    eng_ids = model_nllb.generate(**inputs, forced_bos_token_id=256047)
    eng_text = tokenizer_nllb.batch_decode(eng_ids, skip_special_tokens=True)[0]

    ru_inputs = tokenizer_en_ru(eng_text, return_tensors="pt", padding=True)
    ru_ids = model_en_ru.generate(**ru_inputs)
    return tokenizer_en_ru.batch_decode(ru_ids, skip_special_tokens=True)[0]


def russian_to_hebrew(text):
    en_inputs = tokenizer_ru_en(text, return_tensors="pt", padding=True)
    en_ids = model_ru_en.generate(**en_inputs)
    eng_text = tokenizer_ru_en.batch_decode(en_ids, skip_special_tokens=True)[0]

    tokenizer_nllb.src_lang = "eng_Latn"
    prompt = f">>heb<< {eng_text}"
    inputs = tokenizer_nllb(prompt, return_tensors="pt")
    heb_ids = model_nllb.generate(**inputs, forced_bos_token_id=256067)
    return tokenizer_nllb.batch_decode(heb_ids, skip_special_tokens=True)[0]


# === Telegram бот ===
user_langs = {}  # user_id: "heb" или "rus"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь сообщение, и я переведу его для собеседника.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user_id = msg.from_user.id
    text = msg.text.strip()

    # Определим язык
    if user_id not in user_langs:
        user_langs[user_id] = "heb" if is_hebrew(text) else "rus"

    lang = user_langs[user_id]
    if lang == "heb":
        translated = hebrew_to_russian(text)
    else:
        translated = russian_to_hebrew(text)

    await msg.reply_text(f"Перевод:\n{translated}")

# === Запуск бота ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token("7432088348:AAHxETWkHSn3mtYE_AGy6we1EsHrinwW9sg").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен")
    app.run_polling()
