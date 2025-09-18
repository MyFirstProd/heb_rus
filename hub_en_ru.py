from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer

# === Перевод иврит → английский ===
nllb_path = "facebook/nllb-200-1.3B"
tokenizer_nllb = AutoTokenizer.from_pretrained(nllb_path, use_fast=False)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_path)

hebrew_text = "היום שנת 2025"
tokenizer_nllb.src_lang = "heb"
prompt = f">>eng_Latn<< {hebrew_text}"
inputs = tokenizer_nllb(prompt, return_tensors="pt")
# bos_eng_id = tokenizer_nllb.convert_tokens_to_ids(">>eng_Latn<<")

generated_eng = model_nllb.generate(**inputs, forced_bos_token_id=256047)
english_text = tokenizer_nllb.batch_decode(generated_eng, skip_special_tokens=True)[0]

# === Перевод английский → русский m2m100_418M===
model_path = "Helsinki-NLP/opus-mt-en-ru"
tokenizer_ru = MarianTokenizer.from_pretrained(model_path)
model_ru = MarianMTModel.from_pretrained(model_path)

tokenized = tokenizer_ru(english_text, return_tensors="pt", padding=True)
generated = model_ru.generate(**tokenized)
russian_text = tokenizer_ru.batch_decode(generated, skip_special_tokens=True)[0]

print("🗣 Hebrew:", hebrew_text)
print("🔤 English:", english_text)
print("🇷🇺 Russian:", russian_text)
