from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка модели и токенизатора без FastTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")

# Установка языка источника
tokenizer.src_lang = "heb"

# Исходный текст
hebrew_text = "שלום, איך אתה מרגיש היום?"
inputs = tokenizer(hebrew_text, return_tensors="pt")

# Перевод (русский — ID 256147, см. tokenizer.json)
outputs = model.generate(
    **inputs,
    forced_bos_token_id=256147,
    max_length=256
)

# Декодирование результата
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Перевод:", translation)
