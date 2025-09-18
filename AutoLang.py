from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_model_for_language(text_input):
    lang = detect(text_input)
    print(lang)
    if lang == "en":
        model_name = "Helsinki-NLP/opus-mt-en-ru"
    elif lang == "ru":
        model_name = "Helsinki-NLP/opus-mt-ru-en"
    else:
        model_name = "facebook/nllb-200-1.3B"  # Поддержка >200 языков

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


text = "Bonjour, comment ça va ?"
tokenizer, model = get_model_for_language(text)
