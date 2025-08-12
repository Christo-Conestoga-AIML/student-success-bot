# translation.py
from deep_translator import GoogleTranslator

def translate_text(text: str, target_lang: str) -> str:
    if not text or target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"[Translation] Error: {e}")
        return f"(Translation unavailable) {text}"
