"""
utils.py
Utilitas untuk normalisasi teks, deteksi bahasa, dan language tagging
pada teks code-switching ID-EN-AR.
"""

import re
import os
from langdetect import detect, detect_langs, LangDetectException


# ─────────────────────────────────────────────
# Pola kata umum per bahasa (kamus mini)
# Digunakan sebagai override ketika langdetect kurang akurat
# ─────────────────────────────────────────────
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
ENGLISH_WORDS = {
    "i", "you", "he", "she", "we", "they", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "for", "with", "by", "from", "this", "that", "it", "be", "have",
    "do", "not", "what", "how", "when", "where", "why", "who", "will",
    "can", "could", "would", "should", "may", "might", "just", "very",
    "so", "if", "about", "like", "think", "know", "get", "go", "come",
    "please", "thank", "yes", "no", "ok", "okay",
}
INDONESIAN_WORDS = {
    "saya", "aku", "kamu", "dia", "kami", "kita", "mereka", "ini", "itu",
    "yang", "dan", "atau", "tetapi", "tapi", "di", "ke", "dari", "untuk",
    "dengan", "adalah", "ada", "tidak", "bukan", "sudah", "akan", "bisa",
    "harus", "mau", "ingin", "perlu", "juga", "saja", "aja", "lagi",
    "sudah", "belum", "pernah", "sangat", "sangatlah", "memang", "jadi",
    "kalau", "karena", "karena", "namun", "namun", "bahwa", "apa", "siapa",
    "gimana", "bagaimana", "kenapa", "kapan", "dimana", "mana",
}


def detect_token_language(token: str) -> str:
    """
    Deteksi bahasa satu token/kata.
    Return: 'ar' | 'en' | 'id' | 'unknown'
    """
    token_lower = token.lower().strip(".,!?;:'\"()")
    
    # Cek karakter Arab
    if ARABIC_PATTERN.search(token):
        return "ar"
    
    # Cek kamus statis
    if token_lower in ENGLISH_WORDS:
        return "en"
    if token_lower in INDONESIAN_WORDS:
        return "id"
    
    # Fallback ke langdetect untuk kata yang lebih panjang
    if len(token_lower) > 3:
        try:
            lang = detect(token_lower)
            if lang in ("ar", "en", "id"):
                return lang
        except LangDetectException:
            pass
    
    return "unknown"


def tag_code_switching(text: str) -> list[dict]:
    """
    Tokenisasi dan tagging bahasa per token.
    
    Returns:
        List of dict: [{"token": str, "lang": str}, ...]
    
    Contoh output:
        [
          {"token": "Saya", "lang": "id"},
          {"token": "sudah", "lang": "id"},
          {"token": "submit", "lang": "en"},
          {"token": "الواجب", "lang": "ar"},
        ]
    """
    tokens = text.split()
    tagged = []
    for token in tokens:
        lang = detect_token_language(token)
        tagged.append({"token": token, "lang": lang})
    return tagged


def detect_dominant_language(text: str) -> str:
    """
    Deteksi bahasa dominan dalam sebuah teks.
    Return: 'id' | 'en' | 'ar' | 'mixed'
    """
    if not text.strip():
        return "unknown"
    
    # Cek jika ada konten Arab yang signifikan
    arabic_chars = len(ARABIC_PATTERN.findall(text))
    if arabic_chars > 2:
        try:
            langs = detect_langs(text)
            # Kalau ada Arab yang terdeteksi, tandai
            lang_codes = [l.lang for l in langs]
            if "ar" in lang_codes:
                return "ar" if len(lang_codes) == 1 else "mixed"
        except LangDetectException:
            pass
    
    try:
        langs = detect_langs(text)
        top_lang = langs[0]
        # Jika probabilitas dominan > 0.85, anggap monolingual
        if top_lang.prob > 0.85 and top_lang.lang in ("id", "en", "ar"):
            return top_lang.lang
        return "mixed"
    except LangDetectException:
        return "unknown"


def normalize_text(text: str) -> str:
    """
    Normalisasi teks transkrip:
    - Hapus karakter berulang (misal: "sangatttt" → "sangat")
    - Perbaiki spasi berlebih
    - Kapitalisasi awal kalimat
    - Hapus tanda baca ganda
    """
    # Hapus karakter berulang lebih dari 2x (kecuali huruf Arab)
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)
    
    # Normalkan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus tanda baca ganda
    text = re.sub(r'[.!?]{2,}', '.', text)
    
    # Kapitalisasi awal kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.capitalize() for s in sentences if s]
    text = ' '.join(sentences)
    
    return text


def split_by_language(text: str) -> list[dict]:
    """
    Pecah teks menjadi segmen berdasarkan bahasa dominan.
    Berguna untuk TTS multi-bahasa.
    
    Returns:
        List of dict: [{"text": str, "lang": str}, ...]
    """
    tagged = tag_code_switching(text)
    
    if not tagged:
        return []
    
    segments = []
    current_lang = tagged[0]["lang"]
    current_tokens = [tagged[0]["token"]]
    
    for item in tagged[1:]:
        if item["lang"] == current_lang or item["lang"] == "unknown":
            current_tokens.append(item["token"])
        else:
            segments.append({
                "text": " ".join(current_tokens),
                "lang": current_lang
            })
            current_lang = item["lang"]
            current_tokens = [item["token"]]
    
    # Tambahkan segmen terakhir
    if current_tokens:
        segments.append({
            "text": " ".join(current_tokens),
            "lang": current_lang
        })
    
    return segments


def build_tagging_summary(tagged_tokens: list[dict]) -> str:
    """
    Buat ringkasan tagging bahasa untuk ditampilkan di log/laporan.
    """
    lang_counts = {}
    for item in tagged_tokens:
        lang = item["lang"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    total = len(tagged_tokens)
    summary_parts = []
    for lang, count in sorted(lang_counts.items()):
        pct = (count / total) * 100 if total > 0 else 0
        label = {"id": "Indonesia", "en": "Inggris", "ar": "Arab", "unknown": "Tidak diketahui"}.get(lang, lang)
        summary_parts.append(f"{label}: {count} token ({pct:.1f}%)")
    
    return " | ".join(summary_parts)


if __name__ == "__main__":
    # Test utilitas
    sample = "Saya sudah submit the assignment tapi masih ada masalah في الكود"
    print("Input:", sample)
    print("Normalized:", normalize_text(sample))
    print("Dominant lang:", detect_dominant_language(sample))
    print("\nTagging per token:")
    tagged = tag_code_switching(sample)
    for item in tagged:
        print(f"  [{item['lang']:7}] {item['token']}")
    print("\nSummary:", build_tagging_summary(tagged))
    print("\nSplit by language:")
    for seg in split_by_language(sample):
        print(f"  [{seg['lang']}] {seg['text']}")
