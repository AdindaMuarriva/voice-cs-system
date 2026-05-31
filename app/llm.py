"""
llm.py
Integrasi Google Gemini API untuk menghasilkan respons kontekstual
dari transkrip code-switching ID-EN-AR.

Mode yang didukung:
- preserve  : Pertahankan pola code-switching dalam respons
- normalize : Normalisasi ke satu bahasa (default: Bahasa Indonesia)
- translate : Terjemahkan ke bahasa target (opsional)
"""

import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")
RPM_SLEEP    = float(os.getenv("RPM_SLEEP", "2"))

# ── Multi-key rotation ───────────────────────────────────────────
# Baca semua key: GEMINI_API_KEY, GEMINI_API_KEY_1, _2, _3, dst.
def _load_api_keys() -> list[str]:
    keys = []
    k = os.getenv("GEMINI_API_KEY", "")
    if k:
        keys.append(k)
    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if k:
            keys.append(k)
    # Deduplikasi sambil jaga urutan
    seen, unique = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique

_API_KEYS: list[str] = _load_api_keys()
_key_index   = 0
_key_exhausted: set = set()   # indeks key yang quota-nya habis hari ini


def _get_active_key() -> str:
    if not _API_KEYS:
        raise ValueError(
            "Tidak ada GEMINI_API_KEY di .env!\n"
            "Tambahkan: GEMINI_API_KEY=AIza... di file .env"
        )
    return _API_KEYS[_key_index]


def _rotate_key(reason: str = "") -> bool:
    """
    Rotasi ke key berikutnya yang belum exhausted.
    Return True jika berhasil rotasi, False jika semua key sudah habis.
    """
    global _key_index
    _key_exhausted.add(_key_index)
    for i in range(len(_API_KEYS)):
        next_idx = (_key_index + 1 + i) % len(_API_KEYS)
        if next_idx not in _key_exhausted:
            _key_index = next_idx
            label = reason if reason else "rate limit"
            print(f"[LLM] 🔄 Rotasi ke API key #{_key_index + 1}/{len(_API_KEYS)} ({label})")
            return True
    return False   # semua key exhausted


def print_key_status():
    """Tampilkan status semua key (berguna untuk debug)."""
    print(f"[LLM] Total API key: {len(_API_KEYS)}")
    for i, key in enumerate(_API_KEYS):
        masked = key[:8] + "..." + key[-4:]
        status = "✅ aktif" if i == _key_index else ("❌ exhausted" if i in _key_exhausted else "⏳ standby")
        print(f"  Key #{i+1}: {masked} — {status}")


# Konversasi (history)
_conversation_history: list[dict] = []


def _get_client():
    """Inisialisasi client Google Gen AI dengan key aktif."""
    try:
        from google import genai
        client = genai.Client(api_key=_get_active_key())
        return client
    except ImportError:
        raise ImportError(
            "Library google-genai belum terinstall.\n"
            "Jalankan: pip install -U google-genai"
        )


def _build_system_prompt(mode: str = "preserve") -> str:
    base = (
        "Kamu adalah asisten percakapan yang cerdas dan ramah. "
        "Pengguna akan berbicara dengan campuran Bahasa Indonesia, Inggris, dan Arab (code-switching). "
        "Berikan respons yang informatif, ringkas, dan relevan. "
        "Jangan terlalu panjang - cukup 2-4 kalimat untuk percakapan normal. "
    )
    if mode == "preserve":
        return base + (
            "PENTING: Pertahankan pola code-switching dalam responsmu. "
            "Gunakan campuran Bahasa Indonesia, Inggris, dan Arab secara natural "
            "seperti yang dilakukan penutur asli. "
            "Contoh: 'Oke, jadi basically kamu perlu submit dulu ya, insya Allah bisa.' "
        )
    elif mode == "normalize":
        return base + (
            "PENTING: Berikan SEMUA respons hanya dalam Bahasa Indonesia yang baku dan jelas. "
            "Jangan campur bahasa lain, termasuk kata serapan dari Inggris atau Arab "
            "kecuali sudah menjadi bagian resmi Bahasa Indonesia. "
        )
    elif mode == "translate":
        return base + (
            "PENTING: Terjemahkan seluruh konten ke Bahasa Indonesia yang baku. "
            "Respons hanya dalam Bahasa Indonesia. "
        )
    return base


def generate_response(
    transcript: str,
    mode: str = "preserve",
    tagged_text: str | None = None,
    max_tokens: int = 512,
    keep_history: bool = False,
) -> dict:
    """
    Kirim transkrip ke Gemini dan dapatkan respons.
    Otomatis rotasi API key jika kena 429.
    """
    if not _API_KEYS:
        raise ValueError("Tidak ada GEMINI_API_KEY di .env!")

    system_prompt = _build_system_prompt(mode)

    user_message = transcript
    if tagged_text:
        user_message = (
            f"[Input pengguna dengan tagging bahasa: {tagged_text}]\n"
            f"[Transkrip bersih: {transcript}]"
        )

    messages = []
    if keep_history and _conversation_history:
        messages.extend(_conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})

    start = time.time()

    from google import genai
    from google.genai import types

    # Coba semua key yang tersedia
    MAX_KEY_ATTEMPTS = len(_API_KEYS)
    MAX_RETRY_PER_KEY = 2

    for key_attempt in range(MAX_KEY_ATTEMPTS):
        client = _get_client()

        contents = [
            types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["content"])]
            )
            for msg in messages
        ]

        for retry in range(MAX_RETRY_PER_KEY):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                    )
                )
                response_text = response.text.strip()
                finish_reason = (
                    str(response.candidates[0].finish_reason)
                    if response.candidates else "unknown"
                )

                # Berhasil — simpan history jika perlu
                if keep_history:
                    _conversation_history.append({"role": "user",  "content": user_message})
                    _conversation_history.append({"role": "model", "content": response_text})

                time.sleep(RPM_SLEEP)

                return {
                    "response_text":      response_text,
                    "mode":               mode,
                    "model":              GEMINI_MODEL,
                    "api_key_index":      _key_index + 1,
                    "processing_time_s":  round(time.time() - start, 2),
                    "finish_reason":      finish_reason,
                }

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_quota_zero = "limit: 0" in err_str or "quota" in err_str.lower()

                if is_rate_limit:
                    if is_quota_zero:
                        # Quota harian habis untuk key ini — langsung rotasi
                        print(f"[LLM] ❌ Key #{_key_index + 1} quota harian habis, rotasi key...")
                        break   # keluar dari retry loop, coba key berikutnya

                    # RPM sementara — tunggu dulu, baru retry
                    delay_match = re.search(r'retry[^\d]*(\d+)', err_str, re.IGNORECASE)
                    wait_s = int(delay_match.group(1)) + 2 if delay_match else 35
                    if retry < MAX_RETRY_PER_KEY - 1:
                        print(f"[LLM] ⏳ Rate limit sementara, tunggu {wait_s}s (retry {retry+1}/{MAX_RETRY_PER_KEY})...")
                        time.sleep(wait_s)
                        continue
                    else:
                        # Setelah retry habis, anggap key ini exhausted
                        print(f"[LLM] ❌ Key #{_key_index + 1} tetap rate limit setelah retry, rotasi key...")
                        break
                else:
                    # Error bukan rate limit — langsung raise
                    raise RuntimeError(f"Gemini API error: {e}")

        # Rotasi ke key berikutnya
        rotated = _rotate_key("quota exhausted")
        if not rotated:
            raise RuntimeError(
                f"Semua {len(_API_KEYS)} API key sudah exhausted quota hariannya.\n"
                "Solusi:\n"
                "  1. Buat project baru di https://aistudio.google.com dan tambahkan key baru ke .env\n"
                "  2. Tunggu hingga besok (quota reset tiap 24 jam)\n"
                "  3. Upgrade ke Gemini API berbayar"
            )
        # Reinit client dengan key baru
        client = _get_client()

    raise RuntimeError("Gagal menghubungi Gemini API setelah mencoba semua key.")


def reset_conversation():
    global _conversation_history
    _conversation_history = []


def evaluate_response_quality(response: str, transcript: str) -> dict:
    words = response.split()
    has_content = len(words) >= 5
    if len(words) < 5:
        quality = "sangat pendek"
    elif len(words) > 200:
        quality = "terlalu panjang"
    else:
        quality = "wajar"
    return {
        "length_words":      len(words),
        "has_content":       has_content,
        "estimated_quality": quality
    }


if __name__ == "__main__":
    print_key_status()
    print()
    test = "Saya mau tanya, how do you say terima kasih in Arabic?"
    print(f"Input: {test}\n")
    for mode in ["preserve", "normalize"]:
        print(f"--- Mode: {mode} ---")
        result = generate_response(test, mode=mode)
        print(f"Respons : {result['response_text']}")
        print(f"Key #   : {result['api_key_index']}")
        print(f"Waktu   : {result['processing_time_s']}s\n")