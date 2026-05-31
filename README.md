# 🎙️ Voice CS System — Code-Switching Speech-to-Speech

> **UAS Praktikum Natural Language Processing 2025/2026 Genap**  
> Program Studi Informatika, Universitas Syiah Kuala

Sistem multilingual **Speech-to-Speech end-to-end** yang menerima ujaran *code-switching* Bahasa **Indonesia–Inggris–Arab**, memprosesnya melalui pipeline STT → LLM → TTS, dan menghasilkan respons suara kembali.

---

## 📌 Deskripsi Singkat

Sistem ini dibangun secara individu sebagai proyek akhir praktikum NLP. Fokus utama adalah penyusunan korpus speech *code-switching* ID–EN–AR yang terkontrol, serta implementasi pipeline percakapan berbasis suara yang mendukung tiga mode operasi:

| Mode | Deskripsi |
|------|-----------|
| `preserve` | Pertahankan pola *code-switching* dalam respons |
| `normalize` | Normalisasi respons ke Bahasa Indonesia baku |
| `translate` | Terjemahkan seluruh konten ke Bahasa Indonesia  |

---

## 🏗️ Arsitektur Pipeline

```
Audio Input (.wav)
      │
      ▼
┌─────────────┐
│  STT        │  OpenAI Whisper (base) — transkripsi multilingual
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Processing │  Normalisasi teks + language tagging per token
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  LLM        │  Google Gemini 2.5 Flash Lite — generate respons
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  TTS        │  Coqui TTS / gTTS — sintesis suara output
└─────┬───────┘
      │
      ▼
Audio Output (.wav)
```

---

## 📁 Struktur Proyek

```
voice-cs-system/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI backend (endpoint utama)
│   ├── stt.py                # Speech-to-Text (Whisper)
│   ├── llm.py                # LLM (Gemini API + rotasi multi-key)
│   ├── tts.py                # Text-to-Speech (Coqui / gTTS fallback)
│   ├── utils.py              # Normalisasi & language tagging
│   └── coqui_tts/            # Model TTS lokal 
│       ├── config.json
│       ├── checkpoint_*.pth
│       └── speakers.pth
│
├── data/
│   ├── corpus/
│   │   ├── audio/            # File rekaman .wav 
│   │   └── transcripts/      # Hasil transkripsi .json
│   └── manifests/
│
├── models/
│   └── whisper.cpp/          # Build whisper.cpp di sini 
│
├── log/                      # Log hasil pipeline 
│   ├── analisis/             # Hasil JSON per file
│   └── tts_output/           # Audio TTS output
│
├── temp/                     # File audio sementara 
│
├── gradio_app/
│   ├── __init__.py
│   └── app.py                # UI demo Gradio
│
├── analisis_pipeline.py      # Script batch analisis seluruh korpus
├── resume_pipeline.py        # Resume pipeline dari checkpoint
├── .env.example              # Template konfigurasi (salin ke .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Instalasi

### Prasyarat
- Python 3.11+
- Git

### 1. Clone Repository

```bash
git clone https://github.com/AdindaMuarriva/voice-cs-system.git
cd voice-cs-system
```

### 2. Virtual Environment

```bash
# Linux / macOS
python3 -m venv env
source env/bin/activate

# Windows
python -m venv env
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -U google-genai
pip install transformers==5.0.0   # Fix kompatibilitas Coqui TTS
```

### 4. Konfigurasi `.env`

```bash
cp .env.example .env
```


### 5. Setup Whisper (Opsional — untuk performa lebih baik)

```bash
git clone https://github.com/ggml-org/whisper.cpp.git models/whisper.cpp
cd models/whisper.cpp
cmake -B build
cmake --build build --config Release
./models/download-ggml-model.sh large-v3-turbo
cd ../..
```

Jika whisper.cpp tidak tersedia, sistem otomatis fallback ke `openai-whisper` Python:

```bash
pip install openai-whisper
```

---

##  Menjalankan Sistem

### Backend FastAPI

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Dokumentasi API otomatis tersedia di: **http://localhost:8000/docs**

### UI Demo Gradio

Buka terminal baru (backend harus tetap berjalan):

```bash
python gradio_app/app.py
```

Akses di: **http://localhost:7860**

---

## Analisis Korpus (Batch Pipeline)

Letakkan file audio di `data/corpus/audio/`, lalu jalankan:

```bash
# Proses semua file (dengan skip otomatis untuk yang sudah selesai)
python analisis_pipeline.py --mode preserve --limit 460 --sleep 5

# Resume dari file yang gagal sebelumnya
python resume_pipeline.py --limit 460 --sleep 5 --retry-llm-only
```

Parameter tersedia:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `--mode` | `preserve` | Mode pipeline: `preserve`, `normalize`, `translate` |
| `--limit` | semua | Batasi jumlah file yang diproses |
| `--sleep` | `2.0` | Jeda antar request LLM (detik) — naikkan jika kena rate limit |
| `--lang` | `auto` | Hint bahasa STT: `auto`, `id`, `en`, `ar` |
| `--retry-llm-only` | — | Hanya retry file yang STT-nya sudah ada tapi LLM gagal |

Laporan otomatis tersimpan di `log/` dalam format **JSON**, **CSV**, dan **TXT**.

---

## Endpoint API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/voice-chat` | Pipeline penuh: audio input → audio output |
| `POST` | `/transcribe` | STT saja: audio → teks + tagging |
| `POST` | `/text-chat` | LLM saja: teks → respons |
| `POST` | `/reset-conversation` | Reset history percakapan |
| `GET`  | `/health` | Status sistem |

Contoh request menggunakan `curl`:

```bash
curl -X POST http://localhost:8000/voice-chat \
  -F "audio=@data/corpus/audio/2222_audio01.wav" \
  -F "mode=preserve" \
  -F "stt_language=auto" \
  --output response.wav
```

---

## Referensi

- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Coqui TTS (fork aktif)](https://github.com/idiap/coqui-ai-TTS)
- [Indonesian TTS Model](https://github.com/wikipedia/indonesian-tts)
- [FastAPI](https://fastapi.tiangolo.com)
- [Gradio](https://www.gradio.app/docs)

---
