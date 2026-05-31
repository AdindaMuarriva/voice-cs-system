# 🎙️ Code-Switching Speech-to-Speech System
**UAS Praktikum NLP 2025/2026 Genap | Program Studi Informatika, Universitas Syiah Kuala**

Sistem multilingual Speech-to-Speech end-to-end yang mendukung ujaran code-switching Bahasa **Indonesia – Inggris – Arab**.

---

## 📐 Arsitektur Pipeline

```
Audio Input (.wav)
      │
      ▼
[STT] Whisper / whisper.cpp
      │  Transkripsi teks (ID/EN/AR)
      ▼
[Processing] Normalisasi + Language Tagging
      │  Deteksi bahasa, normalisasi teks
      ▼
[LLM] Google Gemini API
      │  Generate respons (mode: preserve/normalize/translate)
      ▼
[TTS] Coqui TTS (model Indonesia)
      │  Sintesis suara
      ▼
Audio Output (.wav)
```

---

## 📁 Struktur Proyek

```
voice-cs-system/
├── app/
│   ├── main.py              ← FastAPI backend
│   ├── stt.py               ← Speech-to-Text (Whisper)
│   ├── llm.py               ← LLM (Gemini API)
│   ├── tts.py               ← Text-to-Speech (Coqui)
│   ├── utils.py             ← Normalisasi & language tagging
│   └── coqui_tts/           ← Model TTS (diisi manual)
├── data/
│   ├── corpus/
│   │   ├── audio/           ← File rekaman .wav
│   │   └── transcripts/     ← Hasil transkripsi .json
│   └── manifests/
├── models/
│   └── whisper.cpp/         ← Diisi setelah clone
├── log/                     ← Log hasil pipeline
├── gradio_app/
│   └── app.py               ← UI demo Gradio
├── analisis_pipeline.py     ← Script batch analisis corpus
├── .env.example             ← Template konfigurasi
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Instalasi

### 1. Clone & Virtual Environment
```bash
git clone <repo-url>
cd voice-cs-system
python3 -m venv env
source env/bin/activate        # Linux/macOS
# env\Scripts\activate         # Windows
pip install -r requirements.txt
pip install -U google-genai
pip install transformers==5.0.0   # Fix Coqui TTS compatibility
```

### 2. Konfigurasi `.env`
```bash
cp .env.example .env
# Edit .env dan isi GEMINI_API_KEY
```

### 3. Install & Build whisper.cpp
```bash
git clone https://github.com/ggml-org/whisper.cpp.git models/whisper.cpp
cd models/whisper.cpp
cmake -B build
cmake --build build --config Release
./models/download-ggml-model.sh large-v3-turbo   # atau model lebih kecil
cd ../..
```

### 4. Download model Coqui TTS Indonesia
Simpan model ke `app/coqui_tts/`:
- `config.json`
- `checkpoint_100000.pth`
- `speakers.pth` (jika multi-speaker)

Referensi: [wikipedia/indonesian-tts](https://github.com/wikipedia/indonesian-tts)

---

## 🚀 Menjalankan Sistem

### Backend FastAPI
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Dokumentasi API: http://localhost:8000/docs

### UI Gradio (Opsional)
```bash
python gradio_app/app.py
```
Akses di: http://localhost:7860

---

## 📊 Analisis Korpus

Letakkan file audio di `data/corpus/audio/`, lalu jalankan:
```bash
# Analisis semua file
python analisis_pipeline.py

# Mode normalize, batasi 5 file
python analisis_pipeline.py --mode normalize --limit 5

# Dengan bahasa hint
python analisis_pipeline.py --lang id --sleep 3.0
```
Laporan tersimpan di `log/`.

---

## 📂 Format Penamaan File Audio Korpus
```
{id}_{utteranceid}.wav
Contoh: 2030_audio01.wav
```
`id` = 2 digit awal + 2 digit akhir NPM.

---

## 🔑 Endpoint API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/voice-chat` | Pipeline penuh: audio → audio |
| POST | `/text-chat` | Teks → respons LLM |
| POST | `/transcribe` | Audio → transkripsi saja |
| POST | `/reset-conversation` | Reset history |
| GET  | `/health` | Status sistem |

---

## 📏 Metrik Evaluasi

| Komponen | Metrik |
|----------|--------|
| STT | WER (Word Error Rate), CER (Character Error Rate) |
| LLM | Correctness (penilaian manual) |
| TTS | Naturalness (penilaian subjektif) |
| End-to-end | Latency (s), Intelligibility |
