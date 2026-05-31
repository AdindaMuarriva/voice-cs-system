"""
gradio_app/app.py
Antarmuka Gradio untuk demo sistem Speech-to-Speech Code-Switching.
Koneksi ke FastAPI backend di localhost:8000.
"""

import os
import sys
import json
import time
import tempfile
import requests
from pathlib import Path

import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Timeout per tahap (detik) — bisa diatur via env
TIMEOUT_STT  = int(os.getenv("TIMEOUT_STT",  "180"))   
TIMEOUT_LLM  = int(os.getenv("TIMEOUT_LLM",  "60"))
TIMEOUT_TTS  = int(os.getenv("TIMEOUT_TTS",  "120"))
TIMEOUT_FULL = int(os.getenv("TIMEOUT_FULL", "420"))   


# ── Cek koneksi backend ──────────────────────────────────────────

def check_backend() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            return "🟢 Backend online"
        return f"🔴 Backend merespons status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return "🔴 Backend offline — jalankan: uvicorn app.main:app --reload --port 8000"
    except Exception as e:
        return f"🔴 Error: {e}"


# ── Pipeline Lengkap ─────────────────────────────────────────────

def call_voice_chat(audio_path, mode, stt_lang, segment_tts, progress=gr.Progress()):
    if audio_path is None:
        return None, "❌ Tidak ada audio. Rekam atau upload file .wav dulu.", "{}", "{}"

    # Cek backend dulu
    progress(0.05, desc="Memeriksa koneksi backend...")
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=5)
    except Exception:
        return None, "🔴 Backend tidak bisa dihubungi. Pastikan uvicorn sudah jalan di port 8000.", "{}", "{}"

    progress(0.10, desc="Mengirim audio ke backend...")
    t_start = time.time()

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Kirim request dengan timeout besar
        progress(0.20, desc="⏳ STT: Whisper sedang transkripsi audio...")
        resp = requests.post(
            f"{BACKEND_URL}/voice-chat",
            files={"audio": ("input.wav", audio_bytes, "audio/wav")},
            data={
                "mode": mode,
                "stt_language": stt_lang,
                "segment_tts": str(segment_tts).lower(),
                "save_log": "true",
            },
            timeout=TIMEOUT_FULL,
            stream=False,
        )

        elapsed = round(time.time() - t_start, 1)

        if resp.status_code == 200:
            progress(0.90, desc="Menerima audio respons...")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(resp.content)
            tmp.close()

            transcript    = resp.headers.get("X-Transcript",    "(tidak tersedia)")
            response_text = resp.headers.get("X-Response-Text", "(tidak tersedia)")
            latency       = resp.headers.get("X-Total-Latency-S", str(elapsed))
            mode_used     = resp.headers.get("X-Mode", mode)

            progress(1.0, desc="Selesai!")
            status = f"✅ Selesai | Mode: {mode_used} | Latency backend: {latency}s | Total: {elapsed}s"

            stt_info = json.dumps({
                "Transkripsi": transcript,
                "Mode": mode_used,
                "Latency backend (s)": latency,
                "Waktu total request (s)": elapsed,
            }, ensure_ascii=False, indent=2)

            llm_info = json.dumps({
                "Respons LLM": response_text,
                "Mode": mode_used,
            }, ensure_ascii=False, indent=2)

            return tmp.name, status, stt_info, llm_info

        else:
            try:
                detail = resp.json().get("detail", resp.text[:300])
            except Exception:
                detail = resp.text[:300]
            return None, f"❌ Backend error {resp.status_code}: {detail}", "{}", "{}"

    except requests.exceptions.Timeout:
        elapsed = round(time.time() - t_start, 1)
        msg = (
            f"⏰ Timeout setelah {elapsed}s.\n\n"
            "Kemungkinan penyebab:\n"
            "• Whisper masih load model (normal di percobaan pertama, tunggu ~2 menit)\n"
            "• Model Coqui TTS belum tersedia (TTS fallback ke gTTS yang lambat)\n"
            "• File audio terlalu panjang (>30 detik)\n\n"
            f"Timeout saat ini: {TIMEOUT_FULL}s. Naikkan di .env: TIMEOUT_FULL=600"
        )
        return None, msg, "{}", "{}"

    except requests.exceptions.ConnectionError:
        return None, "🔴 Koneksi terputus. Backend mungkin crash — cek terminal uvicorn.", "{}", "{}"

    except Exception as e:
        return None, f"❌ Error tak terduga: {str(e)}", "{}", "{}"


# ── STT Only ─────────────────────────────────────────────────────

def call_transcribe_only(audio_path, stt_lang, progress=gr.Progress()):
    if audio_path is None:
        return "Tidak ada audio.", "{}"
    try:
        progress(0.2, desc="Mengirim audio ke Whisper...")
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/transcribe",
                files={"audio": ("input.wav", f, "audio/wav")},
                data={"language": stt_lang},
                timeout=TIMEOUT_STT,
            )
        progress(0.9, desc="Menerima hasil transkripsi...")
        if resp.status_code == 200:
            data = resp.json()
            detail = json.dumps({
                "Transkripsi"       : data.get("transcript"),
                "Normalisasi"       : data.get("normalized"),
                "Bahasa terdeteksi" : data.get("detected_language"),
                "Bahasa dominan"    : data.get("dominant_language"),
                "Tagging"           : data.get("tagging_summary"),
                "Durasi audio (s)"  : data.get("audio_duration_s"),
                "Waktu proses (s)"  : data.get("processing_time_s"),
            }, ensure_ascii=False, indent=2)
            progress(1.0, desc="Selesai!")
            return data.get("transcript", ""), detail
        return f"Error {resp.status_code}: {resp.text[:200]}", "{}"
    except requests.exceptions.Timeout:
        return f"⏰ Timeout ({TIMEOUT_STT}s). Whisper masih lambat — coba model lebih kecil (base/small).", "{}"
    except Exception as e:
        return f"❌ {str(e)}", "{}"


# ── LLM Only ─────────────────────────────────────────────────────

def call_text_chat(text, mode, progress=gr.Progress()):
    if not text.strip():
        return "Masukkan teks dulu.", "{}"
    try:
        progress(0.3, desc="Mengirim ke Gemini API...")
        resp = requests.post(
            f"{BACKEND_URL}/text-chat",
            data={"text": text, "mode": mode},
            timeout=TIMEOUT_LLM,
        )
        progress(0.9, desc="Menerima respons LLM...")
        if resp.status_code == 200:
            data = resp.json()
            detail = json.dumps({
                "Input"           : data.get("input"),
                "Normalisasi"     : data.get("normalized"),
                "Tagging"         : data.get("tagging_summary"),
                "Respons"         : data.get("response"),
                "Model"           : data.get("model"),
                "Waktu proses (s)": data.get("processing_time_s"),
            }, ensure_ascii=False, indent=2)
            progress(1.0, desc="Selesai!")
            return data.get("response", ""), detail
        return f"Error {resp.status_code}: {resp.text[:200]}", "{}"
    except requests.exceptions.Timeout:
        return f"⏰ Timeout Gemini ({TIMEOUT_LLM}s). Cek rate limit atau koneksi internet.", "{}"
    except Exception as e:
        return f"❌ {str(e)}", "{}"


def reset_conv():
    try:
        resp = requests.post(f"{BACKEND_URL}/reset-conversation", timeout=5)
        return "✅ History percakapan direset." if resp.status_code == 200 else f"Error: {resp.text}"
    except Exception as e:
        return f"❌ {str(e)}"


# ── Bangun UI ────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Voice CS System",
        theme=gr.themes.Soft(primary_hue="indigo"),
    ) as demo:

        gr.Markdown("""
        # 🎙️ Voice CS System
        Pipeline: `Audio → STT (Whisper) → Normalisasi/Tagging → LLM (Gemini) → TTS (Coqui) → Audio`
        """)

        # Status backend
        with gr.Row():
            backend_status = gr.Textbox(
                label="Status Backend",
                value=check_backend(),
                interactive=False,
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)
        refresh_btn.click(fn=check_backend, outputs=backend_status)

        with gr.Tabs():

            # ── Tab 1: Pipeline Lengkap ──────────────────────────
            with gr.Tab("🔄 Pipeline Lengkap"):
                gr.Markdown(
                    f"> **Timeout saat ini: {TIMEOUT_FULL}s.** "
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input  = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Input Audio")
                        mode_select  = gr.Radio(
                            choices=["preserve", "normalize", "translate"],
                            value="preserve", label="Mode Sistem",
                            info="preserve=pertahankan CS | normalize=satu bahasa | translate=terjemahkan",
                        )
                        stt_lang     = gr.Dropdown(choices=["auto", "id", "en", "ar"], value="auto", label="Bahasa STT")
                        segment_tts  = gr.Checkbox(label="Segmentasi TTS per bahasa", value=False)
                        with gr.Row():
                            run_btn   = gr.Button("▶ Jalankan", variant="primary")
                            reset_btn = gr.Button("🔄 Reset Chat")

                    with gr.Column(scale=1):
                        audio_output = gr.Audio(label="🔊 Respons Suara")
                        status_box   = gr.Textbox(label="Status Pipeline", interactive=False, lines=4)
                        with gr.Accordion("📊 Detail STT + Processing", open=False):
                            stt_detail = gr.Code(language="json", label="STT Output")
                        with gr.Accordion("🤖 Detail LLM", open=False):
                            llm_detail = gr.Code(language="json", label="LLM Output")

                run_btn.click(
                    fn=call_voice_chat,
                    inputs=[audio_input, mode_select, stt_lang, segment_tts],
                    outputs=[audio_output, status_box, stt_detail, llm_detail],
                )
                reset_btn.click(fn=reset_conv, outputs=status_box)

            # ── Tab 2: Test STT ──────────────────────────────────
            with gr.Tab("🎤 Test STT"):
                gr.Markdown(f"Test Whisper saja. Timeout: **{TIMEOUT_STT}s**")
                with gr.Row():
                    with gr.Column():
                        stt_audio    = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                        stt_lang_only = gr.Dropdown(choices=["auto", "id", "en", "ar"], value="auto", label="Bahasa")
                        stt_btn      = gr.Button("▶ Transkripsi", variant="primary")
                    with gr.Column():
                        stt_result   = gr.Textbox(label="Hasil Transkripsi", lines=3)
                        stt_detail_o = gr.Code(language="json", label="Detail")
                stt_btn.click(fn=call_transcribe_only, inputs=[stt_audio, stt_lang_only], outputs=[stt_result, stt_detail_o])

            # ── Tab 3: Test LLM ──────────────────────────────────
            with gr.Tab("🤖 Test LLM"):
                gr.Markdown(f"Test Gemini API dengan input teks. Timeout: **{TIMEOUT_LLM}s**")
                with gr.Row():
                    with gr.Column():
                        text_input   = gr.Textbox(label="Input Teks", placeholder="Saya mau tanya, how do you say terima kasih in Arabic?", lines=3)
                        mode_text    = gr.Radio(choices=["preserve", "normalize"], value="preserve", label="Mode")
                        text_btn     = gr.Button("▶ Generate", variant="primary")
                    with gr.Column():
                        text_resp    = gr.Textbox(label="Respons LLM", lines=4)
                        text_detail  = gr.Code(language="json", label="Detail")
                text_btn.click(fn=call_text_chat, inputs=[text_input, mode_text], outputs=[text_resp, text_detail])

            # ── Tab 4: Diagnostik ────────────────────────────────
            with gr.Tab("🔧 Diagnostik"):
                gr.Markdown("""
                ## Troubleshooting Timeout

                ### Jika Whisper timeout
                Tambahkan di file `.env`:
                ```
                TIMEOUT_STT=300
                TIMEOUT_FULL=600
                WHISPER_MODEL_PATH=models/whisper.cpp/models/ggml-base.bin
                ```
                Model `base` jauh lebih cepat dari `large-v3-turbo` (~10x).

                ### Jika TTS timeout
                Pastikan model Coqui TTS sudah ada di `app/coqui_tts/`.
                Jika belum, sistem fallback ke **gTTS** yang butuh internet.

                ### Cek log backend
                Lihat output terminal tempat `uvicorn` berjalan untuk melihat
                tahap mana yang paling lambat (ditandai `[STT]`, `[LLM]`, `[TTS]`).

                ### Format audio
                Pastikan file `.wav`, mono, sample rate 16000Hz untuk hasil terbaik.
                Konversi dengan: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
                """)
                diag_status = gr.Textbox(label="Status Backend", interactive=False)
                diag_btn    = gr.Button("Cek Koneksi Backend")
                diag_btn.click(fn=check_backend, outputs=diag_status)

        gr.Markdown("---\n*Program Studi Informatika, Universitas Syiah Kuala | UAS Praktikum NLP 2025/2026 Genap*")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)