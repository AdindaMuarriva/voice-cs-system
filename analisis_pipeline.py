"""
analisis_pipeline.py
Script batch untuk menguji seluruh korpus audio code-switching
melalui pipeline penuh dan menghasilkan laporan evaluasi.

Jalankan: python analisis_pipeline.py
"""

import os
import sys
import json
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime

# Pastikan bisa import modul dari app/
sys.path.insert(0, str(Path(__file__).parent))

from app.stt import transcribe, compute_wer, compute_cer
from app.llm import generate_response, reset_conversation
from app.tts import synthesize_with_fallback
from app.utils import (
    normalize_text,
    tag_code_switching,
    detect_dominant_language,
    split_by_language,
    build_tagging_summary,
)

# ── Konfigurasi ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CORPUS_DIR  = BASE_DIR / "data" / "corpus" / "audio"
TRANS_DIR   = BASE_DIR / "data" / "corpus" / "transcripts"
OUTPUT_DIR  = BASE_DIR / "log" / "analisis"
REPORT_DIR  = BASE_DIR / "log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)

# ── Referensi transkripsi (isi manual jika ada) ──────────────────
# Format: {"nama_file.wav": "teks referensi transkripsi"}
REFERENCE_TRANSCRIPTS: dict[str, str] = {
    # Contoh:
    # "2030_audio01.wav": "Saya sudah submit the assignment tapi masih ada masalah",
    # "2030_audio02.wav": "How do you say terima kasih in Arabic",
}


def process_single_audio(
    audio_path: Path,
    mode: str = "preserve",
    stt_language: str = "auto",
) -> dict:
    """
    Proses satu file audio melalui pipeline lengkap.
    Return dict berisi semua hasil dan metrik.
    """
    result = {
        "file": audio_path.name,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "error": None,
    }
    
    # ── STT ──────────────────────────────────────────────────────
    print(f"  [STT] Transkripsi {audio_path.name}...")
    try:
        stt_start = time.time()
        stt_result = transcribe(str(audio_path), language=stt_language)
        result["stt"] = {
            "transcript": stt_result["text"],
            "detected_language": stt_result.get("language"),
            "audio_duration_s": stt_result.get("duration_s"),
            "processing_time_s": round(time.time() - stt_start, 2),
            "source": stt_result.get("source"),
        }
        
        # Hitung WER/CER jika ada referensi
        ref = REFERENCE_TRANSCRIPTS.get(audio_path.name, "")
        if ref:
            result["stt"]["wer"] = compute_wer(ref, stt_result["text"])
            result["stt"]["cer"] = compute_cer(ref, stt_result["text"])
            result["stt"]["reference"] = ref
        
    except Exception as e:
        result["error"] = f"STT error: {str(e)}"
        print(f"  ❌ STT error: {e}")
        return result
    
    # ── Normalisasi & Tagging ─────────────────────────────────────
    transcript = stt_result["text"]
    try:
        normalized = normalize_text(transcript)
        tagged_tokens = tag_code_switching(transcript)
        dominant_lang = detect_dominant_language(transcript)
        tagging_summary = build_tagging_summary(tagged_tokens)
        
        result["processing"] = {
            "normalized": normalized,
            "dominant_language": dominant_lang,
            "tagging_summary": tagging_summary,
            "token_count": len(tagged_tokens),
            "language_distribution": _count_lang_dist(tagged_tokens),
        }
        print(f"  [PROC] Dominant: {dominant_lang} | Tagging: {tagging_summary}")
        
    except Exception as e:
        result["error"] = f"Processing error: {str(e)}"
        return result
    
    # ── LLM ──────────────────────────────────────────────────────
    print(f"  [LLM] Generate respons (mode={mode})...")
    try:
        llm_start = time.time()
        llm_result = generate_response(
            transcript=normalized,
            mode=mode,
            tagged_text=tagging_summary,
        )
        result["llm"] = {
            "response_text": llm_result["response_text"],
            "model": llm_result["model"],
            "processing_time_s": round(time.time() - llm_start, 2),
            "finish_reason": llm_result.get("finish_reason"),
            "response_word_count": len(llm_result["response_text"].split()),
        }
        print(f"  [LLM] Respons: {llm_result['response_text'][:80]}...")
        
    except Exception as e:
        result["error"] = f"LLM error: {str(e)}"
        print(f"  ❌ LLM error: {e}")
        return result
    
    # ── TTS ──────────────────────────────────────────────────────
    print(f"  [TTS] Sintesis suara...")
    try:
        tts_output_path = str(
            OUTPUT_DIR / f"tts_{audio_path.stem}_{mode}.wav"
        )
        tts_start = time.time()
        tts_result = synthesize_with_fallback(
            text=llm_result["response_text"],
            output_path=tts_output_path,
            lang=dominant_lang if dominant_lang not in ["mixed", "unknown"] else "id",
        )
        result["tts"] = {
            "output_path": tts_result["audio_path"],
            "duration_s": tts_result.get("duration_s"),
            "processing_time_s": round(time.time() - tts_start, 2),
        }
        print(f"  [TTS] Selesai: {tts_result.get('duration_s')}s audio")
        
    except Exception as e:
        result["tts"] = {"error": str(e)}
        print(f"  ⚠️  TTS error (tidak fatal): {e}")
    
    # ── Latency summary ───────────────────────────────────────────
    result["latency"] = {
        "stt_s": result.get("stt", {}).get("processing_time_s", 0),
        "llm_s": result.get("llm", {}).get("processing_time_s", 0),
        "tts_s": result.get("tts", {}).get("processing_time_s", 0),
    }
    total = sum(result["latency"].values())
    result["latency"]["total_s"] = round(total, 2)
    
    return result


def _count_lang_dist(tagged_tokens: list[dict]) -> dict:
    """Hitung distribusi bahasa dari token."""
    dist = {}
    for tok in tagged_tokens:
        lang = tok["lang"]
        dist[lang] = dist.get(lang, 0) + 1
    return dist


def run_pipeline_analysis(
    corpus_dir: Path = CORPUS_DIR,
    mode: str = "preserve",
    stt_language: str = "auto",
    limit: int | None = None,
    sleep_between: float = 2.0,
) -> list[dict]:
    """
    Jalankan analisis pipeline untuk seluruh file audio di corpus_dir.
    
    Args:
        corpus_dir: Folder berisi file .wav
        mode: Mode pipeline
        limit: Batasi jumlah file (None = semua)
        sleep_between: Jeda antar file (detik) untuk menghindari rate limit
    
    Returns:
        List hasil per file
    """
    audio_files = sorted(corpus_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"❌ Tidak ada file .wav di: {corpus_dir}")
        print("   Pastikan file audio sudah di-copy ke data/corpus/audio/")
        return []
    
    if limit:
        audio_files = audio_files[:limit]
    
    print(f"\n{'='*60}")
    print(f"📊 ANALISIS PIPELINE CORPUS")
    print(f"   Folder  : {corpus_dir}")
    print(f"   Jumlah  : {len(audio_files)} file")
    print(f"   Mode    : {mode}")
    print(f"   Bahasa  : {stt_language}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Memproses: {audio_path.name}")
        
        result = process_single_audio(audio_path, mode=mode, stt_language=stt_language)
        all_results.append(result)
        
        # Simpan hasil per file
        result_path = OUTPUT_DIR / f"result_{audio_path.stem}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if result["error"]:
            print(f"  ⚠️  Ada error: {result['error']}")
        else:
            print(f"  ✅ Selesai | Latency total: {result['latency'].get('total_s', '?')}s")
        
        # Jeda antar file
        if i < len(audio_files):
            time.sleep(sleep_between)
    
    return all_results


def generate_report(results: list[dict], output_prefix: str = "laporan_analisis"):
    """
    Generate laporan evaluasi lengkap:
    - JSON: data mentah semua hasil
    - CSV: ringkasan per file
    - TXT: laporan teks yang bisa dibaca
    """
    if not results:
        print("Tidak ada hasil untuk dilaporkan.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ── Hitung statistik agregat ──────────────────────────────────
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    
    wer_scores = [r["stt"]["wer"] for r in successful if "wer" in r.get("stt", {})]
    cer_scores = [r["stt"]["cer"] for r in successful if "cer" in r.get("stt", {})]
    total_latencies = [r["latency"]["total_s"] for r in successful if "latency" in r]
    stt_times = [r["stt"]["processing_time_s"] for r in successful if "stt" in r]
    llm_times = [r["llm"]["processing_time_s"] for r in successful if "llm" in r]
    tts_times = [r["tts"]["processing_time_s"] for r in successful if "tts" in r and "error" not in r["tts"]]
    
    lang_dist_total: dict[str, int] = {}
    for r in successful:
        for lang, count in r.get("processing", {}).get("language_distribution", {}).items():
            lang_dist_total[lang] = lang_dist_total.get(lang, 0) + count
    
    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0
    
    stats = {
        "total_files": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": f"{len(successful)/len(results)*100:.1f}%",
        "stt_metrics": {
            "avg_wer": avg(wer_scores) if wer_scores else "N/A (tidak ada referensi)",
            "avg_cer": avg(cer_scores) if cer_scores else "N/A (tidak ada referensi)",
            "avg_processing_time_s": avg(stt_times),
        },
        "llm_metrics": {
            "avg_processing_time_s": avg(llm_times),
        },
        "tts_metrics": {
            "avg_processing_time_s": avg(tts_times),
        },
        "latency": {
            "avg_total_s": avg(total_latencies),
            "min_total_s": round(min(total_latencies), 2) if total_latencies else 0,
            "max_total_s": round(max(total_latencies), 2) if total_latencies else 0,
        },
        "language_distribution_total": lang_dist_total,
        "failed_files": [r["file"] for r in failed],
    }
    
    # ── Simpan JSON laporan ───────────────────────────────────────
    json_report = REPORT_DIR / f"{output_prefix}_{timestamp}.json"
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump({"statistics": stats, "results": results}, f, ensure_ascii=False, indent=2)
    
    # ── Simpan CSV ringkasan ──────────────────────────────────────
    csv_report = REPORT_DIR / f"{output_prefix}_{timestamp}.csv"
    with open(csv_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Mode", "Error",
            "Transcript", "Dominant Lang", "Tagging Summary",
            "LLM Response",
            "WER", "CER",
            "STT Time (s)", "LLM Time (s)", "TTS Time (s)", "Total Latency (s)"
        ])
        for r in results:
            writer.writerow([
                r.get("file", ""),
                r.get("mode", ""),
                r.get("error", ""),
                r.get("stt", {}).get("transcript", ""),
                r.get("processing", {}).get("dominant_language", ""),
                r.get("processing", {}).get("tagging_summary", ""),
                r.get("llm", {}).get("response_text", ""),
                r.get("stt", {}).get("wer", ""),
                r.get("stt", {}).get("cer", ""),
                r.get("stt", {}).get("processing_time_s", ""),
                r.get("llm", {}).get("processing_time_s", ""),
                r.get("tts", {}).get("processing_time_s", ""),
                r.get("latency", {}).get("total_s", ""),
            ])
    
    # ── Simpan laporan teks ───────────────────────────────────────
    txt_report = REPORT_DIR / f"{output_prefix}_{timestamp}.txt"
    with open(txt_report, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LAPORAN ANALISIS PIPELINE CODE-SWITCHING SPEECH-TO-SPEECH\n")
        f.write(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## STATISTIK AGREGAT\n")
        f.write(f"Total file diproses  : {stats['total_files']}\n")
        f.write(f"Berhasil             : {stats['successful']}\n")
        f.write(f"Gagal                : {stats['failed']}\n")
        f.write(f"Tingkat keberhasilan : {stats['success_rate']}\n\n")
        
        f.write("## METRIK STT\n")
        for k, v in stats["stt_metrics"].items():
            f.write(f"  {k}: {v}\n")
        
        f.write("\n## LATENCY\n")
        for k, v in stats["latency"].items():
            f.write(f"  {k}: {v}s\n")
        
        f.write("\n## DISTRIBUSI BAHASA (token)\n")
        for lang, count in stats["language_distribution_total"].items():
            label = {"id": "Indonesia", "en": "Inggris", "ar": "Arab"}.get(lang, lang)
            f.write(f"  {label}: {count} token\n")
        
        if stats["failed_files"]:
            f.write("\n## FILE YANG GAGAL DIPROSES\n")
            for fname in stats["failed_files"]:
                f.write(f"  - {fname}\n")
        
        f.write("\n## DETAIL PER FILE\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"\nFile: {r.get('file')}\n")
            if r.get("error"):
                f.write(f"  ERROR: {r['error']}\n")
                continue
            f.write(f"  Transkripsi : {r.get('stt', {}).get('transcript', '')}\n")
            f.write(f"  Dominan     : {r.get('processing', {}).get('dominant_language', '')}\n")
            f.write(f"  Tagging     : {r.get('processing', {}).get('tagging_summary', '')}\n")
            f.write(f"  Respons LLM : {r.get('llm', {}).get('response_text', '')[:100]}...\n")
            f.write(f"  Latency     : {r.get('latency', {}).get('total_s', '?')}s\n")
            if "wer" in r.get("stt", {}):
                f.write(f"  WER         : {r['stt']['wer']}\n")
                f.write(f"  CER         : {r['stt']['cer']}\n")
    
    print(f"\n{'='*60}")
    print("📄 LAPORAN TERSIMPAN:")
    print(f"   JSON : {json_report}")
    print(f"   CSV  : {csv_report}")
    print(f"   TXT  : {txt_report}")
    print(f"{'='*60}")
    print(f"\n✅ Berhasil: {stats['successful']}/{stats['total_files']}")
    print(f"⚡ Rata-rata latency: {stats['latency']['avg_total_s']}s")
    if wer_scores:
        print(f"📊 Rata-rata WER: {stats['stt_metrics']['avg_wer']}")
    print()
    
    return str(json_report), str(csv_report), str(txt_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis pipeline corpus CS Speech-to-Speech")
    parser.add_argument("--corpus", type=str, default=str(CORPUS_DIR), help="Folder audio corpus")
    parser.add_argument("--mode", type=str, default="preserve", choices=["preserve", "normalize", "translate"])
    parser.add_argument("--lang", type=str, default="auto", help="Bahasa STT: auto|id|en|ar")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah file yang diproses")
    parser.add_argument("--sleep", type=float, default=2.0, help="Jeda antar file (detik)")
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus)
    
    if not corpus_path.exists():
        print(f"❌ Folder tidak ditemukan: {corpus_path}")
        print("   Buat folder dan masukkan file audio .wav ke dalamnya.")
        sys.exit(1)
    
    results = run_pipeline_analysis(
        corpus_dir=corpus_path,
        mode=args.mode,
        stt_language=args.lang,
        limit=args.limit,
        sleep_between=args.sleep,
    )
    
    if results:
        generate_report(results)
