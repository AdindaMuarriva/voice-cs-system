"""
resume_pipeline.py
Lanjutkan analisis hanya untuk file yang belum diproses atau gagal di LLM/TTS.
File yang sudah berhasil penuh akan di-skip otomatis.

Jalankan: python resume_pipeline.py --limit 230 --sleep 35
"""

import json
import time
import argparse
from pathlib import Path

# Import modul pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))

from analisis_pipeline import (
    process_single_audio,
    generate_report,
    CORPUS_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
)


def load_existing_results() -> dict[str, dict]:
    """
    Baca semua hasil yang sudah ada di log/analisis/.
    Return dict: {nama_file.wav: result_dict}
    """
    existing = {}
    for json_file in OUTPUT_DIR.glob("result_*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            fname = data.get("file", "")
            if fname:
                existing[fname] = data
        except Exception:
            pass
    return existing


def is_fully_successful(result: dict) -> bool:
    """
    Cek apakah satu file sudah berhasil diproses penuh
    (STT + LLM + TTS semuanya tidak error).
    """
    if result.get("error"):
        return False
    if not result.get("stt", {}).get("transcript"):
        return False
    if not result.get("llm", {}).get("response_text"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Resume analisis pipeline dari file yang gagal")
    parser.add_argument("--corpus", type=str, default=str(CORPUS_DIR))
    parser.add_argument("--mode",   type=str, default="preserve",
                        choices=["preserve", "normalize", "translate"])
    parser.add_argument("--lang",   type=str, default="auto")
    parser.add_argument("--limit",  type=int, default=230,
                        help="Total target file (termasuk yang sudah selesai)")
    parser.add_argument("--sleep",  type=float, default=35.0,
                        help="Jeda antar request LLM (detik)")
    parser.add_argument("--retry-llm-only", action="store_true",
                        help="Hanya retry file yang STT-nya sudah berhasil tapi LLM gagal")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    all_audio   = sorted(corpus_path.glob("*.wav"))[:args.limit]

    if not all_audio:
        print(f"❌ Tidak ada .wav di {corpus_path}")
        return

    # Baca hasil yang sudah ada
    existing = load_existing_results()

    # Kategorikan file
    done    = [f for f in all_audio if is_fully_successful(existing.get(f.name, {}))]
    pending = [f for f in all_audio if not is_fully_successful(existing.get(f.name, {}))]

    # Jika --retry-llm-only, filter hanya yang STT-nya sudah ada tapi LLM kosong
    if args.retry_llm_only:
        pending = [
            f for f in pending
            if existing.get(f.name, {}).get("stt", {}).get("transcript")
            and not existing.get(f.name, {}).get("llm", {}).get("response_text")
        ]

    print(f"\n{'='*60}")
    print(f"📊 RESUME ANALISIS PIPELINE")
    print(f"   Target total  : {len(all_audio)} file")
    print(f"   Sudah selesai : {len(done)} file (di-skip)")
    print(f"   Perlu diproses: {len(pending)} file")
    print(f"   Mode          : {args.mode}")
    print(f"   Sleep LLM     : {args.sleep}s")
    print(f"{'='*60}\n")

    if not pending:
        print("✅ Semua file sudah berhasil diproses!")
    else:
        # Estimasi waktu
        est_minutes = round(len(pending) * args.sleep / 60, 1)
        print(f"⏱️  Estimasi waktu minimum: ~{est_minutes} menit\n")

        new_results = []
        for i, audio_path in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Memproses: {audio_path.name}")

            # Jika ada hasil parsial (misalnya STT sudah ada tapi LLM gagal),
            # coba lanjutkan dari LLM saja
            existing_result = existing.get(audio_path.name, {})
            if existing_result.get("stt", {}).get("transcript") and args.retry_llm_only:
                result = _retry_llm_only(existing_result, args.mode, args.sleep)
            else:
                result = process_single_audio(audio_path, mode=args.mode, stt_language=args.lang)

            new_results.append(result)

            # Simpan/update hasil
            result_path = OUTPUT_DIR / f"result_{audio_path.stem}.json"
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            if result.get("error"):
                print(f"  ⚠️  Error: {result['error'][:80]}")
            else:
                print(f"  ✅ OK | Latency: {result.get('latency', {}).get('total_s', '?')}s")

            # Jeda antar file untuk hindari rate limit
            if i < len(pending):
                time.sleep(args.sleep)

    # Kumpulkan SEMUA hasil (lama + baru) untuk laporan final
    print("\n📄 Menggenerate laporan final dari semua hasil...")
    all_results = []
    for audio_path in all_audio:
        result_file = OUTPUT_DIR / f"result_{audio_path.stem}.json"
        if result_file.exists():
            try:
                all_results.append(
                    json.loads(result_file.read_text(encoding="utf-8"))
                )
            except Exception:
                pass

    if all_results:
        generate_report(all_results, output_prefix="laporan_final")
    else:
        print("❌ Tidak ada hasil untuk dilaporkan.")


def _retry_llm_only(existing_result: dict, mode: str, sleep_s: float) -> dict:
    """
    Gunakan ulang hasil STT yang sudah ada, hanya retry bagian LLM dan TTS.
    """
    import time
    from app.llm import generate_response
    from app.tts import synthesize_with_fallback
    from app.utils import build_tagging_summary, detect_dominant_language

    result = dict(existing_result)
    transcript  = result["stt"]["transcript"]
    normalized  = result.get("processing", {}).get("normalized", transcript)
    tagging_sum = result.get("processing", {}).get("tagging_summary", "")
    dominant    = result.get("processing", {}).get("dominant_language", "id")

    print(f"  [LLM] Retry LLM dari STT yang sudah ada...")
    try:
        llm_start  = time.time()
        llm_result = generate_response(
            transcript=normalized,
            mode=mode,
            tagged_text=tagging_sum,
        )
        result["llm"] = {
            "response_text"    : llm_result["response_text"],
            "model"            : llm_result["model"],
            "processing_time_s": round(time.time() - llm_start, 2),
            "finish_reason"    : llm_result.get("finish_reason"),
        }
        result["error"] = None
        print(f"  [LLM] OK: {llm_result['response_text'][:60]}...")
    except Exception as e:
        result["error"] = f"LLM error: {str(e)}"
        print(f"  ❌ LLM retry gagal: {e}")
        return result

    # TTS
    try:
        from pathlib import Path as P
        out_path = str(OUTPUT_DIR / f"tts_{P(result['file']).stem}_{mode}.wav")
        tts_start  = time.time()
        tts_result = synthesize_with_fallback(
            text=llm_result["response_text"],
            output_path=out_path,
            lang=dominant if dominant not in ["mixed", "unknown"] else "id",
        )
        result["tts"] = {
            "output_path"      : tts_result.get("audio_path"),
            "duration_s"       : tts_result.get("duration_s"),
            "processing_time_s": round(time.time() - tts_start, 2),
        }
    except Exception as e:
        result["tts"] = {"error": str(e)}

    return result


if __name__ == "__main__":
    main()