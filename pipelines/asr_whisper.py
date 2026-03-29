# file: asr_whisper.py
import os, sys, subprocess, tempfile, shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

print("[debug] python:", sys.executable)

# Resolver ffmpeg: copiar para 'ffmpeg.exe' real em pasta temporária e colocar no PATH ===
try:
    import imageio_ffmpeg as iioff
    ff_src = Path(iioff.get_ffmpeg_exe())  # ex.: .../ffmpeg-win-x86_64-v7.1.exe
    print("[debug] ffmpeg real exe (imageio):", ff_src)

    WRAP_DIR = Path(tempfile.mkdtemp(prefix="ffmpeg_wrap_"))
    ff_dst = WRAP_DIR / "ffmpeg.exe"
    # copia binário para nome exato 'ffmpeg.exe'
    shutil.copyfile(ff_src, ff_dst)

    # opcional: ffprobe (algumas libs tentam chamar; podemos apontar para o mesmo binário)
    fp_dst = WRAP_DIR / "ffprobe.exe"
    shutil.copyfile(ff_src, fp_dst)

    # prepend no PATH
    os.environ["PATH"] = str(WRAP_DIR) + os.pathsep + os.environ.get("PATH", "")
    print("[debug] wrap dir:", WRAP_DIR)

    # sanity check
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception as e:
    print("[erro] não consegui preparar ffmpeg.exe:", e)
    print("       Rode:  .\\.venv\\Scripts\\python.exe -m pip install imageio-ffmpeg")
    sys.exit(1)

# Conferir yt-dlp no MESMO Python (via módulo) ===
try:
    import yt_dlp
    print("[debug] yt_dlp version:", getattr(yt_dlp, "__version__", "unknown"))
except Exception as e:
    print("[erro] yt-dlp não encontrado neste Python:", e)
    print("       Rode:  .\\.venv\\Scripts\\python.exe -m pip install yt-dlp")
    sys.exit(1)

import whisper

AUDIO_DIR = Path("data/audio"); AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "small"          # "base" = mais rápido; "small/medium" = melhor qualidade
TOP_N = 3                     # teste curto
USE_CPU = True                # se tiver CUDA, mude para False

def ytdlp_best_audio(video_id: str) -> Path:
    """Baixa o melhor áudio (sem conversão) usando ESTE Python (do venv)."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Reaproveita se já existe
    for f in AUDIO_DIR.glob(f"{video_id}.*"):
        if f.suffix.lower() in [".webm",".m4a",".opus",".mp3",".wav",".flac"]:
            print(f"[debug] reuse audio: {f.name}")
            return f

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio",
        "-o", str(AUDIO_DIR / f"{video_id}%(autonumber)s.%(ext)s"),  # evita colidir nomes
        url,
    ]
    print("[debug] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # pega o arquivo mais recente com esse prefixo
    candidates = sorted(AUDIO_DIR.glob(f"{video_id}*.webm")) \
              + sorted(AUDIO_DIR.glob(f"{video_id}*.m4a")) \
              + sorted(AUDIO_DIR.glob(f"{video_id}*.opus")) \
              + sorted(AUDIO_DIR.glob(f"{video_id}*.mp3")) \
              + sorted(AUDIO_DIR.glob(f"{video_id}*.wav")) \
              + sorted(AUDIO_DIR.glob(f"{video_id}*.flac"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum áudio encontrado para {video_id}")
    return candidates[-1]

def main():
    src = "outputs/videos_top50.csv"
    print("[debug] source csv:", src)
    df = pd.read_csv(src)

    if "viewCount" in df.columns:
        df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce")
        df = df.sort_values("viewCount", ascending=False)

    vids = df["videoId"].dropna().unique().tolist()[:TOP_N]
    if not vids:
        print("[erro] Nenhum videoId encontrado no CSV.")
        sys.exit(1)
    print("[debug] vids:", vids)

    model = whisper.load_model(MODEL_NAME)
    transcribe_kwargs = {"fp16": False} if USE_CPU else {}

    rows = []
    try:
        for vid in tqdm(vids, desc="Transcrevendo"):
            try:
                audio_file = ytdlp_best_audio(vid)
                # whisper usa 'ffmpeg.exe' (que acabamos de injetar) para decodificar
                result = whisper.transcribe(model, str(audio_file), **transcribe_kwargs)
                rows.append({
                    "videoId": vid,
                    "language": result.get("language"),
                    "text": result.get("text")
                })
            except subprocess.CalledProcessError as e:
                print(f"[warn] download falhou para {vid}: {e}")
            except Exception as e:
                print(f"[warn] {vid} falhou:", repr(e))
    except KeyboardInterrupt:
        print("\n[info] Interrompido. Salvando parciais...")

    out = pd.DataFrame(rows)
    out.to_csv("outputs/transcripts.csv", index=False)
    print(f" Salvo transcripts.csv com {len(out)} linhas (TOP_N={TOP_N})")

if __name__ == "__main__":
    main()
