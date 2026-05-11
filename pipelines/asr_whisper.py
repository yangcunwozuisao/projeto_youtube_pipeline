"""
asr_whisper.py — Transcrição de áudio com OpenAI Whisper via yt-dlp.

Correção: o código anterior usava extensão .exe para o ffmpeg, quebrando
em Linux e macOS. Agora a extensão é determinada pelo sistema operacional.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

print("[debug] python:", sys.executable)
print("[debug] sistema:", platform.system())

# Configuração de extensão por plataforma
_IS_WINDOWS = platform.system() == "Windows"
_EXE_EXT    = ".exe" if _IS_WINDOWS else ""

#Preparação do ffmpeg
try:
    import imageio_ffmpeg as iioff

    ff_src = Path(iioff.get_ffmpeg_exe())
    print("[debug] ffmpeg real exe (imageio):", ff_src)

    WRAP_DIR  = Path(tempfile.mkdtemp(prefix="ffmpeg_wrap_"))
    ff_dst    = WRAP_DIR / f"ffmpeg{_EXE_EXT}"
    fp_dst    = WRAP_DIR / f"ffprobe{_EXE_EXT}"

    shutil.copyfile(ff_src, ff_dst)
    shutil.copyfile(ff_src, fp_dst)

    # Em sistemas Unix, garantir permissão de execução
    if not _IS_WINDOWS:
        ff_dst.chmod(0o755)
        fp_dst.chmod(0o755)

    os.environ["PATH"] = str(WRAP_DIR) + os.pathsep + os.environ.get("PATH", "")
    print("[debug] wrap dir:", WRAP_DIR)

    subprocess.run(
        [str(ff_dst), "-version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

except Exception as e:
    print("[erro] não consegui preparar ffmpeg:", e)
    print("       Rode: pip install imageio-ffmpeg")
    sys.exit(1)

# Verificação do yt-dlp
try:
    import yt_dlp
    print("[debug] yt_dlp version:", getattr(yt_dlp, "__version__", "unknown"))
except Exception as e:
    print("[erro] yt-dlp não encontrado:", e)
    print("       Rode: pip install yt-dlp")
    sys.exit(1)

import whisper

# Constantes
AUDIO_DIR    = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "base"
USE_CPU      = True
DEFAULT_TOP_N = 20

AUDIO_EXTS   = [".webm", ".m4a", ".opus", ".mp3", ".wav", ".flac"]

def get_top_n() -> int:
    raw = os.getenv("ASR_TOP_N")
    if raw is None or raw == "":
        return DEFAULT_TOP_N
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("ASR_TOP_N deve ser um número inteiro.")
    return value if value > 0 else DEFAULT_TOP_N


def ytdlp_best_audio(video_id: str) -> Path:
    url = f"https://www.youtube.com/watch?v={video_id}"

    for f in AUDIO_DIR.glob(f"{video_id}.*"):
        if f.suffix.lower() in AUDIO_EXTS:
            print(f"[debug] reutilizando áudio: {f.name}")
            return f

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
        "-o", str(AUDIO_DIR / f"{video_id}%(autonumber)s.%(ext)s"),
        url,
    ]
    print("[debug] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    candidates = []
    for ext in AUDIO_EXTS:
        candidates += sorted(AUDIO_DIR.glob(f"{video_id}*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"Nenhum áudio encontrado para {video_id}")

    return candidates[0]

def load_existing_transcripts() -> pd.DataFrame:
    out_path = Path("outputs/transcripts.csv")
    if not out_path.exists():
        return pd.DataFrame(columns=["videoId", "language", "text"])
    try:
        old = pd.read_csv(out_path)
        for col in ("videoId", "language", "text"):
            if col not in old.columns:
                old[col] = None
        return old[["videoId", "language", "text"]]
    except Exception as e:
        print(f"[warn] não consegui ler transcripts.csv existente: {e}")
        return pd.DataFrame(columns=["videoId", "language", "text"])

def main() -> None:
    # Fonte de vídeos: prefere top50, senão usa o CSV bruto
    src_candidates = ["outputs/videos_top50.csv", "outputs/videos.csv"]
    src = next((p for p in src_candidates if Path(p).exists()), None)
    if src is None:
        print("[erro] Nenhum arquivo de vídeos encontrado. Rode os steps 1 e 2 primeiro.")
        sys.exit(1)

    print("[debug] source csv:", src)
    df = pd.read_csv(src)

    if "viewCount" in df.columns:
        df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce")
        df = df.sort_values("viewCount", ascending=False)

    top_n    = get_top_n()
    all_vids = df["videoId"].dropna().astype(str).unique().tolist()

    if not all_vids:
        print("[erro] Nenhum videoId encontrado no CSV.")
        sys.exit(1)

    vids = all_vids[:top_n]
    print(f"[debug] ASR_TOP_N: {top_n}")
    print(f"[debug] vídeos selecionados antes do filtro: {len(vids)}")

    old_df     = load_existing_transcripts()
    existing   = set(old_df["videoId"].dropna().astype(str).tolist())
    vids       = [v for v in vids if v not in existing]

    print(f"[debug] vídeos restantes após pular transcritos: {len(vids)}")

    if not vids:
        print("[info] Todos os vídeos selecionados já foram transcritos.")
        return

    model = whisper.load_model(MODEL_NAME)
    transcribe_kwargs = {"fp16": False} if USE_CPU else {}

    rows: list[dict] = []

    try:
        for vid in tqdm(vids, desc="Transcrevendo"):
            try:
                audio_file = ytdlp_best_audio(vid)
                result = model.transcribe(str(audio_file), **transcribe_kwargs)
                if result and result.get("text"):
                    rows.append({
                        "videoId":  vid,
                        "language": result.get("language"),
                        "text":     result.get("text"),
                    })
                else:
                    print(f"[warn] transcrição vazia: {vid}")

            except subprocess.CalledProcessError as e:
                print(f"[warn] download falhou para {vid}: {e}")
            except Exception as e:
                print(f"[warn] {vid} falhou: {repr(e)}")

    except KeyboardInterrupt:
        print("\n[info] Interrompido. Salvando parciais...")

    new_df = pd.DataFrame(rows)
    out    = pd.concat([old_df, new_df], ignore_index=True)

    if not out.empty:
        out = out.drop_duplicates(subset=["videoId"], keep="last")

    out.to_csv("outputs/transcripts.csv", index=False)
    print(f"Salvo transcripts.csv com {len(out)} linhas no total")


if __name__ == "__main__":
    main()