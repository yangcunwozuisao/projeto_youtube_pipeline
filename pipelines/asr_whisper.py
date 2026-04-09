import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

print("[debug] python:", sys.executable)

try:
    import imageio_ffmpeg as iioff

    ff_src = Path(iioff.get_ffmpeg_exe())
    print("[debug] ffmpeg real exe (imageio):", ff_src)

    WRAP_DIR = Path(tempfile.mkdtemp(prefix="ffmpeg_wrap_"))
    ff_dst = WRAP_DIR / "ffmpeg.exe"
    shutil.copyfile(ff_src, ff_dst)

    fp_dst = WRAP_DIR / "ffprobe.exe"
    shutil.copyfile(ff_src, fp_dst)

    os.environ["PATH"] = str(WRAP_DIR) + os.pathsep + os.environ.get("PATH", "")
    print("[debug] wrap dir:", WRAP_DIR)

    subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

except Exception as e:
    print("[erro] não consegui preparar ffmpeg.exe:", e)
    print("       Rode: .\\.venv\\Scripts\\python.exe -m pip install imageio-ffmpeg")
    sys.exit(1)

try:
    import yt_dlp
    print("[debug] yt_dlp version:", getattr(yt_dlp, "__version__", "unknown"))
except Exception as e:
    print("[erro] yt-dlp não encontrado neste Python:", e)
    print("       Rode: .\\.venv\\Scripts\\python.exe -m pip install yt-dlp")
    sys.exit(1)

import whisper

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 更快：small -> base；如果还慢，可以改成 "tiny"
MODEL_NAME = "base"
USE_CPU = True

# 默认只处理前 20 个；想更多可以改
DEFAULT_TOP_N = 20


def get_top_n():
    """
    Lê a quantidade de vídeos a transcrever a partir da variável de ambiente ASR_TOP_N.

    Regras:
    - se não existir -> usa DEFAULT_TOP_N
    - se <= 0 -> usa DEFAULT_TOP_N
    - se inválido -> erro
    """
    raw = os.getenv("ASR_TOP_N")

    if raw is None or raw == "":
        return DEFAULT_TOP_N

    try:
        value = int(raw)
    except ValueError:
        raise ValueError("ASR_TOP_N deve ser um número inteiro.")

    if value <= 0:
        return DEFAULT_TOP_N

    return value


def ytdlp_best_audio(video_id: str) -> Path:
    """Baixa o melhor áudio usando este Python."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    for f in AUDIO_DIR.glob(f"{video_id}.*"):
        if f.suffix.lower() in [".webm", ".m4a", ".opus", ".mp3", ".wav", ".flac"]:
            print(f"[debug] reuse audio: {f.name}")
            return f

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
        "-o", str(AUDIO_DIR / f"{video_id}%(autonumber)s.%(ext)s"),
        url,
    ]

    print("[debug] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    candidates = (
        sorted(AUDIO_DIR.glob(f"{video_id}*.m4a")) +
        sorted(AUDIO_DIR.glob(f"{video_id}*.webm")) +
        sorted(AUDIO_DIR.glob(f"{video_id}*.opus")) +
        sorted(AUDIO_DIR.glob(f"{video_id}*.mp3")) +
        sorted(AUDIO_DIR.glob(f"{video_id}*.wav")) +
        sorted(AUDIO_DIR.glob(f"{video_id}*.flac"))
    )

    if not candidates:
        raise FileNotFoundError(f"Nenhum áudio encontrado para {video_id}")

    return candidates[0]


def load_existing_transcripts():
    out_path = Path("outputs/transcripts.csv")
    if not out_path.exists():
        return pd.DataFrame(columns=["videoId", "language", "text"])

    try:
        old = pd.read_csv(out_path)
        required = {"videoId", "language", "text"}
        for col in required:
            if col not in old.columns:
                old[col] = None
        return old[["videoId", "language", "text"]]
    except Exception as e:
        print(f"[warn] não consegui ler transcripts.csv existente: {e}")
        return pd.DataFrame(columns=["videoId", "language", "text"])


def main():
    src = "outputs/videos_top50.csv" if Path("outputs/videos_top50.csv").exists() else "outputs/videos.csv"
    print("[debug] source csv:", src)

    df = pd.read_csv(src)

    if "viewCount" in df.columns:
        df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce")
        df = df.sort_values("viewCount", ascending=False)

    top_n = get_top_n()

    all_vids = df["videoId"].dropna().astype(str).unique().tolist()

    if not all_vids:
        print("[erro] Nenhum videoId encontrado no CSV.")
        sys.exit(1)

    vids = all_vids[:top_n]
    print(f"[debug] ASR_TOP_N: {top_n}")
    print(f"[debug] vídeos selecionados antes do filtro: {len(vids)}")

    # 跳过已经转录过的
    old_df = load_existing_transcripts()
    existing_ids = set(old_df["videoId"].dropna().astype(str).tolist())

    vids = [v for v in vids if v not in existing_ids]

    print(f"[debug] vídeos restantes após pular já transcritos: {len(vids)}")
    print("[debug] vids:", vids[:10], "..." if len(vids) > 10 else "")

    if not vids:
        print("[info] Todos os vídeos selecionados já foram transcritos.")
        return

    model = whisper.load_model(MODEL_NAME)
    transcribe_kwargs = {"fp16": False} if USE_CPU else {}

    rows = []

    try:
        for vid in tqdm(vids, desc="Transcrevendo"):
            try:
                audio_file = ytdlp_best_audio(vid)
                result = whisper.transcribe(model, str(audio_file), **transcribe_kwargs)

                if result and result.get("text"):
                    rows.append({
                        "videoId": vid,
                        "language": result.get("language"),
                        "text": result.get("text")
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
    out = pd.concat([old_df, new_df], ignore_index=True)

    if not out.empty:
        out = out.drop_duplicates(subset=["videoId"], keep="last")

    out.to_csv("outputs/transcripts.csv", index=False)

    print(f"Salvo transcripts.csv com {len(out)} linhas no total")


if __name__ == "__main__":
    main()