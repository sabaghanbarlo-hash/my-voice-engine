"""Read a script aloud in a cloned voice with Voice Studio (OmniVoice engine). Free, CPU only."""
import argparse, hashlib, re, subprocess, sys, wave
from pathlib import Path
import numpy as np

MODEL = "k2-fsa/OmniVoice"
SR = 24000


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def to_wav(src, dst):
    run(["ffmpeg", "-y", "-i", str(src), "-ar", str(SR), "-ac", "1", str(dst)])


def read_wav(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def write_wav(p, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def silence(sec):
    return np.zeros(int(sec * SR), np.float32)


def trim(a, keep=0.04, thr=0.012):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] - int(keep * SR))
    e = min(len(a), idx[-1] + int(keep * SR))
    return a[s:e]


def file_md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def clean(text):
    """Drop markdown headings, [cues] and divider lines; keep paragraphs."""
    paras, cur = [], []
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if not s or re.fullmatch(r"-{3,}", s):
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        if re.match(r"^#+\s", s):
            if cur:
                paras.append(" ".join(cur)); cur = []
            s = re.sub(r"^#+\s*", "", s)
            if s:
                paras.append(s)
            continue
        s = re.sub(r"\[[^\]]*\]", "", s).strip()
        if s:
            cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return paras


def sentences(par, limit=300):
    out = []
    for s in re.split(r"(?<=[.!?\u061f\u2026])\s+", par.strip()):
        s = s.strip()
        if not s:
            continue
        while len(s) > limit:
            cut = max(s.rfind(", ", 0, limit), s.rfind("; ", 0, limit), s.rfind(" ", 0, limit))
            cut = cut if cut > 40 else limit
            out.append(s[:cut + 1].strip()); s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


_VS = {}


def load_model(ref):
    import torch
    from omnivoice import OmniVoice
    if "m" not in _VS:
        print("loading the Voice Studio model (first time downloads it)...", flush=True)
        _VS["m"] = OmniVoice.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32)
    m = _VS["m"]
    if ref and _VS.get("ref") != str(ref):
        txt = Path(str(ref) + ".txt")
        if not txt.exists():
            txt = Path(ref).with_suffix(".txt")
        rt = txt.read_text().strip() if txt.exists() else None
        if rt is None:
            m.load_asr_model("openai/whisper-small")  # no transcript of the sample: read it ourselves
        _VS["prompt"] = m.create_voice_clone_prompt(ref_audio=str(ref), ref_text=rt)
        _VS["ref"] = str(ref)
    return m


def speak(sent, ref, cache, speed, steps, lang):
    rid = file_md5(ref)[:10] if ref else "design"
    key = hashlib.md5(f"{rid}|{speed}|{steps}|{lang}|{sent}".encode()).hexdigest()[:16]
    f = cache / f"{key}.wav"
    if not f.exists():
        import torchaudio
        m = load_model(ref)
        kw = dict(text=sent, language=lang, num_step=steps, speed=speed)
        if ref:
            kw["voice_clone_prompt"] = _VS["prompt"]
        else:
            kw["instruct"] = "female, young adult, warm, expressive"
        wav = m.generate(**kw)[0]
        raw = cache / f"{key}_raw.wav"
        torchaudio.save(str(raw), wav.detach().cpu().reshape(1, -1), m.sampling_rate)
        to_wav(raw, f)
        raw.unlink(missing_ok=True)
    return trim(read_wav(f), keep=0.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--ref", default=None, help="10-30 s sample of the voice to clone (wav)")
    ap.add_argument("--out", default="output/voice.mp3")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--language", default="en")
    ap.add_argument("--work", default=".work")
    a = ap.parse_args()

    paras = clean(Path(a.script).read_text(encoding="utf-8"))
    if not paras:
        sys.exit("The script is empty.")
    if a.ref and not Path(a.ref).exists():
        sys.exit(f"Voice sample not found: {a.ref}")
    cache = Path(a.work) / "vs_cache"
    cache.mkdir(parents=True, exist_ok=True)
    total = sum(len(sentences(p)) for p in paras)
    print(f"{len(paras)} paragraphs, {total} sentences", flush=True)

    pieces, n = [], 0
    for pi, par in enumerate(paras):
        for s in sentences(par):
            n += 1
            print(f"[{n}/{total}] {s[:70]}", flush=True)
            pieces.append(speak(s, a.ref, cache, a.speed, a.steps, a.language))
            pieces.append(silence(0.16))
        if pieces:
            pieces[-1] = silence(0.5 if pi < len(paras) - 1 else 0.2)
    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak * 0.92

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".wav")
    write_wav(tmp, audio)
    run(["ffmpeg", "-y", "-i", str(tmp), "-c:a", "libmp3lame", "-q:a", "3", str(out)])
    tmp.unlink(missing_ok=True)
    print(f"done: {out} ({len(audio) / SR:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
