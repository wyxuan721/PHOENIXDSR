# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Config
SAMPLE_RATE = 16000
AUDIO_ROOT = Path(r"")
OUTPUT_DIR = Path(r"")
FINAL_DIR = OUTPUT_DIR / "final_model"  # best model

def id_to_wav_path(utt_id_no_ext: str, audio_root: Path) -> Path:
    base = utt_id_no_ext
    if base.endswith(".wav"):
        base = base[:-4]
    subdir = base[:7] if len(base) >= 7 else base
    return audio_root / subdir / f"{base}.wav"

def load_audio_resample(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)

def find_latest_checkpoint(ckpt_root: Path) -> Path:
    # find latest checkpoint by step number
    cands = []
    if ckpt_root.is_dir():
        for p in ckpt_root.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    step = int(p.name.split("-")[-1])
                    cands.append((step, p))
                except:
                    pass
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def load_model_and_processor(use_latest_ckpt=False):
    # load final model or latest checkpoint
    if use_latest_ckpt:
        latest = find_latest_checkpoint(OUTPUT_DIR / "ckpts")
        if latest is not None:
            print(f"[INFO] Loading from latest checkpoint: {latest}")
            proc = Wav2Vec2Processor.from_pretrained(str(latest / "processor")) \
                   if (latest / "processor").exists() else Wav2Vec2Processor.from_pretrained(str(FINAL_DIR / "processor"))
            model = Wav2Vec2ForCTC.from_pretrained(str(latest))
            return model, proc

    print(f"[INFO] Loading from final model: {FINAL_DIR}")
    proc = Wav2Vec2Processor.from_pretrained(str(FINAL_DIR / "processor"))
    model = Wav2Vec2ForCTC.from_pretrained(str(FINAL_DIR))
    return model, proc

def infer(wav_path: Path, device: str = None, group_tokens=False):
    model, processor = load_model_and_processor(use_latest_ckpt=False)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    wav = load_audio_resample(str(wav_path), SAMPLE_RATE)
    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)

    # group_tokens=False keeps raw token sequence
    text = processor.batch_decode(pred_ids, group_tokens=group_tokens)[0]
    return text

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", type=str, help="wav file path")
    g.add_argument("--utt-id", type=str, help="utt id like SSB00050001")
    parser.add_argument("--show-steps", action="store_true", help="show steps when loading latest checkpoint")
    parser.add_argument("--group-tokens", action="store_true", help="merge repeated tokens when decoding")
    args = parser.parse_args()

    if args.utt_id:
        wav_path = id_to_wav_path(args.utt_id, AUDIO_ROOT)
    else:
        wav_path = Path(args.wav)

    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    result = infer(wav_path, group_tokens=False if not args.group_tokens else True)

    print("======================================")
    print("Audio:", wav_path)
    print("Decoded (phoneme sequence):")
    print(result)
    print("======================================")

if __name__ == "__main__":
    main()
