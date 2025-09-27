import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
from dataclasses import dataclass

from pypinyin import pinyin, Style
from datasets import Dataset, DatasetDict

import evaluate
import torch
from torch.utils.data import default_collate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm

# ======================
# Configuration
# ======================
LABEL_FILE = ""
AUDIO_ROOT = Path("")
OUTPUT_DIR = Path("")
MODEL_NAME = ""

SAMPLE_RATE = 16000
DEV_RATIO = 0.05          # validation ratio
KEEP_MARKERS = True
SEED = 3407

USE_WANDB = True
WANDB_PROJECT = "aishell3-phoneme-ctc"
WANDB_ENTITY = None
WANDB_RUN_NAME = "w2v2-phoneme-ctc-1"

MARKERS = {"%", "$"}

# ======================
# Text to phoneme conversion
# ======================
def _hanzi_and_markers(s: str) -> List[str]:
    """Keep only Chinese characters and markers (%/$)."""
    out = []
    for ch in s:
        if ch in MARKERS:
            out.append(ch)
        elif '\u4e00' <= ch <= '\u9fff':
            out.append(ch)
    return out

def extract_phonemes_hanzi_with_markers(text_with_markers: str) -> List[str]:
    """Convert Chinese text (with %/$ markers) into phoneme sequence."""
    tokens = _hanzi_and_markers(text_with_markers)
    hanzi_only = "".join(t for t in tokens if t not in MARKERS)

    if len(hanzi_only) == 0:
        return [tk for tk in tokens if tk in MARKERS]

    initials = pinyin(hanzi_only, style=Style.INITIALS, strict=False,
                      neutral_tone_with_five=True)
    finals_t3 = pinyin(hanzi_only, style=Style.FINALS_TONE3, strict=False,
                       neutral_tone_with_five=True)

    phonemes: List[str] = []
    last_final_idx: int = -1
    i = 0

    for tk in tokens:
        if tk in MARKERS:
            phonemes.append(tk)
            continue

        ch = tk
        ini = initials[i][0]
        fin = finals_t3[i][0]
        i += 1

        # Merge 儿 into the previous final
        if ch == "儿":
            if last_final_idx >= 0:
                prev_fin = phonemes[last_final_idx]
                if prev_fin and prev_fin[-1].isdigit():
                    phonemes[last_final_idx] = prev_fin[:-1] + "r" + prev_fin[-1]
                else:
                    phonemes[last_final_idx] = prev_fin + "r"
            continue

        if ini:
            phonemes.append(ini)
        if fin:
            phonemes.append(fin)
            last_final_idx = len(phonemes) - 1

    return phonemes

def parse_and_convert_line(line: str) -> Optional[Tuple[str, List[str]]]:
    """Parse a line from label file and return (utt_id, phoneme tokens)."""
    line = line.strip()
    if not line:
        return None

    # Pipe-separated format
    if '|' in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            return None
        utt_id = parts[0]
        hanzi_marked = parts[2]
        phs = extract_phonemes_hanzi_with_markers(hanzi_marked)
        if not KEEP_MARKERS:
            phs = [t for t in phs if t not in MARKERS]
        return utt_id.replace(".wav", ""), phs

    # TSV format
    if '\t' in line:
        utt_id, label = line.split('\t', 1)
        hanzi_only = ''.join(ch for ch in label if '\u4e00' <= ch <= '\u9fff')
        phs = extract_phonemes_hanzi_with_markers(hanzi_only) if hanzi_only else []
        if not KEEP_MARKERS:
            phs = [t for t in phs if t not in MARKERS]
        return utt_id.replace(".wav", ""), phs

    return None

def id_to_wav_path(utt_id_no_ext: str, audio_root: Path) -> Path:
    """Convert utterance ID to wav file path (AISHELL-3 style)."""
    base = utt_id_no_ext
    if base.endswith(".wav"):
        base = base[:-4]
    subdir = base[:7] if len(base) >= 7 else base
    return audio_root / subdir / f"{base}.wav"

# ======================
# Dataset building
# ======================
def load_items(label_file: str, audio_root: Path):
    """Load items from label file and map to phoneme + wav path."""
    items = []
    missing = 0
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Parsing labels"):
        parsed = parse_and_convert_line(line)
        if not parsed:
            continue
        utt_id, phs = parsed
        wav_path = id_to_wav_path(utt_id, audio_root)
        if not wav_path.is_file():
            missing += 1
            continue
        items.append({
            "utt_id": utt_id,
            "path": str(wav_path),
            "phonemes": phs
        })
    print(f"[INFO] Loaded items: {len(items)}, missing audio: {missing}")
    return items

def build_vocab(items: List[dict], extra_tokens: List[str] = None):
    """Build vocabulary from phoneme tokens."""
    vocab = {}
    idx = 0
    specials = ["[PAD]", "[UNK]"]
    for sp in specials:
        vocab[sp] = idx; idx += 1

    for it in tqdm(items, desc="Building vocab"):
        for t in it["phonemes"]:
            if t not in vocab:
                vocab[t] = idx
                idx += 1

    if extra_tokens:
        for t in extra_tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab

def save_processor(vocab: dict, out_dir: Path):
    """Save tokenizer and feature extractor as processor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        word_delimiter_token=" ",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    processor.save_pretrained(str(out_dir / "processor"))
    return processor

def load_audio_resample(path: str, target_sr: int) -> np.ndarray:
    """Load and resample audio to target sample rate."""
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)

def make_hf_dataset(items: List[dict]) -> Dataset:
    """Convert items to HuggingFace Dataset."""
    return Dataset.from_list(items)

def split_dataset(ds: Dataset, dev_ratio: float, seed: int) -> DatasetDict:
    """Split dataset into train and validation sets."""
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_dev = max(1, int(n * dev_ratio))
    dev_idx = set(idxs[:n_dev])
    ds_train = ds.select([i for i in range(n) if i not in dev_idx])
    ds_dev = ds.select([i for i in range(n) if i in dev_idx])
    return DatasetDict(train=ds_train, validation=ds_dev)

def prepare_batch(batch, processor: Wav2Vec2Processor):
    """Prepare batch: extract features and labels."""
    wav = load_audio_resample(batch["path"], SAMPLE_RATE)
    batch["input_values"] = processor(wav, sampling_rate=SAMPLE_RATE).input_values[0]
    target = " ".join(batch["phonemes"])
    with processor.as_target_processor():
        batch["labels"] = processor(target).input_ids
    return batch

@dataclass
class DataCollatorCTC:
    """Custom collator for CTC training."""
    processor: Wav2Vec2Processor

    def __call__(self, features):
        input_vals = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(
            input_vals,
            padding=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                labels,
                padding=True,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# ======================
# Training pipeline
# ======================
def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup wandb logging
    if USE_WANDB:
        try:
            import wandb
            os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
            if WANDB_ENTITY:
                os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
            os.environ.setdefault("WANDB_RUN_GROUP", "w2v2-phoneme")
            os.environ.setdefault("WANDB_DIR", str(OUTPUT_DIR.resolve()))
            report_to = ["wandb"]
        except Exception as e:
            print(f"[WARN] wandb not available, fallback to local log: {e}")
            report_to = ["none"]
    else:
        report_to = ["none"]

    # Load dataset items
    items = load_items(LABEL_FILE, AUDIO_ROOT)
    if len(items) == 0:
        raise RuntimeError("No samples found. Check labels or audio path.")

    # Build vocab & processor
    vocab = build_vocab(items)
    processor = save_processor(vocab, OUTPUT_DIR)

    # Build HF dataset
    ds = make_hf_dataset(items)
    ds = ds.map(
        lambda b: prepare_batch(b, processor),
        num_proc=1,
        desc="Preparing dataset"
    )
    dsd = split_dataset(ds, DEV_RATIO, SEED)

    data_collator = DataCollatorCTC(processor=processor)

    per_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids, group_tokens=False)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        per = per_metric.compute(predictions=pred_str, references=label_str)
        return {"per": per}

    print(f"[INFO] Vocab size: {len(vocab)}. Example tokens: {list(vocab.keys())[:30]}")

    # Load model
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        vocab_size=len(processor.tokenizer.get_vocab())
    )
    model.freeze_feature_encoder()

    # Training arguments
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "ckpts"),
        run_name=WANDB_RUN_NAME if USE_WANDB else None,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=0,
        group_by_length=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to=report_to,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model
    final_dir = OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir / "processor"))
    print("[DONE] Training complete. Model saved to:", final_dir)

def debug_smoke_test(n=10):
    """Quick debug test for parsing & phoneme extraction."""
    print("\n[DEBUG] === Smoke test on parsing/phonemes ===")
    items = load_items(LABEL_FILE, AUDIO_ROOT)
    print(f"[DEBUG] items={len(items)} (showing first {n})")
    for it in items[:n]:
        utt = it["utt_id"]
        phs = it["phonemes"]
        wav = it["path"]
        print(f"- {utt}  wav? {'OK' if os.path.isfile(wav) else 'MISSING'}")
        print(f"  phonemes({len(phs)}): {' '.join(phs)}")
    print("[DEBUG] === End ===\n")

if __name__ == "__main__":
    debug_smoke_test()
