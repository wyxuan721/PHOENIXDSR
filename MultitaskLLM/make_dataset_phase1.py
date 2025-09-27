#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pypinyin import lazy_pinyin, Style

# Regex for basic CJK characters
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

def parse_args():
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description="Build dataset.json for LLaMA Factory (zh<->pinyin, TONE3).")
    ap.add_argument("--transcript", type=str, required=True, help="Path to AISHELL transcript file")
    ap.add_argument("--out", type=str, required=True, help="Output dataset.json path")
    ap.add_argument("--only", type=str, default="both", choices=["both", "zh2py", "py2zh"],
                    help="Task type: both / zh2py / py2zh")
    ap.add_argument("--min-chars", type=int, default=1, help="Min Chinese chars")
    ap.add_argument("--max-chars", type=int, default=200, help="Max Chinese chars")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by (utt_id, text)")
    return ap.parse_args()

def join_zh_tokens(tokenized: str) -> str:
    # Remove spaces and keep only Chinese characters
    s = "".join(tokenized.split())
    zh_only = "".join(ch for ch in s if CJK_RE.match(ch))
    return zh_only

def is_meaningful_zh(txt: str, min_chars: int) -> bool:
    # Check if string has at least min_chars Chinese characters
    return sum(1 for _ in CJK_RE.finditer(txt) for __ in _[0]) >= min_chars

def text_to_initial_final_tone3(txt: str) -> str:
    # Convert Chinese to phoneme sequence: initials + finals with tone (TONE3)
    from pypinyin import pinyin, Style
    zh = "".join(ch for ch in txt if "\u4e00" <= ch <= "\u9fff")
    if not zh:
        return ""
    initials = pinyin(zh, style=Style.INITIALS, strict=False, neutral_tone_with_five=True)
    finals_t3 = pinyin(zh, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True)
    phs, last_final_idx = [], -1
    for i, ch in enumerate(zh):
        ini = initials[i][0] if initials[i] else ""
        fin = finals_t3[i][0] if finals_t3[i] else ""
        if ch == "儿":
            if last_final_idx >= 0 and phs[last_final_idx]:
                prev = phs[last_final_idx]
                phs[last_final_idx] = (prev[:-1] + "r" + prev[-1]) if prev[-1].isdigit() else (prev + "r")
            continue
        if ini:
            phs.append(ini)
        if fin:
            phs.append(fin)
            last_final_idx = len(phs) - 1
    return " ".join(t for t in phs if t.strip())

def read_transcript(fp: Path) -> List[Tuple[str, str]]:
    # Read transcript file and return list of (utt_id, text)
    rows: List[Tuple[str, str]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith('"'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, tokenized = parts[0].strip(), parts[1].strip()
            if not utt_id or not tokenized:
                continue
            zh = join_zh_tokens(tokenized)
            if not zh:
                continue
            rows.append((utt_id, zh))
    return rows

def build_item_zh2py(zh: str) -> Dict:
    # Build task: Chinese -> Pinyin
    py = text_to_initial_final_tone3(zh)
    return {
        "instruction": "请把下面的中文句子转为拼音。",
        "input": zh,
        "output": py,
        "system": "你是一个中文转拼音助手。\n"
    }

def build_item_py2zh(zh: str) -> Dict:
    # Build task: Pinyin -> Chinese
    py = text_to_initial_final_tone3(zh)
    return {
        "instruction": "请把给定的拼音，还原为规范的中文文本。",
        "input": py,
        "output": zh,
        "system": "你是一个中文拼音还原助手。\n"
    }

def main():
    # Main pipeline
    args = parse_args()
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        raise FileNotFoundError(f"transcript not found: {transcript_path}")
    pairs = read_transcript(transcript_path)
    seen = set()
    items: List[Dict] = []
    for utt_id, zh in pairs:
        if args.dedup and (utt_id, zh) in seen:
            continue
        if not is_meaningful_zh(zh, args.min_chars):
            continue
        if len(zh) > args.max_chars:
            continue
        if args.only in ("both", "zh2py"):
            items.append(build_item_zh2py(zh))
        if args.only in ("both", "py2zh"):
            items.append(build_item_py2zh(zh))
        seen.add((utt_id, zh))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote dataset: {out_path.resolve()} (samples: {len(items)})")
    n_zh2py = sum(1 for it in items if it["instruction"].startswith("请把下面的中文句子转为拼音"))
    n_py2zh = sum(1 for it in items if it["instruction"].startswith("请把给定的拼音"))
    print("  Task stats:", {"zh2py": n_zh2py, "py2zh": n_py2zh})

if __name__ == "__main__":
    main()
