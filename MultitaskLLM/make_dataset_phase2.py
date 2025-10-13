from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import re
from typing import Set, Optional, Dict, List

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# --- Constants & basic helpers ---
MARKERS = {"%", "$"}
UNK_TOK = "[UNK]"
EPS_DEL = "<DEL>"

INITIALS = {"b","p","m","f","d","t","n","l","g","k","h",
            "j","q","x","zh","ch","sh","r","z","c","s","y","w"}

UTT_PREFIX_RE = re.compile(r"^S(\d{3})")
UTT_PID_RE = re.compile(r"^[Ss](\d{3})")

def strip_tone(tok: str) -> str:
    return re.sub(r"([a-zA-Züv]+)[1-5]$", r"\1", tok or "")

def has_tone(tok: str) -> bool:
    return bool(re.search(r"[1-5]$", tok or ""))

def is_initial(tok: Optional[str]) -> bool:
    if tok is None: return False
    if tok in MARKERS or tok == UNK_TOK: return False
    return strip_tone(tok) in INITIALS

def is_final(tok: Optional[str]) -> bool:
    if tok is None: return False
    if tok in MARKERS or tok == UNK_TOK: return False
    return strip_tone(tok) not in INITIALS

def de_erhua_token(tok: str) -> str:
    if not tok or not is_final(tok):
        return tok
    base_no_tone = strip_tone(tok)
    if base_no_tone == "er":
        return tok
    m = re.match(r"^([a-züv]+)r([1-5])$", tok)
    if m:
        if m.group(1) == "e":
            return tok
        return m.group(1) + m.group(2)
    if base_no_tone.endswith("r") and base_no_tone != "er":
        return tok[:-1]
    return tok

def norm_seq(seq_str: str, drop_markers=True, drop_tone=False, ignore_erhua=False) -> List[str]:
    toks: List[str] = []
    for t in (seq_str or "").strip().split():
        if drop_markers and t in (MARKERS | {UNK_TOK}):
            continue
        x = t
        if ignore_erhua:
            x = de_erhua_token(x)
        if drop_tone:
            x = strip_tone(x)
        x = x.strip()
        if x:
            toks.append(x)
    return toks

def same_final_base(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None: return False
    return is_final(a) and is_final(b) and strip_tone(a) == strip_tone(b)

# --- Phoneme alignment (weighted edit distance) ---
def weighted_align_phoneme(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str], str]]:
    n, m = len(ref), len(hyp)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i*1.0; bt[i][0] = 'D'
    for j in range(1, m+1): dp[0][j] = j*1.0; bt[0][j] = 'I'
    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            hj = hyp[j-1]
            if ri == hj:
                subs_cost = 0.0
            elif same_final_base(ri, hj):
                subs_cost = 0.2
            else:
                if (is_initial(ri) and is_initial(hj)) or (is_final(ri) and is_final(hj)):
                    subs_cost = 1.0
                else:
                    subs_cost = 1.5
            subs = dp[i-1][j-1] + subs_cost
            dele = dp[i-1][j] + 1.0
            ins  = dp[i][j-1] + 1.0
            best = min(subs, dele, ins)
            dp[i][j] = best
            bt[i][j] = 'M' if (best==subs and subs_cost==0.0) else ('S' if best==subs else 'D' if best==dele else 'I')
    i, j = n, m
    out: List[Tuple[Optional[str], Optional[str], str]] = []
    while i>0 or j>0:
        op = bt[i][j]
        if op in ('M','S'):
            out.append((ref[i-1], hyp[j-1], op)); i-=1; j-=1
        elif op=='D':
            out.append((ref[i-1], None, 'D')); i-=1
        else:
            out.append((None, hyp[j-1], 'I')); j-=1
    out.reverse()
    return out

# --- Stage 1: build pairs.jsonl from text/audio ---
def _hanzi_and_markers(s: str) -> List[str]:
    return [ch for ch in s if (ch in MARKERS) or ('\u4e00' <= ch <= '\u9fff')]

def extract_phonemes(text: str, keep_markers: bool=True) -> List[str]:
    try:
        from pypinyin import pinyin, Style
    except Exception as e:
        raise RuntimeError("需要安装 pypinyin 才能运行 --stage pairs：pip install pypinyin") from e

    tokens = _hanzi_and_markers(text)
    hanzi_only = "".join(t for t in tokens if t not in MARKERS)
    if not hanzi_only:
        return [tk for tk in tokens if tk in MARKERS] if keep_markers else []
    initials = pinyin(hanzi_only, style=Style.INITIALS, strict=False, neutral_tone_with_five=True)
    finals_t3 = pinyin(hanzi_only, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True)
    phs: List[str] = []
    last_final_idx = -1
    i = 0
    for tk in tokens:
        if tk in MARKERS:
            if keep_markers: phs.append(tk)
            continue
        ch = tk
        ini = initials[i][0]
        fin = finals_t3[i][0]
        i += 1
        if ch == "儿":
            if last_final_idx >= 0:
                prev = phs[last_final_idx]
                phs[last_final_idx] = (prev[:-1]+"r"+prev[-1]) if prev and prev[-1].isdigit() else (prev+"r")
            continue
        if ini: phs.append(ini)
        if fin:
            phs.append(fin)
            last_final_idx = len(phs)-1
    return phs

def load_audio_resample(path: Path, target_sr: int):
    try:
        import soundfile as sf, numpy as np, librosa
    except Exception as e:
        raise RuntimeError("需要安装 soundfile/numpy/librosa 才能运行 --stage pairs") from e
    wav, sr = sf.read(str(path))
    if getattr(wav, 'ndim', 1) > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav.astype('float32'), orig_sr=sr, target_sr=target_sr)
    return wav.astype('float32')

def infer_ctc(wav_path: Path, model, proc, sample_rate: int, group_tokens=False) -> str:
    import torch
    wav = load_audio_resample(wav_path, sample_rate)
    ins = proc(wav, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    dev = next(model.parameters()).device
    input_values = ins.input_values.to(dev)
    attention_mask = ins.attention_mask.to(dev) if "attention_mask" in ins else None
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        pred_ids = logits.argmax(dim=-1)[0].tolist()
    blank_id = proc.tokenizer.pad_token_id
    tokens: List[str] = []
    prev = None
    for i in pred_ids:
        if i == blank_id:
            prev = i; continue
        if group_tokens and i == prev:
            continue
        tokens.append(proc.tokenizer.convert_ids_to_tokens(i))
        prev = i
    wd = proc.tokenizer.word_delimiter_token
    tokens = [t for t in tokens if t and t != wd]
    return " ".join(tokens)

def stage_pairs(text_dir: Path, audio_root: Path, final_model: Path, outdir: Path,
                drop_markers: bool, keep_tone: bool, sample_rate: int = 16000,
                save_pairs_path: Optional[Path] = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pairs_path = save_pairs_path or (outdir / "pairs.jsonl")

    def parse_label_file(fpath: Path) -> List[Tuple[str, str, str]]:
        subdir = fpath.stem.split('_')[0]
        items = []
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('"'):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    items.append((utt_id, text, subdir))
        return items

    items: List[Tuple[str, str, str]] = []
    for f in sorted(text_dir.glob("*_label.txt")):
        items.extend(parse_label_file(f))

    print(f"[Stage:pairs] label lines: {len(items)}")

    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch
    except Exception as e:
        raise RuntimeError("需要安装 transformers/torch 才能运行 --stage pairs") from e

    proc = Wav2Vec2Processor.from_pretrained(str(final_model / "processor"))
    model = Wav2Vec2ForCTC.from_pretrained(str(final_model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()

    def audio_path(subdir: str, utt_id: str) -> Path:
        return audio_root / subdir / f"{utt_id}.wav"

    n_miss = 0
    with pairs_path.open("w", encoding="utf-8") as jf:
        for utt_id, text, subdir in tqdm(items, desc="[pairs] infer", unit="utt"):
            wav = audio_path(subdir, utt_id)
            if not wav.is_file():
                n_miss += 1
                continue
            ref_tokens = extract_phonemes(text, keep_markers=not drop_markers)
            if drop_markers:
                ref_tokens = [t for t in ref_tokens if t not in MARKERS]
            if not keep_tone:
                ref_tokens = [strip_tone(t) for t in ref_tokens]
            obs_raw = infer_ctc(wav, model, proc, sample_rate)
            obs_tokens = norm_seq(obs_raw, drop_markers=drop_markers, drop_tone=(not keep_tone), ignore_erhua=False)

            jf.write(json.dumps({
                "utt_id": utt_id,
                "text": text,
                "ref": " ".join(ref_tokens),
                "obs": " ".join(obs_tokens)
            }, ensure_ascii=False) + "\n")

    print(f"[Stage:pairs] wrote: {pairs_path.resolve()} (missing audio: {n_miss})")
    return pairs_path

# --- Stage 2: compute channel_global.json from pairs ---
def parse_patient_from_utt(utt_id: Optional[str]) -> Optional[str]:
    if not utt_id: return None
    m = UTT_PID_RE.match(utt_id)
    return m.group(1) if m else None

def build_backoff_probs(global_counts: Dict[str, Counter], alpha_backoff: float) -> Tuple[Dict[str,float], Dict[str,float], Dict[str,float]]:
    agg_all = Counter(); agg_init = Counter(); agg_final = Counter()
    for t, ctr in global_counts.items():
        (agg_init if is_initial(t) else agg_final).update(ctr)
        agg_all.update(ctr)

    def norm_ctr(ctr: Counter) -> Dict[str, float]:
        total = sum(ctr.values()); keys = list(ctr.keys()); K = len(keys)
        if total == 0 or K == 0: return {}
        denom = total + alpha_backoff * K
        return {k: (ctr[k] + alpha_backoff) / denom for k in keys}

    return norm_ctr(agg_all), norm_ctr(agg_init), norm_ctr(agg_final)

def export_channel_with_shrinkage(pairs_path: Path, outdir: Path,
                                  drop_markers: bool, keep_tone: bool, ignore_erhua: bool,
                                  alpha: float, beta: float, backoff: str,
                                  backoff_add_k: int, alpha_backoff: float,
                                  topk_csv: int = 10) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    global_counts: Dict[str, Counter] = defaultdict(Counter)
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[analyze] align&count", unit="utt"):
            if not line.strip():
                continue
            obj = json.loads(line)
            ref = norm_seq(obj["ref"], drop_markers=drop_markers, drop_tone=(not keep_tone), ignore_erhua=ignore_erhua)
            obs = norm_seq(obj["obs"], drop_markers=drop_markers, drop_tone=(not keep_tone), ignore_erhua=ignore_erhua)
            al = weighted_align_phoneme(ref, obs)
            for t,o,op in al:
                if op in ('M','S'):
                    global_counts[t][o] += 1
                elif op=='D':
                    global_counts[t][EPS_DEL] += 1

    bo_all, bo_init, bo_final = build_backoff_probs(global_counts, alpha_backoff)

    def get_bo(t: str) -> Dict[str,float]:
        if backoff == "global":
            return bo_all
        return bo_init if is_initial(t) else bo_final

    channel_map: Dict[str, Dict[str, float]] = {}

    def topk_from(prob_map: Dict[str,float], k: int) -> List[str]:
        return [k_ for k_, _ in sorted(prob_map.items(), key=lambda x: -x[1])[:k]]

    topk_all = topk_from(bo_all, backoff_add_k)
    topk_init = topk_from(bo_init, backoff_add_k)
    topk_final = topk_from(bo_final, backoff_add_k)

    rows_csv: List[Tuple[str,str,str]] = []

    for t, ctr in global_counts.items():
        n = sum(ctr.values())
        rho = n / (n + beta) if n > 0 else 0.0
        keys = set(ctr.keys()); keys.add(t); keys.add(EPS_DEL)
        bo = get_bo(t)
        if backoff == "global":
            keys.update(topk_all)
        else:
            keys.update(topk_init if is_initial(t) else topk_final)
        mle = {}
        if n > 0:
            denom = n + (alpha * len(ctr) if alpha>0 else 0.0)
            for o in ctr:
                mle[o] = (ctr[o] + (alpha if alpha>0 else 0.0)) / (denom if denom>0 else 1.0)
        s_bo = sum(bo.get(o, 0.0) for o in keys)
        bo_subset = {o: (bo.get(o, 0.0) / s_bo) if s_bo>0 else (1.0/len(keys)) for o in keys}
        probs = {o: rho*mle.get(o,0.0) + (1.0-rho)*bo_subset.get(o,0.0) for o in keys}
        s = sum(probs.values())
        if s>0:
            for o in list(probs.keys()):
                probs[o] /= s
        channel_map[t] = dict(sorted(probs.items(), key=lambda x: -x[1]))
        for o,p in list(channel_map[t].items())[:topk_csv]:
            rows_csv.append((t,o,f"{p:.6f}"))

    channel_path = outdir / "channel_global.json"
    channel_path.write_text(json.dumps(channel_map, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = outdir / "channel_global.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("true,obs,prob\n")
        for r in rows_csv:
            f.write(",".join(r)+"\n")
    print(f"[Stage:analyze] wrote: {channel_path.resolve()} (and channel_global.csv)")
    return channel_path

# --- Stage 3: build dataset.jsonl using pairs + channel ---
def load_channel_obs_given_true(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))

def reverse_to_true_given_obs(p_obs_given_true: Dict[str, Dict[str, float]], min_mass: float = 1e-9) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for true_tok, row in p_obs_given_true.items():
        for obs_tok, p in row.items():
            if p > 0:
                buckets[obs_tok][true_tok] += float(p)
    out: Dict[str, Dict[str, float]] = {}
    for obs_tok, ctr in buckets.items():
        s = sum(ctr.values())
        if s <= 0: continue
        out[obs_tok] = {t: float(c)/s for t, c in ctr.items() if c >= min_mass}
    return out

def prior_topk_for_obs_tokens(obs_tokens: List[str], p_true_given_obs: Dict[str, Dict[str, float]],
                              topk: int = 3, min_prob: float = 0.05) -> Dict[str, List[Tuple[str, float]]]:
    uniq = list(dict.fromkeys(obs_tokens))
    out: Dict[str, List[Tuple[str,float]]] = {}
    for o in uniq:
        row = p_true_given_obs.get(o, {})
        if not row: continue
        ranked = sorted(row.items(), key=lambda x: -x[1])
        ranked = [(t, float(p)) for t, p in ranked[:topk] if p >= min_prob]
        if ranked:
            out[o] = ranked
    return out


def shift_utt_id(utt_id: str, offset: int) -> str:
    m = UTT_PREFIX_RE.match(utt_id or "")
    if not m: return utt_id
    num = int(m.group(1))
    return f"S{num+offset:03d}" + utt_id[m.end():]

def parse_label_file_textmap(fpath: Path) -> Dict[str, str]:
    out = {}
    with fpath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith('"'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2: continue
            uid, txt = parts[0], parts[1].strip()
            out[uid] = txt
    return out

def build_text_lookup_strict(cdsd_root: Optional[Path]) -> Dict[str, str]:
    lut: Dict[str, str] = {}
    if not cdsd_root or not cdsd_root.exists():
        return lut
    text1 = cdsd_root / "1h" / "Text"
    if text1.is_dir():
        for f in sorted(text1.glob("*_label.txt")):
            lut.update(parse_label_file_textmap(f))
    text10 = cdsd_root / "10h" / "Text"
    if text10.is_dir():
        for f in sorted(text10.glob("*_label.txt")):
            d = parse_label_file_textmap(f)
            for uid, txt in d.items():
                lut[shift_utt_id(uid, +100)] = txt
    return lut

def decide_restore_or_delete_for_insert(aligned, idx, tau_insert, p_true_given_obs, window=2):
    t, o, op = aligned[idx]
    assert op == 'I'
    if o is None:
        return ("I_DEL", None)
    is_init = is_initial(o)
    is_fin  = is_final(o)
    for d in range(1, window+1):
        k = idx - d
        if k >= 0 and aligned[k][2] == 'D':
            t_del = aligned[k][0]
            if (is_init and is_final(t_del)) or (is_fin and is_initial(t_del)):
                return ("I_R", t_del)
        k = idx + d
        if k < len(aligned) and aligned[k][2] == 'D':
            t_del = aligned[k][0]
            if (is_init and is_final(t_del)) or (is_fin and is_initial(t_del)):
                return ("I_R", t_del)
    row = p_true_given_obs.get(o, {})
    if row:
        t1, p1 = max(row.items(), key=lambda x: x[1])
        if p1 >= tau_insert:
            return ("I_R", t1)
    return ("I_DEL", None)

def stage_build_dataset(pairs_path: Path, channel_path: Path, out_path: Path,
                        cdsd_root: Optional[Path], drop_markers: bool, keep_tone: bool,
                        ignore_erhua: bool, tau_insert: float,
                        prior_topk: int, prior_min_prob: float,
                        save_summary: bool = True, log_missing_text: Optional[Path] = None,
                        export_format: str = "qwen_instruct_json",
                        system_prompt: Optional[str] = None,
                        tasks: Optional[Set[str]] = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = tasks or {"TEXT", "PHONEME", "LABEL"}
    DEFAULT_SYSTEM = "你是一个拼音序列纠错助手。"
    system_prompt = system_prompt or DEFAULT_SYSTEM

    p_obs_given_true = load_channel_obs_given_true(channel_path)
    p_true_given_obs = reverse_to_true_given_obs(p_obs_given_true)

    text_lut = build_text_lookup_strict(cdsd_root) if cdsd_root else {}

    try:
        total_lines = sum(1 for _ in open(pairs_path, "r", encoding="utf-8") if _.strip())
    except Exception:
        total_lines = None

    stat = Counter(); stat_ir = 0; stat_idel = 0
    skipped_no_text = 0; miss_ids: List[str] = []

    qwen_records: List[Dict] = []
    fout = None
    if export_format == "multitask_jsonl":
        fout = out_path.open("w", encoding="utf-8")

    with pairs_path.open("r", encoding="utf-8") as f:
        it = tqdm(f, total=total_lines, desc="[build] dataset", unit="utt") if total_lines else f
        for line in it:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            utt_id = (obj.get("utt_id") or "").strip()
            text = (obj.get("text") or obj.get("txt") or obj.get("transcript") or "").strip()
            if not text and text_lut:
                text = text_lut.get(utt_id, "").strip()
            if not text:
                skipped_no_text += 1
                if log_missing_text is not None:
                    miss_ids.append(utt_id)
                continue

            ref_tokens = norm_seq(obj["ref"], drop_markers=drop_markers,
                                  drop_tone=(not keep_tone), ignore_erhua=ignore_erhua)
            obs_tokens = norm_seq(obj["obs"], drop_markers=drop_markers,
                                  drop_tone=(not keep_tone), ignore_erhua=ignore_erhua)
            al = weighted_align_phoneme(ref_tokens, obs_tokens)

            phn_target: List[str] = []
            label_seq: List[str] = []
            for idx, (t, o, op) in enumerate(al):
                if op in ('M','S'):
                    phn_target.append(t); label_seq.append(op)
                elif op == 'D':
                    phn_target.append(t); label_seq.append('D')
                else:
                    decision, t_restore = decide_restore_or_delete_for_insert(
                        aligned=al, idx=idx, tau_insert=tau_insert,
                        p_true_given_obs=p_true_given_obs, window=2
                    )
                    if decision == 'I_R' and t_restore is not None:
                        phn_target.append(t_restore); label_seq.append('I_R'); stat_ir += 1
                    else:
                        label_seq.append('I_DEL'); stat_idel += 1

            prior_map = prior_topk_for_obs_tokens(
                obs_tokens, p_true_given_obs, topk=prior_topk, min_prob=prior_min_prob
            )
            prior_items = []
            for o, lst in prior_map.items():
                inner = ", ".join([f"{t}({p:.2f})" for t, p in lst])
                prior_items.append(f"{o}:[{inner}]")
            prior_str = "{ " + ", ".join(prior_items) + " }" if prior_items else "{}"

            flags = []
            if ignore_erhua: flags.append("ignore_erhua")
            if not keep_tone: flags.append("drop_tone")
            flag_str = "{ " + ", ".join(flags) + " }" if flags else "{}"

            if export_format == "qwen_instruct_json":
                if "TEXT" in tasks:
                    qwen_records.append({
                        "utt_id": utt_id,
                        "instruction": "请根据需要纠错的拼音序列，参考可能的标准拼音结果及概率，把拼音序列还原为标准中文文本。",
                        "input": f"需要纠错的拼音序列： {' '.join(obs_tokens)}\n可能的标准拼音结果及概率： {prior_str}",
                        "output": text,
                        "system": system_prompt,
                    })
                    stat["TEXT"] += 1
                if "PHONEME" in tasks:
                    qwen_records.append({
                        "utt_id": utt_id,
                        "instruction": "请根据需要纠错的拼音序列，参考可能的标准拼音结果及概率，推测标准拼音序列。",
                        "input": f"需要纠错的拼音序列： {' '.join(obs_tokens)}\n可能的标准拼音结果及概率： {prior_str}",
                        "output": " ".join(phn_target),
                        "system": system_prompt,
                    })
                    stat["PHONEME"] += 1
                if "LABEL" in tasks:
                    qwen_records.append({
                        "utt_id": utt_id,
                        "instruction": "请根据需要纠错的拼音序列和标准拼音序列，输出编辑行为标签。（取值集合：M,S,D,I_R,I_DEL）。",
                        "input": f"需要纠错的拼音序列： {' '.join(obs_tokens)}\n标准拼音序列： {' '.join(phn_target)}",
                        "output": " ".join(label_seq),
                        "system": system_prompt,
                    })
                    stat["LABEL"] += 1

                if "LABEL" in tasks:
                    qwen_records.append({
                        "utt_id": utt_id,
                        "instruction": "请根据需要纠错的拼音序列，参考可能的标准拼音结果及概率，推测编辑行为标签。（取值集合：M,S,D,I_R,I_DEL）。",
                        "input": f"需要纠错的拼音序列： {' '.join(obs_tokens)}\n可能的标准拼音结果及概率： {prior_str}",
                        "output": " ".join(label_seq),
                        "system": system_prompt,
                    })
                    stat["LABEL"] += 1

            else:
                inst = (
                    "<INST>\n"
                    f"flags={flag_str};\n"
                    f"prior_topk={prior_str}\n"
                    f"OBS= {' '.join(obs_tokens)}\n"
                    "</INST>"
                )
                if "PHONEME" in tasks:
                    fout.write(json.dumps({
                        "task": "PHONEME", "utt_id": utt_id,
                        "input": inst + "\n<TASK=PHONEME>",
                        "output": f"REF_PHN= {' '.join(phn_target)}"
                    }, ensure_ascii=False) + "\n"); stat["PHONEME"] += 1
                if "TEXT" in tasks:
                    fout.write(json.dumps({
                        "task": "TEXT", "utt_id": utt_id,
                        "input": inst + "\n<TASK=TEXT>",
                        "output": f"REF_TXT= {text}".strip()
                    }, ensure_ascii=False) + "\n"); stat["TEXT"] += 1
                if "LABEL" in tasks:
                    fout.write(json.dumps({
                        "task": "LABEL", "utt_id": utt_id,
                        "input": inst + "\n<TASK=LABEL>",
                        "output": f"LABELS= {' '.join(label_seq)}"
                    }, ensure_ascii=False) + "\n"); stat["LABEL"] += 1

    if export_format == "qwen_instruct_json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(qwen_records, f, ensure_ascii=False, indent=2)
    else:
        fout.close()

    if save_summary:
        summary = {
            "total_samples_written": int(sum(stat.values())),
            "by_task": dict(stat),
            "insert_restore": int(stat_ir),
            "insert_delete": int(stat_idel),
            "skipped_no_text": int(skipped_no_text),
            "tau_insert": float(tau_insert),
            "prior_topk": int(prior_topk),
            "prior_min_prob": float(prior_min_prob),
            "keep_tone": bool(keep_tone),
            "ignore_erhua": bool(ignore_erhua),
            "cdsd_root_used": str(cdsd_root.resolve()) if cdsd_root else None,
            "export_format": export_format
        }
        (out_path.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Stage:build] wrote: {out_path.resolve()}")
    return out_path

# --- CLI helpers ---
def parse_tasks(tasks_str: Optional[str]) -> Set[str]:
    valid = {"TEXT", "PHONEME", "LABEL"}
    alias = {"T": "TEXT", "P": "PHONEME", "L": "LABEL"}

    if not tasks_str or not tasks_str.strip():
        return set(valid)

    result: Set[str] = set()
    for raw in re.split(r"[,\s]+", tasks_str.strip()):
        if not raw:
            continue
        key = raw.upper()
        key = alias.get(key, key)
        if key not in valid:
            raise ValueError(f"Unknown task: {raw!r}. Valid: TEXT, PHONEME, LABEL (or T/P/L).")
        result.add(key)

    if not result:
        raise ValueError("At least one task is required.")
    return result

# --- Entry point ---
def main():
    ap = argparse.ArgumentParser(description="一体化：pairs -> channel -> dataset.jsonl")
    ap.add_argument("--stage", choices=["pairs","analyze","build","all"], default="all",
                    help="运行阶段（默认 all）")
    ap.add_argument("--text-dir", type=str, help="CDSD 文本目录（如 .../10h/Text）")
    ap.add_argument("--audio-root", type=str, help="CDSD 音频根目录（如 .../10h/Audio）")
    ap.add_argument("--final-model", type=str, help="CTC 模型目录（包含 processor/ 与 模型本体）")
    ap.add_argument("--pairs", type=str, help="可选：已存在的 pairs.jsonl（若提供可跳过 Stage:pairs）")
    ap.add_argument("--channel", type=str, help="可选：已存在的 channel_global.json（若提供可跳过 Stage:analyze）")
    ap.add_argument("--outdir", type=str, required=True, help="输出根目录（三阶段统一放这里）")
    ap.add_argument("--dataset", type=str, default=None, help="最终 dataset.jsonl 路径（默认 outdir/dataset.jsonl）")

    ap.add_argument("--drop-markers", action="store_true", help="去掉 %/$/[UNK]")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--drop-tone", action="store_true", help="去掉调号")
    grp.add_argument("--keep-tone", action="store_true", help="保留调号(默认)")
    ap.add_argument("--ignore-erhua", action="store_true", help="忽略儿化差异")

    ap.add_argument("--alpha", type=float, default=0.0, help="行内 Laplace 平滑（一般置 0）")
    ap.add_argument("--beta", type=float, default=10.0, help="收缩强度（越大越靠拢回退分布）")
    ap.add_argument("--backoff", type=str, choices=["class","global"], default="class", help="回退分布粒度")
    ap.add_argument("--backoff-add-k", type=int, default=20, help="把回退分布的 Top-K 观测键并入每一行")
    ap.add_argument("--alpha-backoff", type=float, default=1e-6, help="回退分布的轻微平滑")

    ap.add_argument("--cdsd-root", type=str, default=None, help="用于文本严格回填的 CDSD 根（含 1h/10h/Text）")
    ap.add_argument("--tau-insert", type=float, default=0.4, help="I 位补回阈值（基于 P(true|obs) Top-1）")
    ap.add_argument("--prior-topk", type=int, default=10, help="提示里每个 obs 的 true 候选最多列多少个")
    ap.add_argument("--prior-min-prob", type=float, default=0.001, help="先验候选最小概率阈值")
    ap.add_argument("--log-missing-text", type=str, default=None, help="把缺文本的 utt_id 另存到该 txt")
    ap.add_argument("--export-format", type=str,
                    choices=["qwen_instruct_json", "multitask_jsonl"],
                    default="qwen_instruct_json",
                    help="最终导出格式：qwen_instruct_json(默认) 或 兼容旧版 multitask_jsonl")
    ap.add_argument("--system-file", type=str, default=None,
                    help="自定义系统提示词 txt 文件路径（不指定则使用内置的‘构音障碍语言治疗师’提示）")
    ap.add_argument("--tasks",type=str,default="TEXT,PHONEME,LABEL",help="选择导出的任务，逗号分隔；可选：TEXT,PHONEME,LABEL；支持别名 T,P,L。例如：--tasks TEXT 或 --tasks T,P")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    keep_tone = args.keep_tone or (not args.drop_tone)

    pairs_path = Path(args.pairs) if args.pairs else (outdir / "pairs.jsonl")
    channel_path = Path(args.channel) if args.channel else (outdir / "channel_global.json")
    dataset_path = Path(args.dataset) if args.dataset else (outdir / "dataset.jsonl")

    if args.stage in ("pairs","all"):
        if args.pairs and Path(args.pairs).is_file():
            print(f"[Skip:pairs] 已指定现有 pairs: {args.pairs}")
        else:
            if not (args.text_dir and args.audio_root and args.final_model):
                ap.error("--stage pairs/all 需要 --text-dir --audio-root --final-model")
            pairs_path = stage_pairs(
                text_dir=Path(args.text_dir), audio_root=Path(args.audio_root),
                final_model=Path(args.final_model), outdir=outdir,
                drop_markers=args.drop_markers, keep_tone=keep_tone,
                sample_rate=16000, save_pairs_path=pairs_path if args.pairs else None
            )

    if args.stage in ("analyze","all"):
        if args.channel and Path(args.channel).is_file():
            print(f"[Skip:analyze] 已指定现有 channel: {args.channel}")
        else:
            if not pairs_path.is_file():
                ap.error(f"找不到 pairs.jsonl: {pairs_path}")
            channel_path = export_channel_with_shrinkage(
                pairs_path=pairs_path, outdir=outdir,
                drop_markers=args.drop_markers, keep_tone=keep_tone, ignore_erhua=args.ignore_erhua,
                alpha=args.alpha, beta=args.beta, backoff=args.backoff,
                backoff_add_k=args.backoff_add_k, alpha_backoff=args.alpha_backoff,
                topk_csv=10
            )

    if args.stage in ("build","all"):
        if not pairs_path.is_file():
            ap.error(f"找不到 pairs.jsonl: {pairs_path}")
        if not channel_path.is_file():
            ap.error(f"找不到 channel_global.json: {channel_path}")
        stage_build_dataset(
            pairs_path=pairs_path, channel_path=channel_path, out_path=dataset_path,
            cdsd_root=Path(args.cdsd_root) if args.cdsd_root else None,
            drop_markers=args.drop_markers, keep_tone=keep_tone, ignore_erhua=args.ignore_erhua,
            tau_insert=args.tau_insert, prior_topk=args.prior_topk, prior_min_prob=args.prior_min_prob,
            save_summary=True,
            log_missing_text=Path(args.log_missing_text) if args.log_missing_text else None,
            export_format=args.export_format,
            system_prompt=(Path(args.system_file).read_text(encoding="utf-8") if args.system_file else None),
            tasks=parse_tasks(args.tasks)
        )

    print("[DONE]")

if __name__ == "__main__":
    main()
