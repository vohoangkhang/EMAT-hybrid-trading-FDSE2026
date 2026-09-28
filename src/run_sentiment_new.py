"""
Tạo sentiment scores cho VI news dùng transformer trực tiếp.
Không qua generate_sentiment.py (tránh bug fallback).
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

SYMBOLS = ["MSFT", "JPM", "XOM", "VCB", "HPG", "FPT"]
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
BATCH_SIZE = 8  # Nho hon de tranh OOM tren 4GB VRAM
MAX_LENGTH = 256  # Giam de tiet kiem VRAM

OUTPUT_COLS = [
    "Date", "Header", "Content",
    "sentiment_score", "impact_score", "relevance_score",
    "confidence", "short_term_score", "medium_term_score",
    "sentiment_momentum", "sentiment_volatility", "sentiment_trend",
]


def load_pipeline():
    """Load sentiment pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {gpu_name}")

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
        top_k=None,
    )
    print(f"  Model: {MODEL_NAME}")
    return pipe


def parse_output(output):
    """Parse 3-class output thanh (score, confidence)."""
    probs = {}
    for item in output:
        probs[item["label"].lower()] = item["score"]

    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    p_neu = probs.get("neutral", 0.0)

    score = p_pos - p_neg  # [-1, 1]
    confidence = max(p_pos, p_neg, p_neu)
    return score, confidence


def compute_derived(df):
    """Tinh cac chi so phu tu sentiment_score."""
    s = df["sentiment_score"]
    df["impact_score"] = s.abs()
    df["relevance_score"] = df["confidence"]
    df["short_term_score"] = s
    df["medium_term_score"] = s.ewm(span=5, min_periods=1).mean()
    df["sentiment_momentum"] = s.diff().fillna(0.0)
    df["sentiment_volatility"] = s.rolling(window=10, min_periods=1).std().fillna(0.0)
    ema_long = s.ewm(span=20, min_periods=1).mean()
    ema_short = s.ewm(span=5, min_periods=1).mean()
    df["sentiment_trend"] = ema_long - ema_short
    return df


def process_symbol(symbol, pipe):
    """Xu ly sentiment cho 1 ma."""
    # Tim file news_vi
    vi_path = config.RAW_CLEAN_NEWS_VI / f"{symbol}_vi_news.csv"
    if not vi_path.exists():
        print(f"  Khong tim thay: {vi_path}")
        return None

    try:
        df = pd.read_csv(vi_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(vi_path, encoding="utf-8")

    if df.empty:
        return None

    # Chuan hoa cot
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc == "date": col_map[col] = "Date"
        elif lc == "header": col_map[col] = "Header"
        elif lc == "content": col_map[col] = "Content"
    df = df.rename(columns=col_map)

    if "Header" not in df.columns:
        return None
    if "Content" not in df.columns:
        df["Content"] = df["Header"]

    df = df.dropna(subset=["Header"]).reset_index(drop=True)
    print(f"  {len(df)} tin tuc")

    # Tao texts
    texts = []
    for _, row in df.iterrows():
        h = str(row["Header"]).strip()
        c = str(row["Content"]).strip()
        # Giu text ngan de tranh OOM
        combined = f"{h}. {c}" if c != h and c != "nan" else h
        texts.append(combined[:500])

    # Chay sentiment theo batch
    scores = []
    confidences = []
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t if t.strip() else "neutral" for t in batch]

        try:
            outputs = pipe(batch)
            for output in outputs:
                s, c = parse_output(output)
                scores.append(s)
                confidences.append(c)
        except Exception as e:
            # Xu ly tung text neu batch fail
            for text in batch:
                try:
                    output = pipe([text])[0]
                    s, c = parse_output(output)
                    scores.append(s)
                    confidences.append(c)
                except Exception:
                    scores.append(0.0)
                    confidences.append(0.0)

        done = min(i + BATCH_SIZE, total)
        if done % 200 == 0 or done == total:
            print(f"    {done}/{total} ({done*100//total}%)")

    df["sentiment_score"] = scores
    df["confidence"] = confidences
    df = compute_derived(df)

    # Dam bao du cot
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[OUTPUT_COLS]
    numeric_cols = OUTPUT_COLS[3:]
    df[numeric_cols] = df[numeric_cols].round(6)

    return df


def run():
    print("\n" + "=" * 60)
    print("  TAO SENTIMENT SCORES (XLM-RoBERTa)")
    print("=" * 60)

    pipe = load_pipeline()

    output_dir = config.RAW_CLEAN_SENTIMENT_VI
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = config.SENTIMENT_VI_SRC
    data_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for symbol in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"  [{symbol}]")
        print(f"{'─' * 50}")

        df = process_symbol(symbol, pipe)

        if df is not None and len(df) > 0:
            df.to_csv(output_dir / f"{symbol}_sentiment.csv", index=False, encoding="utf-8-sig")
            df.to_csv(data_dir / f"{symbol}_sentiment.csv", index=False, encoding="utf-8-sig")

            avg = df["sentiment_score"].mean()
            std = df["sentiment_score"].std()
            pos = (df["sentiment_score"] > 0.2).mean() * 100
            neg = (df["sentiment_score"] < -0.2).mean() * 100

            print(f"  OK: {len(df)} tin")
            print(f"  Avg: {avg:+.4f} +/- {std:.4f}")
            print(f"  Pos: {pos:.1f}% | Neg: {neg:.1f}%")
            summary[symbol] = {"count": len(df), "avg": avg}
        else:
            print(f"  SKIP")
            summary[symbol] = {"count": 0}

        # Giai phong VRAM
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("  TONG KET")
    print(f"{'=' * 60}")
    for sym, info in summary.items():
        print(f"  {sym}: {info['count']:,} tin" + (f" | avg={info.get('avg', 0):+.4f}" if info['count'] > 0 else ""))
    print("Done!")


if __name__ == "__main__":
    run()
