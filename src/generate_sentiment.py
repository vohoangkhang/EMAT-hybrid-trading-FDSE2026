"""
Tạo sentiment scores cho tin tức tiếng Việt bằng multilingual model.
====================================================================
Model chính: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
Fallback:    nlptown/bert-base-multilingual-uncased-sentiment
Fallback 2:  TextBlob (rule-based, cần translate sang EN)

Output: data_src/raw_clean/sentiment_vi/{SYMBOL}_sentiment.csv
Cột: Date, Header, Content, sentiment_score, impact_score, relevance_score,
     confidence, short_term_score, medium_term_score, sentiment_momentum,
     sentiment_volatility, sentiment_trend
"""
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import numpy as np
import pandas as pd

# ======================================================================
# HẰNG SỐ
# ======================================================================

SYMBOLS = list({
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "JPM": "JPM",
    "XOM": "XOM",
    "VCB": "VCB.VN",
    "HPG": "HPG.VN",
    "FPT": "FPT.VN",
}.keys())

# Model chính: 3 classes (negative=0, neutral=1, positive=2)
PRIMARY_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
# Fallback: 5 stars → map [-1, 1]
FALLBACK_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 16  # Nhỏ để tránh OOM trên GPU yếu / CPU

# Cột output bắt buộc
OUTPUT_COLS = [
    "Date", "Header", "Content",
    "sentiment_score", "impact_score", "relevance_score",
    "confidence", "short_term_score", "medium_term_score",
    "sentiment_momentum", "sentiment_volatility", "sentiment_trend",
]


# ======================================================================
# SENTIMENT ENGINE: TRANSFORMER-BASED
# ======================================================================


class TransformerSentiment:
    """
    Wrapper cho HuggingFace sentiment pipeline.
    Hỗ trợ primary model (3 classes) và fallback (5 stars).
    """

    def __init__(self):
        self.pipeline = None
        self.model_name = None
        self.model_type = None  # 'three_class' hoặc 'five_star'
        self._load_model()

    def _load_model(self):
        """Thử load model chính, nếu fail → fallback."""
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            print("  ⚠ transformers chưa cài. Chạy: pip install transformers torch")
            return

        # Xác định device
        device_id = -1  # CPU mặc định
        try:
            import torch
            if torch.cuda.is_available():
                device_id = 0
                print(f"  🔥 GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("  💻 Chạy trên CPU (sẽ chậm hơn)")
        except ImportError:
            print("  💻 Chạy trên CPU (torch không có CUDA)")

        # Thử model chính
        try:
            print(f"  Đang tải model: {PRIMARY_MODEL}...")
            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model=PRIMARY_MODEL,
                tokenizer=PRIMARY_MODEL,
                device=device_id,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                top_k=None,  # Lấy probabilities cho tất cả classes
            )
            self.model_name = PRIMARY_MODEL
            self.model_type = "three_class"
            print(f"  ✔ Đã tải: {PRIMARY_MODEL}")
            return
        except Exception as e:
            print(f"  ⚠ Không tải được model chính: {e}")

        # Fallback model
        try:
            print(f"  Đang tải fallback: {FALLBACK_MODEL}...")
            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model=FALLBACK_MODEL,
                tokenizer=FALLBACK_MODEL,
                device=device_id,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                top_k=None,
            )
            self.model_name = FALLBACK_MODEL
            self.model_type = "five_star"
            print(f"  ✔ Đã tải fallback: {FALLBACK_MODEL}")
        except Exception as e:
            print(f"  ✘ Không tải được cả 2 model transformer: {e}")
            print("  → Sẽ dùng TextBlob fallback")

    def predict_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """
        Dự đoán sentiment cho batch texts.
        Trả về list of (sentiment_score, confidence).
        sentiment_score: [-1, 1], confidence: [0, 1].
        """
        if self.pipeline is None:
            return self._textblob_fallback(texts)

        results = []
        # Xử lý theo batch để tối ưu throughput
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            # Đảm bảo text không rỗng
            batch = [t if t and len(t.strip()) > 0 else "neutral" for t in batch]

            try:
                outputs = self.pipeline(batch)
                for output in outputs:
                    score, conf = self._parse_output(output)
                    results.append((score, conf))
            except Exception as e:
                # Nếu batch fail → xử lý từng text riêng
                for text in batch:
                    try:
                        output = self.pipeline([text])[0]
                        score, conf = self._parse_output(output)
                        results.append((score, conf))
                    except Exception:
                        results.append((0.0, 0.0))

        return results

    def _parse_output(self, output: list) -> Tuple[float, float]:
        """
        Parse output từ pipeline thành (score, confidence).
        output là list of {'label': ..., 'score': ...}.
        """
        if self.model_type == "three_class":
            return self._parse_three_class(output)
        elif self.model_type == "five_star":
            return self._parse_five_star(output)
        return (0.0, 0.0)

    def _parse_three_class(self, output: list) -> Tuple[float, float]:
        """
        3 classes: negative, neutral, positive.
        score = P(positive) - P(negative), range [-1, 1].
        """
        probs = {}
        for item in output:
            label = item["label"].lower()
            probs[label] = item["score"]

        p_pos = probs.get("positive", probs.get("pos", 0.0))
        p_neg = probs.get("negative", probs.get("neg", 0.0))
        p_neu = probs.get("neutral", probs.get("neu", 0.0))

        score = p_pos - p_neg
        confidence = max(p_pos, p_neg, p_neu)

        return (score, confidence)

    def _parse_five_star(self, output: list) -> Tuple[float, float]:
        """
        5 stars (1-5) → map sang [-1, 1].
        1 star = -1.0, 3 stars = 0.0, 5 stars = 1.0.
        """
        star_scores = {}
        for item in output:
            # Label format: "1 star", "2 stars", ...
            label = item["label"]
            star = int(label.split()[0])
            star_scores[star] = item["score"]

        # Weighted average → map to [-1, 1]
        weighted_sum = sum(star * prob for star, prob in star_scores.items())
        total_prob = sum(star_scores.values())
        avg_star = weighted_sum / max(total_prob, 1e-8)

        # Map: 1→-1, 3→0, 5→1
        score = (avg_star - 3.0) / 2.0
        confidence = max(star_scores.values()) if star_scores else 0.0

        return (score, confidence)

    def _textblob_fallback(self, texts: List[str]) -> List[Tuple[float, float]]:
        """
        Fallback cuối cùng: TextBlob polarity (chỉ hoạt động tốt cho EN).
        Với VI, kết quả sẽ kém nhưng vẫn có output.
        """
        try:
            from textblob import TextBlob
        except ImportError:
            print("  ⚠ TextBlob chưa cài. Trả về score = 0 cho tất cả.")
            return [(0.0, 0.0)] * len(texts)

        print("  ⚠ Dùng TextBlob fallback (chất lượng thấp cho tiếng Việt)")
        results = []
        for text in texts:
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity  # [-1, 1]
                subjectivity = blob.sentiment.subjectivity  # [0, 1]
                # Confidence = 1 - subjectivity (càng khách quan càng tin cậy)
                confidence = max(0.3, 1.0 - subjectivity)
                results.append((polarity, confidence))
            except Exception:
                results.append((0.0, 0.0))

        return results


# ======================================================================
# TÍNH CÁC CHỈ SỐ PHỤ (DERIVED FEATURES)
# ======================================================================


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các cột sentiment phụ từ sentiment_score.
    Yêu cầu df đã có cột: sentiment_score, confidence.
    """
    s = df["sentiment_score"].copy()

    # impact_score: mức ảnh hưởng = |sentiment_score|
    df["impact_score"] = s.abs()

    # relevance_score: dùng confidence làm proxy
    df["relevance_score"] = df["confidence"]

    # short_term_score: = sentiment_score (tác động ngay)
    df["short_term_score"] = s

    # medium_term_score: EMA span=5 → tác động trung hạn
    df["medium_term_score"] = s.ewm(span=5, min_periods=1).mean()

    # sentiment_momentum: delta giữa 2 tin liên tiếp
    df["sentiment_momentum"] = s.diff().fillna(0.0)

    # sentiment_volatility: rolling std 10 tin
    df["sentiment_volatility"] = s.rolling(window=10, min_periods=1).std().fillna(0.0)

    # sentiment_trend: EMA(20) - EMA(5) → xu hướng dài hạn vs ngắn hạn
    ema_long = s.ewm(span=20, min_periods=1).mean()
    ema_short = s.ewm(span=5, min_periods=1).mean()
    df["sentiment_trend"] = ema_long - ema_short

    return df


# ======================================================================
# XỬ LÝ 1 MÃ CỔ PHIẾU
# ======================================================================


def process_symbol(symbol: str, engine: TransformerSentiment) -> Optional[pd.DataFrame]:
    """
    Load news_vi → tính sentiment → tính derived features → lưu file.
    Trả về DataFrame kết quả hoặc None nếu lỗi.
    """
    # Tìm file news_vi (thử raw_clean trước, sau đó data)
    vi_path = config.RAW_CLEAN_NEWS_VI / f"{symbol}_vi_news.csv"
    if not vi_path.exists():
        vi_path = config.NEWS_VI_SRC / f"{symbol}_vi_news.csv"
    if not vi_path.exists():
        print(f"  ⚠ Không tìm thấy news_vi cho {symbol}")
        return None

    try:
        df = pd.read_csv(vi_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(vi_path, encoding="utf-8")

    if df.empty:
        print(f"  ⚠ File news_vi rỗng cho {symbol}")
        return None

    # Chuẩn hóa tên cột (đề phòng format khác nhau)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower == "date":
            col_map[col] = "Date"
        elif lower == "header":
            col_map[col] = "Header"
        elif lower == "content":
            col_map[col] = "Content"
    df = df.rename(columns=col_map)

    # Đảm bảo có đủ cột
    if "Header" not in df.columns:
        print(f"  ⚠ {symbol}: Thiếu cột Header")
        return None
    if "Content" not in df.columns:
        df["Content"] = df["Header"]

    # Loại bỏ dòng trống
    df = df.dropna(subset=["Header"]).reset_index(drop=True)
    print(f"  📊 {symbol}: {len(df)} tin tức")

    if len(df) == 0:
        return None

    # Kết hợp Header + Content làm input cho model
    texts = []
    for _, row in df.iterrows():
        header = str(row["Header"]).strip()
        content = str(row["Content"]).strip()
        # Nối Header + Content, truncate sẽ do tokenizer xử lý
        combined = f"{header}. {content}" if content != header else header
        texts.append(combined)

    # Chạy sentiment prediction
    print(f"  🔍 Đang phân tích sentiment ({len(texts)} texts)...")

    try:
        # Dùng tqdm nếu có
        from tqdm import tqdm
        predictions = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE),
                      desc=f"  {symbol}", unit="batch"):
            batch = texts[i:i + BATCH_SIZE]
            batch_results = engine.predict_batch(batch)
            predictions.extend(batch_results)
    except ImportError:
        # Không có tqdm → in progress thủ công
        predictions = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_results = engine.predict_batch(batch)
            predictions.extend(batch_results)
            batch_idx = i // BATCH_SIZE + 1
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                print(f"    Batch {batch_idx}/{total_batches}")

    # Gán scores vào DataFrame
    scores = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]

    df["sentiment_score"] = scores
    df["confidence"] = confidences

    # Tính các chỉ số phụ
    df = compute_derived_features(df)

    # Đảm bảo đúng thứ tự cột output
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[OUTPUT_COLS]

    # Làm tròn các cột số
    numeric_cols = OUTPUT_COLS[3:]  # Bỏ Date, Header, Content
    df[numeric_cols] = df[numeric_cols].round(6)

    return df


# ======================================================================
# HÀM CHÍNH
# ======================================================================


def run():
    """Hàm chính: tạo sentiment cho tất cả mã cổ phiếu."""
    print("\n" + "=" * 60)
    print("  TẠO SENTIMENT SCORES CHO TIN TỨC TIẾNG VIỆT")
    print("=" * 60)

    # Tạo thư mục output
    output_dir = config.RAW_CLEAN_SENTIMENT_VI
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror sang data/
    data_dir = config.SENTIMENT_VI_SRC
    data_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo engine (tải model 1 lần, dùng cho tất cả symbols)
    print("\n  Khởi tạo sentiment engine...")
    engine = TransformerSentiment()

    if engine.pipeline is not None:
        print(f"  Model: {engine.model_name}")
        print(f"  Type : {engine.model_type}")
    else:
        print("  ⚠ Không có model transformer → dùng TextBlob fallback")

    summary = {}

    for symbol in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"  [{symbol}] Xử lý sentiment")
        print(f"{'─' * 50}")

        try:
            df = process_symbol(symbol, engine)

            if df is not None and len(df) > 0:
                # Lưu raw_clean
                out_file = output_dir / f"{symbol}_sentiment.csv"
                df.to_csv(out_file, index=False, encoding="utf-8-sig")

                # Mirror sang data/
                df.to_csv(data_dir / f"{symbol}_sentiment.csv",
                          index=False, encoding="utf-8-sig")

                # Thống kê nhanh
                avg_score = df["sentiment_score"].mean()
                std_score = df["sentiment_score"].std()
                pos_pct = (df["sentiment_score"] > 0.2).mean() * 100
                neg_pct = (df["sentiment_score"] < -0.2).mean() * 100
                avg_conf = df["confidence"].mean()

                print(f"  ✔ {len(df)} tin → {out_file}")
                print(f"    Avg score: {avg_score:+.4f} ± {std_score:.4f}")
                print(f"    Tích cực: {pos_pct:.1f}% | Tiêu cực: {neg_pct:.1f}%")
                print(f"    Avg confidence: {avg_conf:.4f}")

                summary[symbol] = {
                    "count": len(df),
                    "avg_score": round(avg_score, 4),
                    "std_score": round(std_score, 4),
                    "avg_conf": round(avg_conf, 4),
                }
            else:
                print(f"  ⚠ Không có kết quả cho {symbol}")
                summary[symbol] = {"count": 0, "error": "no data"}

        except Exception as e:
            print(f"  ✘ Lỗi xử lý {symbol}: {e}")
            summary[symbol] = {"count": 0, "error": str(e)}

    # Báo cáo tổng kết
    print(f"\n{'=' * 60}")
    print("  TỔNG KẾT SENTIMENT ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  {'Symbol':<8} | {'Count':>6} | {'Avg Score':>10} | {'Std':>8} | {'Conf':>6}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
    for symbol, info in summary.items():
        if "error" in info and info["count"] == 0:
            print(f"  {symbol:<8} | {'N/A':>6} | {info.get('error', 'no data')}")
        else:
            print(f"  {symbol:<8} | {info['count']:>6} | "
                  f"{info['avg_score']:>+10.4f} | {info['std_score']:>8.4f} | "
                  f"{info['avg_conf']:>6.4f}")
    print(f"{'=' * 60}")

    print(f"\n  📂 Output: {output_dir}")

    return summary


if __name__ == "__main__":
    run()
