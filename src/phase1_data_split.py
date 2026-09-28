"""
Phase 1: Chuẩn Bị Dữ Liệu Đồng Bộ (Data Synchronization)
==========================================================
Chia dữ liệu OHLCV, EMAT Features, Tin tức, Sentiment theo chuẩn 70:10:20
cứng theo thời gian (Time-based split) để tránh Look-ahead Bias.

Logic Weekend Backward Shift: Tin T7/CN dồn về T6 trước đó.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Thêm thư mục src vào path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_prices(symbol: str) -> pd.DataFrame:
    """Load dữ liệu giá OHLCV cho một mã cổ phiếu."""
    path = config.PRICES_SRC / f"{symbol}_prices.csv"
    if not path.exists():
        # Fallback sang raw_clean
        path = config.RAW_CLEAN_PRICES / f"{symbol}_prices.csv"

    df = pd.read_csv(path)
    # Chuẩn hóa tên cột
    df = df.rename(columns={
        "time": "Date", "date": "Date",
        "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_emat_dataset(symbol: str) -> pd.DataFrame:
    """Load EMAT dataset (OHLCV + 7 Technical Indicators)."""
    path = config.EMAT_DATASETS_SRC / f"{symbol}_EMAT_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy EMAT dataset: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_news_en(symbol: str) -> pd.DataFrame:
    """Load tin tức tiếng Anh."""
    # Ưu tiên raw_clean (đầy đủ hơn)
    path = config.RAW_CLEAN_NEWS_EN / f"{symbol}_en_news.csv"
    if not path.exists():
        path = config.NEWS_EN_SRC / f"{symbol}_en_news.csv"
    if not path.exists():
        print(f"  [WARN] Không tìm thấy news EN cho {symbol}")
        return pd.DataFrame(columns=["date", "title", "stock"])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "title"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_news_vi(symbol: str) -> pd.DataFrame:
    """Load tin tức tiếng Việt (full-text)."""
    path = config.RAW_CLEAN_NEWS_VI / f"{symbol}_vi_news.csv"
    if not path.exists():
        path = config.NEWS_VI_SRC / f"{symbol}_vi_news.csv"
    if not path.exists():
        print(f"  [WARN] Không tìm thấy news VI cho {symbol}")
        return pd.DataFrame(columns=["Date", "Header", "Content"])

    df = pd.read_csv(path)
    # Chuẩn hóa tên cột
    df = df.rename(columns={"date": "Date", "header": "Header", "content": "Content"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_sentiment_vi(symbol: str) -> pd.DataFrame:
    """Load dữ liệu sentiment đã đánh giá (tiếng Việt)."""
    path = config.RAW_CLEAN_SENTIMENT_VI / f"{symbol}_sentiment.csv"
    if not path.exists():
        path = config.SENTIMENT_VI_SRC / f"{symbol}_sentiment.csv"
    if not path.exists():
        print(f"  [WARN] Không tìm thấy sentiment VI cho {symbol}")
        return pd.DataFrame(columns=["Date", "Header", "Content", "sentiment_score"])

    df = pd.read_csv(path)
    # Chuẩn hóa tên cột (raw_clean vs data có thể khác)
    df = df.rename(columns={"date": "Date", "header": "Header", "content": "Content"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "sentiment_score"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def shift_weekend_to_friday(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Dồn tin tức Thứ 7 và Chủ nhật về Thứ 6 trước đó.
    Logic thầy Khang: Tin cuối tuần phải gắn với ngày giao dịch gần nhất trước đó.
    """
    df = df.copy()

    def _shift(d):
        if d.dayofweek == 5:  # Thứ 7 → lùi 1 ngày
            return d - pd.Timedelta(days=1)
        elif d.dayofweek == 6:  # Chủ nhật → lùi 2 ngày
            return d - pd.Timedelta(days=2)
        return d

    df[date_col] = df[date_col].apply(_shift)
    return df


def compute_split_dates(dates: pd.Series) -> tuple:
    """
    Tính mốc thời gian chia 70:10:20 dựa trên chuỗi ngày giao dịch.
    Trả về (train_end_date, val_end_date).
    """
    sorted_dates = dates.sort_values().unique()
    n = len(sorted_dates)

    train_end_idx = int(n * config.TRAIN_RATIO) - 1
    val_end_idx = int(n * (config.TRAIN_RATIO + config.VAL_RATIO)) - 1

    train_end_date = pd.Timestamp(sorted_dates[train_end_idx])
    val_end_date = pd.Timestamp(sorted_dates[val_end_idx])

    return train_end_date, val_end_date


def split_by_date(df: pd.DataFrame, train_end: pd.Timestamp,
                  val_end: pd.Timestamp, date_col: str = "Date") -> dict:
    """Chia DataFrame theo mốc thời gian. Không shuffle."""
    return {
        "train": df[df[date_col] <= train_end].copy(),
        "val": df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy(),
        "test": df[df[date_col] > val_end].copy(),
    }


def process_symbol(symbol: str) -> dict:
    """
    Xử lý toàn bộ dữ liệu cho một mã cổ phiếu:
    1. Load tất cả nguồn dữ liệu
    2. Shift weekend cho tin tức
    3. Chia 70:10:20 đồng bộ theo timeline giá
    4. Lưu xuống data/train, data/val, data/test
    """
    print(f"\n{'='*50}")
    print(f"  ĐANG XỬ LÝ: {symbol}")
    print(f"{'='*50}")

    # 1. Load dữ liệu
    df_prices = load_prices(symbol)
    df_emat = load_emat_dataset(symbol)
    df_news_en = load_news_en(symbol)
    df_news_vi = load_news_vi(symbol)
    df_sentiment = load_sentiment_vi(symbol)

    print(f"  Prices      : {len(df_prices):>6} dòng | "
          f"{df_prices['Date'].min().date()} → {df_prices['Date'].max().date()}")
    print(f"  EMAT Dataset: {len(df_emat):>6} dòng | "
          f"{df_emat['Date'].min().date()} → {df_emat['Date'].max().date()}")
    print(f"  News EN     : {len(df_news_en):>6} dòng")
    print(f"  News VI     : {len(df_news_vi):>6} dòng")
    print(f"  Sentiment VI: {len(df_sentiment):>6} dòng")

    # 2. Shift weekend cho tin tức
    if not df_news_en.empty:
        df_news_en = shift_weekend_to_friday(df_news_en, date_col="date")
    if not df_news_vi.empty:
        df_news_vi = shift_weekend_to_friday(df_news_vi, date_col="Date")
    if not df_sentiment.empty:
        df_sentiment = shift_weekend_to_friday(df_sentiment, date_col="Date")

    # 3. Tính mốc chia dựa trên timeline giá (anchor point)
    train_end, val_end = compute_split_dates(df_prices["Date"])
    print(f"\n  Mốc chia thời gian:")
    print(f"    Train : ... → {train_end.date()}")
    print(f"    Val   : {train_end.date()} → {val_end.date()}")
    print(f"    Test  : {val_end.date()} → ...")

    # 4. Chia tất cả dataset theo cùng mốc
    splits_prices = split_by_date(df_prices, train_end, val_end)
    splits_emat = split_by_date(df_emat, train_end, val_end)
    splits_news_en = split_by_date(df_news_en, train_end, val_end, date_col="date")
    splits_news_vi = split_by_date(df_news_vi, train_end, val_end)
    splits_sentiment = split_by_date(df_sentiment, train_end, val_end)

    # 5. Lưu xuống từng thư mục
    stats = {}
    for split_name in ["train", "val", "test"]:
        out_dir = config.DATA_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        splits_prices[split_name].to_csv(
            out_dir / f"{symbol}_prices.csv", index=False)
        splits_emat[split_name].to_csv(
            out_dir / f"{symbol}_emat.csv", index=False)
        splits_news_en[split_name].to_csv(
            out_dir / f"{symbol}_news_en.csv", index=False)
        splits_news_vi[split_name].to_csv(
            out_dir / f"{symbol}_news_vi.csv", index=False)
        splits_sentiment[split_name].to_csv(
            out_dir / f"{symbol}_sentiment_vi.csv", index=False)

        n_prices = len(splits_prices[split_name])
        n_news = len(splits_news_en[split_name]) + len(splits_news_vi[split_name])
        n_sent = len(splits_sentiment[split_name])
        stats[split_name] = {
            "prices": n_prices, "news": n_news, "sentiment": n_sent
        }

    # 6. In báo cáo thống kê
    print(f"\n  Kết quả chia dữ liệu cho {symbol}:")
    print(f"  {'Split':<8} | {'Prices':>8} | {'News':>8} | {'Sentiment':>10}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for split_name, s in stats.items():
        print(f"  {split_name:<8} | {s['prices']:>8} | {s['news']:>8} | {s['sentiment']:>10}")

    # Kiểm tra tính hợp lệ
    total = sum(s["prices"] for s in stats.values())
    assert total == len(df_prices), \
        f"NGHIÊM TRỌNG: Tổng split ({total}) != tổng gốc ({len(df_prices)})"
    print(f"  ✔ Tổng kiểm tra: {total} dòng prices, không overlap.")

    return stats


def verify_no_overlap():
    """Kiểm tra chéo: không có ngày nào xuất hiện trong 2 tập khác nhau."""
    print(f"\n{'='*50}")
    print("  KIỂM TRA CHÉO: KHÔNG OVERLAP")
    print(f"{'='*50}")

    for symbol in config.SYMBOLS:
        dates = {}
        for split in ["train", "val", "test"]:
            path = config.DATA_DIR / split / f"{symbol}_prices.csv"
            if path.exists():
                df = pd.read_csv(path)
                dates[split] = set(pd.to_datetime(df["Date"]).dt.date)

        # Kiểm tra từng cặp
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if a in dates and b in dates:
                overlap = dates[a] & dates[b]
                if overlap:
                    print(f"  ✘ {symbol}: OVERLAP giữa {a} và {b}: {len(overlap)} ngày!")
                    return False

        # Kiểm tra thứ tự thời gian
        if all(s in dates for s in ["train", "val", "test"]):
            train_max = max(dates["train"])
            val_min = min(dates["val"])
            val_max = max(dates["val"])
            test_min = min(dates["test"])

            assert train_max < val_min, \
                f"{symbol}: train max ({train_max}) >= val min ({val_min})"
            assert val_max < test_min, \
                f"{symbol}: val max ({val_max}) >= test min ({test_min})"

        print(f"  ✔ {symbol}: Không overlap, thứ tự thời gian chính xác.")

    return True


def run():
    """Chạy toàn bộ Phase 1."""
    print("\n" + "=" * 60)
    print("  PHASE 1: CHUẨN BỊ DỮ LIỆU ĐỒNG BỘ")
    print("=" * 60)

    all_stats = {}
    for symbol in config.SYMBOLS:
        all_stats[symbol] = process_symbol(symbol)

    verify_no_overlap()

    print(f"\n{'='*60}")
    print("  ✔ PHASE 1 HOÀN TẤT")
    print(f"  Dữ liệu đã lưu tại: {config.DATA_DIR}")
    print(f"{'='*60}")
    return all_stats


if __name__ == "__main__":
    run()
