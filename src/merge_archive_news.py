"""
Merge dữ liệu news từ archive vào data_src hiện có.
Chỉ merge 3 mã có dữ liệu: AAPL, JPM, XOM.
FPT trong archive là FPT Industrial (Italy) → loại bỏ.
"""
import pandas as pd
from pathlib import Path

ARCHIVE_DIR = Path(r"D:\HUTECH\thayKhang\EMAT\archive")
DATA_DIR = Path(r"C:\Users\PC\.gemini\antigravity\worktrees\EMAT\update-ticker-sentiment-data\data_src")
NEWS_EN_RAW = DATA_DIR / "raw_clean" / "news_en"
NEWS_EN_DATA = DATA_DIR / "data" / "news_en"

TARGETS = ["AAPL", "JPM", "XOM"]


def load_archive():
    """Load và merge 2 file archive thành 1 DataFrame chuẩn (title, date, stock)."""
    print("  Đang đọc analyst_ratings_processed.csv...")
    df1 = pd.read_csv(ARCHIVE_DIR / "analyst_ratings_processed.csv", index_col=0)
    # Đã có format: title, date, stock
    df1 = df1[["title", "date", "stock"]].copy()
    print(f"  → {len(df1):,} dòng")

    print("  Đang đọc raw_partner_headlines.csv...")
    df2 = pd.read_csv(ARCHIVE_DIR / "raw_partner_headlines.csv", index_col=0)
    # Format: headline, url, publisher, date, stock → đổi tên cột
    df2 = df2[["headline", "date", "stock"]].copy()
    df2 = df2.rename(columns={"headline": "title"})
    print(f"  → {len(df2):,} dòng")

    # Gộp lại
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.dropna(subset=["title", "date", "stock"])
    print(f"  → Tổng sau gộp: {len(combined):,} dòng")

    return combined


def normalize_date(date_str):
    """Chuẩn hóa date string về YYYY-MM-DD."""
    try:
        dt = pd.to_datetime(date_str, utc=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None


def merge_for_symbol(symbol, archive_df):
    """Merge archive data vào news_en hiện có cho 1 mã."""
    # Lọc dữ liệu archive cho symbol
    arch_sub = archive_df[archive_df["stock"] == symbol].copy()
    if len(arch_sub) == 0:
        print(f"  ⚠ Không có dữ liệu archive cho {symbol}")
        return

    # Chuẩn hóa date
    arch_sub["date"] = arch_sub["date"].apply(normalize_date)
    arch_sub = arch_sub.dropna(subset=["date"])
    arch_sub = arch_sub[["title", "date", "stock"]].reset_index(drop=True)
    print(f"  Archive: {len(arch_sub):,} tin ({arch_sub['date'].min()} → {arch_sub['date'].max()})")

    # Load news_en hiện có
    existing_file = NEWS_EN_RAW / f"{symbol}_en_news.csv"
    if existing_file.exists():
        df_existing = pd.read_csv(existing_file)
        print(f"  Hiện có: {len(df_existing):,} tin")
    else:
        df_existing = pd.DataFrame(columns=["title", "date", "stock"])
        print(f"  Hiện có: 0 tin (file chưa tồn tại)")

    # Merge
    merged = pd.concat([df_existing, arch_sub], ignore_index=True)

    # De-duplicate theo title (lowercase) + date
    merged["_key"] = merged["title"].str.lower().str.strip() + "|" + merged["date"].astype(str)
    before = len(merged)
    merged = merged.drop_duplicates(subset="_key", keep="first")
    merged = merged.drop(columns="_key")
    dupes = before - len(merged)

    # Sắp xếp theo ngày
    merged = merged.sort_values("date").reset_index(drop=True)

    # Lưu vào raw_clean
    NEWS_EN_RAW.mkdir(parents=True, exist_ok=True)
    merged.to_csv(NEWS_EN_RAW / f"{symbol}_en_news.csv", index=False, encoding="utf-8")

    # Mirror vào data/
    NEWS_EN_DATA.mkdir(parents=True, exist_ok=True)
    merged.to_csv(NEWS_EN_DATA / f"{symbol}_en_news.csv", index=False, encoding="utf-8")

    print(f"  Sau merge: {len(merged):,} tin (loại {dupes:,} trùng)")
    print(f"  Date range: {merged['date'].min()} → {merged['date'].max()}")
    print(f"  ✔ Đã lưu: {NEWS_EN_RAW / f'{symbol}_en_news.csv'}")

    return len(merged)


def run():
    print("\n" + "=" * 60)
    print("  MERGE ARCHIVE NEWS VÀO DATA_SRC")
    print("=" * 60)

    # Load archive
    print("\n📂 Loading archive...")
    archive_df = load_archive()

    # Merge từng mã
    summary = {}
    for symbol in TARGETS:
        print(f"\n{'─' * 50}")
        print(f"  [{symbol}] Merge archive → news_en")
        print(f"{'─' * 50}")
        count = merge_for_symbol(symbol, archive_df)
        summary[symbol] = count or 0

    # Tổng kết
    print(f"\n{'=' * 60}")
    print("  TỔNG KẾT MERGE")
    print(f"{'=' * 60}")
    for symbol, count in summary.items():
        print(f"  {symbol}: {count:,} tin (sau merge)")
    print(f"{'=' * 60}")
    print("\n  Bước tiếp: Chạy generate_sentiment cho tin mới → re-train pipeline")


if __name__ == "__main__":
    run()
