"""
Chuẩn hóa và merge dữ liệu scraped_news vào data_src.
=======================================================
Xử lý 4 format nguồn EN (alphavantage, extended, finnhub, newsapi)
và format VI (extended) → chuẩn hóa về format pipeline.

EN output: title, date, stock
VI output: Date, Header, Content
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

SCRAPED_DIR = Path(r"C:\Users\PC\.gemini\antigravity\worktrees\EMAT\update-ticker-sentiment-data\scraped_news")
NEWS_EN_RAW = config.DATA_SRC / "raw_clean" / "news_en"
NEWS_VI_RAW = config.DATA_SRC / "raw_clean" / "news_vi"
NEWS_EN_DATA = config.DATA_SRC / "data" / "news_en"
NEWS_VI_DATA = config.DATA_SRC / "data" / "news_vi"

SYMBOLS = ["AAPL", "MSFT", "JPM", "XOM", "VCB", "HPG", "FPT"]


# ======================================================================
# CHUẨN HÓA TỪNG NGUỒN
# ======================================================================

def _safe_read(path):
    """Đọc CSV với fallback encoding."""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.DataFrame()


def _normalize_date(val):
    """Chuẩn hóa mọi format date về YYYY-MM-DD."""
    if pd.isna(val):
        return None
    val = str(val).strip()

    # AlphaVantage: 20260612T105806
    if len(val) >= 15 and "T" in val and "-" not in val[:8]:
        try:
            dt = datetime.strptime(val[:15], "%Y%m%dT%H%M%S")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Unix timestamp (Finnhub datetime field)
    try:
        ts = int(float(val))
        if ts > 1e9:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass

    # Pandas parse chung
    try:
        dt = pd.to_datetime(val, utc=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        dt = pd.to_datetime(val)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_alphavantage(path, symbol):
    """
    AlphaVantage: title, time_published, summary, overall_sentiment_score, ticker
    Đặc biệt: ĐÃ CÓ sentiment score sẵn.
    """
    df = _safe_read(path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    records_en = []
    records_sent = []  # Lưu sentiment score cho sau này
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        date = _normalize_date(row.get("time_published"))
        summary = str(row.get("summary", "")).strip()
        sent_score = row.get("overall_sentiment_score")

        if not title or not date:
            continue

        records_en.append({"title": title, "date": date, "stock": symbol})

        # Lưu sentiment score sẵn có
        if pd.notna(sent_score):
            records_sent.append({
                "Date": date,
                "Header": title,
                "Content": summary if summary and summary != "nan" else title,
                "sentiment_score": float(sent_score),
            })

    df_en = pd.DataFrame(records_en)
    df_sent = pd.DataFrame(records_sent)
    return df_en, df_sent


def normalize_extended(path, symbol, lang="en"):
    """
    Extended (Finnhub/CafeF): ticker, date, title, link, content
    Dùng cho cả EN và VI.
    """
    df = _safe_read(path)
    if df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        date = _normalize_date(row.get("date"))
        content = str(row.get("content", "")).strip()

        if not title or not date:
            continue

        # Loại bỏ nội dung rác
        if content in ["nan", "", "[Nội dung không khả dụng hoặc trang trống]"]:
            content = title

        if lang == "en":
            records.append({"title": title, "date": date, "stock": symbol})
        else:
            records.append({"Date": date, "Header": title, "Content": content})

    return pd.DataFrame(records)


def normalize_finnhub(path, symbol):
    """Finnhub raw: headline, date, summary, ticker."""
    df = _safe_read(path)
    if df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        title = str(row.get("headline", "")).strip()
        date = _normalize_date(row.get("date"))
        if not title or not date:
            continue
        records.append({"title": title, "date": date, "stock": symbol})

    return pd.DataFrame(records)


def normalize_newsapi(path, symbol):
    """NewsAPI: title, publishedAt, content, ticker."""
    df = _safe_read(path)
    if df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        date = _normalize_date(row.get("publishedAt"))
        if not title or not date or title == "nan":
            continue
        records.append({"title": title, "date": date, "stock": symbol})

    return pd.DataFrame(records)


# ======================================================================
# MERGE VÀ DE-DUPLICATE
# ======================================================================

def deduplicate_en(df):
    """Loại bỏ trùng EN theo title (lowercase) + date."""
    if df.empty:
        return df
    df["_key"] = df["title"].str.lower().str.strip() + "|" + df["date"].astype(str)
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    return df.sort_values("date").reset_index(drop=True)


def deduplicate_vi(df):
    """Loại bỏ trùng VI theo Header (lowercase) + Date."""
    if df.empty:
        return df
    df["_key"] = df["Header"].str.lower().str.strip() + "|" + df["Date"].astype(str)
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    return df.sort_values("Date").reset_index(drop=True)


# ======================================================================
# HÀM CHÍNH
# ======================================================================

def run():
    print("\n" + "=" * 70)
    print("  CHUAN HOA VA MERGE SCRAPED_NEWS VAO DATA_SRC")
    print("=" * 70)

    # Tao thu muc output
    for d in [NEWS_EN_RAW, NEWS_VI_RAW, NEWS_EN_DATA, NEWS_VI_DATA]:
        d.mkdir(parents=True, exist_ok=True)

    # Thu muc luu sentiment EN (tu AlphaVantage)
    sent_en_dir = config.DATA_SRC / "raw_clean" / "sentiment_en"
    sent_en_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for symbol in SYMBOLS:
        print(f"\n{'─' * 60}")
        print(f"  [{symbol}] Xu ly scraped_news")
        print(f"{'─' * 60}")

        # ===== EN NEWS =====
        en_frames = []
        sent_frames = []

        # AlphaVantage (co sentiment san)
        av_path = SCRAPED_DIR / f"{symbol}_en_news_alphavantage.csv"
        if av_path.exists():
            df_av_en, df_av_sent = normalize_alphavantage(av_path, symbol)
            en_frames.append(df_av_en)
            if not df_av_sent.empty:
                sent_frames.append(df_av_sent)
            print(f"  AlphaVantage EN: {len(df_av_en)} tin, {len(df_av_sent)} co sentiment")

        # Extended
        ext_en_path = SCRAPED_DIR / f"{symbol}_en_news_extended.csv"
        if ext_en_path.exists():
            df_ext = normalize_extended(ext_en_path, symbol, "en")
            en_frames.append(df_ext)
            print(f"  Extended EN:     {len(df_ext)} tin")

        # Finnhub
        fh_path = SCRAPED_DIR / f"{symbol}_en_news_finnhub.csv"
        if fh_path.exists():
            df_fh = normalize_finnhub(fh_path, symbol)
            en_frames.append(df_fh)
            print(f"  Finnhub EN:      {len(df_fh)} tin")

        # NewsAPI
        na_path = SCRAPED_DIR / f"{symbol}_en_news_newsapi.csv"
        if na_path.exists():
            df_na = normalize_newsapi(na_path, symbol)
            en_frames.append(df_na)
            print(f"  NewsAPI EN:      {len(df_na)} tin")

        # Merge EN: scraped + existing
        existing_en_path = NEWS_EN_RAW / f"{symbol}_en_news.csv"
        if existing_en_path.exists():
            df_existing = pd.read_csv(existing_en_path)
            en_frames.append(df_existing)
            print(f"  Existing EN:     {len(df_existing)} tin")

        if en_frames:
            all_en = pd.concat(en_frames, ignore_index=True)
            before = len(all_en)
            all_en = deduplicate_en(all_en)
            print(f"  -> Sau merge+dedup EN: {len(all_en)} tin (loai {before - len(all_en)} trung)")

            all_en.to_csv(NEWS_EN_RAW / f"{symbol}_en_news.csv", index=False, encoding="utf-8")
            all_en.to_csv(NEWS_EN_DATA / f"{symbol}_en_news.csv", index=False, encoding="utf-8")
        else:
            all_en = pd.DataFrame()

        # Luu sentiment EN tu AlphaVantage
        if sent_frames:
            all_sent_en = pd.concat(sent_frames, ignore_index=True)
            all_sent_en = all_sent_en.drop_duplicates(
                subset=["Header", "Date"], keep="first"
            ).sort_values("Date").reset_index(drop=True)
            all_sent_en.to_csv(sent_en_dir / f"{symbol}_sentiment_en.csv", index=False, encoding="utf-8")
            print(f"  -> Sentiment EN (AlphaVantage): {len(all_sent_en)} tin luu")

        # ===== VI NEWS =====
        vi_frames = []

        ext_vi_path = SCRAPED_DIR / f"{symbol}_vi_news_extended.csv"
        if ext_vi_path.exists():
            df_vi_ext = normalize_extended(ext_vi_path, symbol, "vi")
            vi_frames.append(df_vi_ext)
            print(f"  Extended VI:     {len(df_vi_ext)} tin")

        # Merge VI: scraped + existing
        existing_vi_path = NEWS_VI_RAW / f"{symbol}_vi_news.csv"
        if existing_vi_path.exists():
            df_existing_vi = pd.read_csv(existing_vi_path, encoding="utf-8-sig")
            # Chuan hoa ten cot
            col_map = {}
            for col in df_existing_vi.columns:
                lc = col.lower().strip()
                if lc == "date":
                    col_map[col] = "Date"
                elif lc == "header":
                    col_map[col] = "Header"
                elif lc == "content":
                    col_map[col] = "Content"
            df_existing_vi = df_existing_vi.rename(columns=col_map)
            if "Content" not in df_existing_vi.columns and "Header" in df_existing_vi.columns:
                df_existing_vi["Content"] = df_existing_vi["Header"]
            if set(["Date", "Header", "Content"]).issubset(df_existing_vi.columns):
                vi_frames.append(df_existing_vi[["Date", "Header", "Content"]])
                print(f"  Existing VI:     {len(df_existing_vi)} tin")

        if vi_frames:
            all_vi = pd.concat(vi_frames, ignore_index=True)
            before = len(all_vi)
            all_vi = deduplicate_vi(all_vi)
            print(f"  -> Sau merge+dedup VI: {len(all_vi)} tin (loai {before - len(all_vi)} trung)")

            all_vi.to_csv(NEWS_VI_RAW / f"{symbol}_vi_news.csv", index=False, encoding="utf-8-sig")
            all_vi.to_csv(NEWS_VI_DATA / f"{symbol}_vi_news.csv", index=False, encoding="utf-8-sig")
        else:
            all_vi = pd.DataFrame()

        summary[symbol] = {
            "en": len(all_en) if not all_en.empty else 0,
            "vi": len(all_vi) if not all_vi.empty else 0,
        }

    # ===== TONG KET =====
    print(f"\n{'=' * 70}")
    print("  TONG KET SAU MERGE")
    print(f"{'=' * 70}")
    print(f"  {'Symbol':<8} | {'EN':>8} | {'VI':>8} | {'TONG':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    total_en, total_vi = 0, 0
    for sym, cnt in sorted(summary.items()):
        t = cnt["en"] + cnt["vi"]
        print(f"  {sym:<8} | {cnt['en']:>8,} | {cnt['vi']:>8,} | {t:>8,}")
        total_en += cnt["en"]
        total_vi += cnt["vi"]
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'TONG':<8} | {total_en:>8,} | {total_vi:>8,} | {total_en+total_vi:>8,}")
    print(f"{'=' * 70}")

    print(f"\n  Output EN: {NEWS_EN_RAW}")
    print(f"  Output VI: {NEWS_VI_RAW}")
    print(f"  Sentiment EN (AlphaVantage): {sent_en_dir}")
    print(f"\n  Buoc tiep: Chay generate_sentiment cho VI moi -> re-train pipeline")


if __name__ == "__main__":
    run()
