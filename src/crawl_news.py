"""
Cào tin tức tài chính cho 7 mã cổ phiếu từ nhiều nguồn.
=====================================================
Nguồn EN: Google News RSS, DuckDuckGo News, Yahoo Finance
Nguồn VI: Google News RSS (vi), DuckDuckGo News (vi)

Output EN: data_src/raw_clean/news_en/{SYMBOL}_en_news.csv  (title, date, stock)
Output VI: data_src/raw_clean/news_vi/{SYMBOL}_vi_news.csv  (Date, Header, Content)
"""
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import pandas as pd

# ======================================================================
# HẰNG SỐ
# ======================================================================

# Mapping symbol → ticker yfinance (tái sử dụng convention từ crawl_prices)
TICKER_MAP = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "JPM": "JPM",
    "XOM": "XOM",
    "VCB": "VCB.VN",
    "HPG": "HPG.VN",
    "FPT": "FPT.VN",
}

# Mapping symbol → tên công ty cho search queries
COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JPM": "JPMorgan",
    "XOM": "ExxonMobil",
    "VCB": "Vietcombank",
    "HPG": "Hòa Phát",
    "FPT": "FPT",
}

# Đa dạng hóa queries để cào nhiều tin hơn
EN_QUERY_TEMPLATES = [
    "{ticker} stock news",
    "{company} Inc stock market",
    "{ticker} earnings",
    "{company} stock price",
]

VI_QUERY_TEMPLATES = [
    "{company} cổ phiếu",
    "{ticker} chứng khoán",
    "{company} tin tức tài chính",
    "{company} giá cổ phiếu",
]

# Delay giữa mỗi request (giây) để tránh rate-limiting
REQUEST_DELAY = 1.5

# ======================================================================
# TIỆN ÍCH
# ======================================================================


def _normalize_date(date_str: str) -> Optional[str]:
    """
    Chuẩn hóa chuỗi ngày về YYYY-MM-DD.
    Hỗ trợ nhiều format phổ biến từ RSS/DuckDuckGo/Yahoo.
    """
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Thử parse trực tiếp bằng pandas (hỗ trợ rất nhiều format)
    try:
        dt = pd.to_datetime(date_str, utc=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # Fallback: thử các format phổ biến
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %Z",   # RSS standard
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def _strip_google_source(title: str) -> str:
    """
    Google News thường thêm ' - Source Name' ở cuối title.
    Loại bỏ phần này để giữ title sạch.
    """
    if not title:
        return ""
    # Tìm dấu " - " cuối cùng và cắt bỏ phần sau
    idx = title.rfind(" - ")
    if idx > 0:
        return title[:idx].strip()
    return title.strip()


def _deduplicate(records: List[Dict]) -> List[Dict]:
    """Loại bỏ bản ghi trùng dựa trên title + date."""
    seen = set()
    unique = []
    for rec in records:
        key = (rec.get("title", "").lower().strip(), rec.get("date", ""))
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ======================================================================
# NGUỒN 1: GOOGLE NEWS RSS
# ======================================================================


def _crawl_google_rss(query: str, lang: str = "en") -> List[Dict]:
    """
    Cào tin từ Google News RSS feed.
    lang='en' → hl=en-US&gl=US&ceid=US:en
    lang='vi' → hl=vi&gl=VN&ceid=VN:vi
    """
    try:
        import feedparser
    except ImportError:
        print("  ⚠ feedparser chưa cài. Chạy: pip install feedparser")
        return []

    if lang == "vi":
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=vi&gl=VN&ceid=VN:vi"
    else:
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

    results = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = _strip_google_source(entry.get("title", ""))
            published = entry.get("published", "")
            date = _normalize_date(published)
            if title and date:
                results.append({"title": title, "date": date})
    except Exception as e:
        print(f"  ⚠ Google RSS lỗi ({query}): {e}")

    return results


# ======================================================================
# NGUỒN 2: DUCKDUCKGO NEWS
# ======================================================================


def _crawl_ddg_news(query: str, max_results: int = 50) -> List[Dict]:
    """
    Cào tin từ DuckDuckGo News API.
    Mỗi result có: date, title, body, url, source.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("  ⚠ duckduckgo-search chưa cài. Chạy: pip install duckduckgo-search")
        return []

    results = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.news(query, max_results=max_results))
        for item in raw:
            title = item.get("title", "").strip()
            date = _normalize_date(item.get("date", ""))
            body = item.get("body", "").strip()
            if title and date:
                results.append({"title": title, "date": date, "body": body})
    except Exception as e:
        print(f"  ⚠ DuckDuckGo lỗi ({query}): {e}")

    return results


# ======================================================================
# NGUỒN 3: YAHOO FINANCE (.news attribute)
# ======================================================================


def _crawl_yahoo_news(yf_ticker: str) -> List[Dict]:
    """
    Lấy tin từ Yahoo Finance qua yfinance Ticker.news.
    Mỗi item có: title, publisher, link, providerPublishTime, ...
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠ yfinance chưa cài. Chạy: pip install yfinance")
        return []

    results = []
    try:
        ticker = yf.Ticker(yf_ticker)
        news_items = ticker.news or []
        for item in news_items:
            title = item.get("title", "").strip()
            # yfinance >= 1.2.x: providerPublishTime hoặc publishedDate
            ts = item.get("providerPublishTime") or item.get("publishedDate")
            date = None
            if ts:
                try:
                    # Unix timestamp
                    date = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    # Có thể là chuỗi ngày
                    date = _normalize_date(str(ts))
            if not date:
                # Thử trích xuất từ các trường khác
                for date_field in ["date", "published", "relatedTickers"]:
                    if date_field in item and item[date_field]:
                        date = _normalize_date(str(item[date_field]))
                        if date:
                            break
            if title and date:
                results.append({"title": title, "date": date})
    except Exception as e:
        print(f"  ⚠ Yahoo Finance lỗi ({yf_ticker}): {e}")

    return results


# ======================================================================
# CÀO TIN TIẾNG ANH
# ======================================================================


def crawl_en_news(symbol: str) -> pd.DataFrame:
    """
    Cào tin tiếng Anh cho 1 mã cổ phiếu từ 3 nguồn, có fallback.
    Trả về DataFrame(title, date, stock).
    """
    company = COMPANY_NAMES[symbol]
    yf_ticker = TICKER_MAP[symbol]
    all_records = []

    # Nguồn 1: Google News RSS
    print(f"    [Google RSS EN] Đang cào...")
    for tmpl in EN_QUERY_TEMPLATES:
        query = tmpl.format(ticker=symbol, company=company)
        records = _crawl_google_rss(query, lang="en")
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)
    print(f"    [Google RSS EN] → {len(all_records)} tin")

    # Nguồn 2: DuckDuckGo News
    count_before = len(all_records)
    print(f"    [DuckDuckGo EN] Đang cào...")
    for tmpl in EN_QUERY_TEMPLATES[:2]:  # Chỉ dùng 2 queries chính để tránh quá tải
        query = tmpl.format(ticker=symbol, company=company)
        records = _crawl_ddg_news(query, max_results=50)
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)
    print(f"    [DuckDuckGo EN] → {len(all_records) - count_before} tin mới")

    # Nguồn 3: Yahoo Finance
    count_before = len(all_records)
    print(f"    [Yahoo Finance] Đang cào...")
    records = _crawl_yahoo_news(yf_ticker)
    all_records.extend(records)
    print(f"    [Yahoo Finance] → {len(records)} tin")

    # De-duplicate
    unique = _deduplicate(all_records)
    print(f"    Tổng sau de-dup: {len(unique)} tin (loại {len(all_records) - len(unique)} trùng)")

    if not unique:
        return pd.DataFrame(columns=["title", "date", "stock"])

    df = pd.DataFrame(unique)
    df["stock"] = symbol
    df = df[["title", "date", "stock"]].sort_values("date").reset_index(drop=True)

    return df


# ======================================================================
# CÀO TIN TIẾNG VIỆT
# ======================================================================


def crawl_vi_news(symbol: str) -> pd.DataFrame:
    """
    Cào tin tiếng Việt cho 1 mã cổ phiếu.
    Trả về DataFrame(Date, Header, Content).
    """
    company = COMPANY_NAMES[symbol]
    all_records = []

    # Nguồn 1: Google News RSS tiếng Việt
    print(f"    [Google RSS VI] Đang cào...")
    for tmpl in VI_QUERY_TEMPLATES:
        query = tmpl.format(ticker=symbol, company=company)
        records = _crawl_google_rss(query, lang="vi")
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)
    print(f"    [Google RSS VI] → {len(all_records)} tin")

    # Nguồn 2: DuckDuckGo News tiếng Việt
    count_before = len(all_records)
    print(f"    [DuckDuckGo VI] Đang cào...")
    for tmpl in VI_QUERY_TEMPLATES[:2]:
        query = tmpl.format(ticker=symbol, company=company)
        records = _crawl_ddg_news(query, max_results=50)
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)
    print(f"    [DuckDuckGo VI] → {len(all_records) - count_before} tin mới")

    # De-duplicate
    unique = _deduplicate(all_records)
    print(f"    Tổng sau de-dup: {len(unique)} tin (loại {len(all_records) - len(unique)} trùng)")

    if not unique:
        return pd.DataFrame(columns=["Date", "Header", "Content"])

    # Chuyển sang format VI: Date, Header, Content
    rows = []
    for rec in unique:
        header = rec.get("title", "")
        # DuckDuckGo có body → dùng làm Content; nếu không → dùng title
        content = rec.get("body", "") or header
        rows.append({
            "Date": rec["date"],
            "Header": header,
            "Content": content,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ======================================================================
# HÀM CHÍNH
# ======================================================================


def run():
    """Hàm chính: cào tin EN + VI cho tất cả mã cổ phiếu."""
    print("\n" + "=" * 60)
    print("  CÀO TIN TỨC TÀI CHÍNH (EN + VI)")
    print("=" * 60)

    symbols = list(TICKER_MAP.keys())
    print(f"  Mã cổ phiếu: {symbols}")

    # Tạo thư mục output
    en_dir = config.RAW_CLEAN_NEWS_EN
    vi_dir = config.RAW_CLEAN_NEWS_VI
    en_dir.mkdir(parents=True, exist_ok=True)
    vi_dir.mkdir(parents=True, exist_ok=True)

    # Cũng tạo thư mục data/ (mirror)
    data_en_dir = config.NEWS_EN_SRC
    data_vi_dir = config.NEWS_VI_SRC
    data_en_dir.mkdir(parents=True, exist_ok=True)
    data_vi_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  [{symbol}] ({COMPANY_NAMES[symbol]})")
        print(f"{'─' * 50}")

        # === CÀO TIN TIẾNG ANH ===
        print(f"\n  📰 Cào tin tiếng Anh...")
        try:
            df_en = crawl_en_news(symbol)
            en_file = en_dir / f"{symbol}_en_news.csv"
            df_en.to_csv(en_file, index=False, encoding="utf-8")
            # Mirror sang data/
            df_en.to_csv(data_en_dir / f"{symbol}_en_news.csv", index=False, encoding="utf-8")
            print(f"  ✔ EN: {len(df_en)} tin → {en_file}")
        except Exception as e:
            print(f"  ✘ EN lỗi: {e}")
            df_en = pd.DataFrame()

        # === CÀO TIN TIẾNG VIỆT ===
        print(f"\n  📰 Cào tin tiếng Việt...")
        try:
            df_vi = crawl_vi_news(symbol)
            vi_file = vi_dir / f"{symbol}_vi_news.csv"
            df_vi.to_csv(vi_file, index=False, encoding="utf-8-sig")
            # Mirror sang data/
            df_vi.to_csv(data_vi_dir / f"{symbol}_vi_news.csv", index=False, encoding="utf-8-sig")
            print(f"  ✔ VI: {len(df_vi)} tin → {vi_file}")
        except Exception as e:
            print(f"  ✘ VI lỗi: {e}")
            df_vi = pd.DataFrame()

        summary[symbol] = {"en": len(df_en), "vi": len(df_vi)}

    # Báo cáo tổng kết
    print(f"\n{'=' * 60}")
    print("  TỔNG KẾT CÀO TIN TỨC")
    print(f"{'=' * 60}")
    print(f"  {'Symbol':<8} | {'EN':>6} | {'VI':>6}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}")
    total_en, total_vi = 0, 0
    for symbol, counts in summary.items():
        print(f"  {symbol:<8} | {counts['en']:>6} | {counts['vi']:>6}")
        total_en += counts["en"]
        total_vi += counts["vi"]
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}")
    print(f"  {'TỔNG':<8} | {total_en:>6} | {total_vi:>6}")
    print(f"{'=' * 60}")

    print(f"\n  📂 Output EN: {en_dir}")
    print(f"  📂 Output VI: {vi_dir}")

    return summary


if __name__ == "__main__":
    run()
