"""
Cào dữ liệu giá OHLCV cho 7 mã chứng khoán mới từ yfinance.
Output: data_src/raw_clean/prices/{SYMBOL}_prices.csv
        data_src/data/prices/{SYMBOL}_prices.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Mapping: tên symbol nội bộ → ticker yfinance
NEW_TICKERS = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "JPM": "JPM",
    "XOM": "XOM",
    "VCB": "VCB.VN",
    "HPG": "HPG.VN",
    "FPT": "FPT.VN",
}

START_DATE = "2009-01-01"
END_DATE = "2025-12-31"


def crawl_all_prices():
    """Cào giá OHLCV cho tất cả mã mới và lưu vào data_src."""
    try:
        import yfinance as yf
    except ImportError:
        print("❌ Chưa cài yfinance. Chạy: pip install yfinance")
        sys.exit(1)

    # Tạo thư mục output
    raw_clean_dir = config.RAW_CLEAN_PRICES
    data_dir = config.PRICES_SRC
    raw_clean_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for symbol, yf_ticker in NEW_TICKERS.items():
        print(f"\n{'='*50}")
        print(f"  Đang cào: {symbol} (yfinance: {yf_ticker})")
        print(f"  Khoảng thời gian: {START_DATE} → {END_DATE}")
        print(f"{'='*50}")

        try:
            df = yf.download(yf_ticker, start=START_DATE, end=END_DATE, progress=False)

            if df.empty:
                print(f"  ⚠ Không có dữ liệu cho {symbol}")
                continue

            # yfinance >= 0.2.x trả về MultiIndex columns cho single ticker
            # Cần flatten nếu có multi-level
            if isinstance(df.columns, __import__('pandas').MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Chuẩn hóa cột theo format hiện có (Date, Open, High, Low, Close, Volume)
            cols_needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
            # Loại bỏ Adj Close nếu có
            df = df[[c for c in cols_needed if c in df.columns]]

            # Đảm bảo thứ tự cột chuẩn
            df = df.sort_values("Date").reset_index(drop=True)

            # Lưu vào cả 2 thư mục (raw_clean và data)
            filename = f"{symbol}_prices.csv"
            df.to_csv(raw_clean_dir / filename, index=False)
            df.to_csv(data_dir / filename, index=False)

            date_min = df["Date"].min()
            date_max = df["Date"].max()
            results[symbol] = {"rows": len(df), "start": str(date_min)[:10], "end": str(date_max)[:10]}

            print(f"  ✔ {symbol}: {len(df)} dòng | {date_min} → {date_max}")
            print(f"    Đã lưu: {raw_clean_dir / filename}")
            print(f"    Đã lưu: {data_dir / filename}")

        except Exception as e:
            print(f"  ✘ Lỗi khi cào {symbol}: {e}")
            results[symbol] = {"error": str(e)}

    # Báo cáo tổng kết
    print(f"\n{'='*60}")
    print("  TỔNG KẾT CÀO GIÁ")
    print(f"{'='*60}")
    print(f"  {'Symbol':<8} | {'Rows':>6} | {'Start':>12} | {'End':>12}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for symbol, info in results.items():
        if "error" in info:
            print(f"  {symbol:<8} | {'ERROR':>6} | {info['error']}")
        else:
            print(f"  {symbol:<8} | {info['rows']:>6} | {info['start']:>12} | {info['end']:>12}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    crawl_all_prices()
