"""
Tạo EMAT Dataset từ dữ liệu giá OHLCV.
Tính 7 Technical Indicators: RSI, MACD, Signal_Line, BB_High, BB_Low, ATR, K_Line.
Output: data_src/data/emat_datasets/{SYMBOL}_EMAT_dataset.csv

Sử dụng 100% pandas — không cần cài thêm thư viện technical analysis.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

NEW_SYMBOLS = ["AAPL", "MSFT", "JPM", "XOM", "VCB", "HPG", "FPT"]


# ============================================================
# Hàm tính Technical Indicators
# ============================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) theo công thức Wilder."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (EMA với alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD Line + Signal Line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands (Upper và Lower)."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bb_high = sma + num_std * std
    bb_low = sma - num_std * std
    return bb_high, bb_low


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR) theo Wilder."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic Oscillator %K."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    denominator = highest_high - lowest_low
    k_line = ((close - lowest_low) / denominator.replace(0, np.nan)) * 100
    return k_line


def generate_emat_dataset(symbol: str) -> pd.DataFrame:
    """
    Đọc prices CSV và tính toàn bộ indicators cho EMAT.
    Trả về DataFrame với 12 cột: Open, High, Low, Close, Volume + 7 indicators.
    """
    # Tìm file prices
    prices_path = config.PRICES_SRC / f"{symbol}_prices.csv"
    if not prices_path.exists():
        prices_path = config.RAW_CLEAN_PRICES / f"{symbol}_prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Không tìm thấy prices cho {symbol}: {prices_path}")

    df = pd.read_csv(prices_path)
    df = df.rename(columns={"time": "Date", "date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Tính 7 Technical Indicators
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal_Line"] = compute_macd(df["Close"])
    df["BB_High"], df["BB_Low"] = compute_bollinger_bands(df["Close"])
    df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"])
    df["K_Line"] = compute_stochastic_k(df["High"], df["Low"], df["Close"])

    # Forward-fill NaN ở đầu chuỗi (do rolling window chưa đủ dữ liệu)
    indicator_cols = ["RSI", "MACD", "Signal_Line", "BB_High", "BB_Low", "ATR", "K_Line"]
    df[indicator_cols] = df[indicator_cols].ffill().bfill()

    # Chọn đúng thứ tự cột theo format hiện có
    output_cols = ["Date"] + config.EMAT_FEATURE_COLS
    df = df[output_cols]

    return df


def run():
    """Tạo EMAT dataset cho tất cả mã mới."""
    print("\n" + "=" * 60)
    print("  TẠO EMAT DATASET (OHLCV + 7 Technical Indicators)")
    print("=" * 60)

    output_dir = config.EMAT_DATASETS_SRC
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in NEW_SYMBOLS:
        print(f"\n{'='*50}")
        print(f"  Đang xử lý: {symbol}")
        print(f"{'='*50}")

        try:
            df = generate_emat_dataset(symbol)
            output_path = output_dir / f"{symbol}_EMAT_dataset.csv"
            df.to_csv(output_path, index=False)

            print(f"  ✔ {symbol}: {len(df)} dòng")
            print(f"    Date range: {df['Date'].min()} → {df['Date'].max()}")
            print(f"    Columns: {list(df.columns)}")
            print(f"    Đã lưu: {output_path}")

            # Kiểm tra nhanh chất lượng
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print(f"    ⚠ Cảnh báo NaN: {null_counts[null_counts > 0].to_dict()}")
            else:
                print(f"    ✔ Không có NaN")

        except FileNotFoundError as e:
            print(f"  ✘ Bỏ qua {symbol}: {e}")
        except Exception as e:
            print(f"  ✘ Lỗi {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  ✔ HOÀN TẤT TẠO EMAT DATASET")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
