"""
Phase 2 - Task 2.3: GARCH Volatility Forecasting
=================================================
GARCH(1,1) dự báo conditional volatility.
Phân loại volatility bằng Rolling Z-score (thay vì fixed threshold).
"""
import numpy as np
import pandas as pd
from arch import arch_model
import pickle
import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def fit_garch_model(returns):
    am = arch_model(returns, vol="Garch", p=config.GARCH_P, q=config.GARCH_Q, dist=config.GARCH_DIST)
    return am.fit(disp="off")


def categorize_volatility_zscore(vol_series):
    """
    Phân loại volatility bằng Rolling Z-score.
    Tự tính mean/std từ rolling window → linh hoạt theo từng giai đoạn thị trường,
    tránh lỗi 100% "Cao" khi phân phối test khác train.
    """
    window = config.GARCH_VOL_ROLLING_WINDOW

    rolling_mean = vol_series.rolling(window=window, min_periods=10).mean()
    rolling_std = vol_series.rolling(window=window, min_periods=10).std()

    # Z-score = (giá trị hiện tại - mean) / std
    zscore = (vol_series - rolling_mean) / (rolling_std + 1e-8)

    # Phân loại dựa trên ngưỡng Z-score
    labels = zscore.apply(
        lambda z: "Cao" if z > config.GARCH_VOL_ZSCORE_HIGH
        else ("Thấp" if z < config.GARCH_VOL_ZSCORE_LOW else "Trung bình")
    )

    # Các hàng đầu chưa đủ rolling window → gán "Trung bình" (mặc định an toàn)
    labels = labels.fillna("Trung bình")

    return labels, zscore


def train():
    print("\n" + "=" * 60)
    print("  PHASE 2.3: HUẤN LUYỆN GARCH (Volatility Forecasting)")
    print("=" * 60)

    models = {}
    for symbol in config.SYMBOLS:
        print(f"\n  --- {symbol} ---")
        path = config.DATA_DIR / "train" / f"{symbol}_prices.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Return"] = df["Close"].pct_change() * config.GARCH_RETURNS_SCALE
        df = df.dropna(subset=["Return"])

        result = fit_garch_model(df["Return"])
        df["Cond_Vol"] = result.conditional_volatility
        df["Real_Vol"] = df["Return"].rolling(5).std()

        # Phân loại bằng Rolling Z-score
        vol_labels, zscore = categorize_volatility_zscore(df["Cond_Vol"])

        valid = df.dropna(subset=["Real_Vol"])
        rmse = np.sqrt(np.mean((valid["Cond_Vol"] - valid["Real_Vol"]) ** 2))

        print(f"  Samples: {len(df)} | RMSE: {rmse:.4f} | AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")
        print(f"  Params: omega={result.params.get('omega',0):.6f}, "
              f"alpha={result.params.get('alpha[1]',0):.6f}, beta={result.params.get('beta[1]',0):.6f}")
        print(f"  Vol distribution (Z-score): {vol_labels.value_counts().to_dict()}")

        # Không lưu quantiles — Rolling Z-score tự tính trên mỗi split
        models[symbol] = {"result": result}

    with open(config.MODELS_DIR / "garch_model.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"\n  ✔ GARCH models đã lưu tại: {config.MODELS_DIR / 'garch_model.pkl'}")
    return models


def evaluate():
    print("\n  ĐÁNH GIÁ GARCH MODEL")
    with open(config.MODELS_DIR / "garch_model.pkl", "rb") as f:
        models = pickle.load(f)

    for split in ["val", "test"]:
        for symbol in config.SYMBOLS:
            if symbol not in models:
                continue
            path = config.DATA_DIR / split / f"{symbol}_prices.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            df["Return"] = df["Close"].pct_change() * config.GARCH_RETURNS_SCALE
            df = df.dropna(subset=["Return"])

            try:
                am = arch_model(df["Return"], vol="Garch", p=config.GARCH_P, q=config.GARCH_Q, dist=config.GARCH_DIST)
                res = am.fit(disp="off", starting_values=models[symbol]["result"].params.values)
                df["Cond_Vol"] = res.conditional_volatility
                df["Real_Vol"] = df["Return"].rolling(5).std()
                valid = df.dropna(subset=["Real_Vol"])
                rmse = np.sqrt(np.mean((valid["Cond_Vol"] - valid["Real_Vol"]) ** 2))

                # Rolling Z-score — tự tính trên chính split hiện tại
                vol_labels, _ = categorize_volatility_zscore(df["Cond_Vol"])
                df["Vol_Label"] = vol_labels

                dist = df["Vol_Label"].value_counts().to_dict()
                print(f"  {symbol} ({split}): RMSE={rmse:.4f} | {dist}")
                with open(config.METRICS_DIR / f"garch_{split}_{symbol}.json", "w", encoding="utf-8") as f:
                    json.dump({"rmse": float(rmse), "dist": dist}, f, indent=2)
            except Exception as e:
                print(f"  {symbol} ({split}): ERROR - {e}")


def run():
    train()
    evaluate()

if __name__ == "__main__":
    run()
