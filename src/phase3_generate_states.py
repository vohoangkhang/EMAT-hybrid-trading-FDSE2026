"""
Phase 3 - State Generator: Tổng hợp đặc trưng từ 4 mô hình
=============================================================
Load EMAT, HMM, GARCH (và Sentiment nếu có) → sinh state vectors cho RL env.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from phase2_train_emat import EMAT
from phase2_train_hmm import prepare_hmm_features
from phase2_train_sentiment import XLMRoBERTa_CNN_BiGRU


def load_all_models():
    """Load tất cả models đã train."""
    device = config.DEVICE
    models = {}

    # EMAT
    emat = EMAT().to(device)
    emat_path = config.MODELS_DIR / "emat_model.pth"
    if emat_path.exists():
        emat.load_state_dict(torch.load(emat_path, map_location=device))
        emat.eval()
        models["emat"] = emat
        print("  ✔ EMAT loaded")
    else:
        print("  ✘ EMAT model not found")

    # HMM
    hmm_path = config.MODELS_DIR / "hmm_model.pkl"
    if hmm_path.exists():
        with open(hmm_path, "rb") as f:
            models["hmm"] = pickle.load(f)
        print("  ✔ HMM loaded")
    else:
        print("  ✘ HMM model not found")

    # GARCH
    garch_path = config.MODELS_DIR / "garch_model.pkl"
    if garch_path.exists():
        with open(garch_path, "rb") as f:
            models["garch"] = pickle.load(f)
        print("  ✔ GARCH loaded")
    else:
        print("  ✘ GARCH model not found")

    # Sentiment
    sentiment_path = config.MODELS_DIR / "sentiment_model.pth"
    if sentiment_path.exists():
        try:
            from transformers import AutoTokenizer
            sentiment_model = XLMRoBERTa_CNN_BiGRU().to(device)
            sentiment_model.load_state_dict(torch.load(sentiment_path, map_location=device))
            sentiment_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_BACKBONE)
            models["sentiment"] = {"model": sentiment_model, "tokenizer": tokenizer}
            print("  ✔ Sentiment loaded")
        except Exception as e:
            print(f"  ✘ Sentiment model error: {e}")
    else:
        print("  ✘ Sentiment model not found")

    return models


def generate_emat_features(model, df_emat, device, symbol):
    """Chạy EMAT inference → softmax probabilities cho mỗi horizon."""
    features = df_emat[config.EMAT_FEATURE_COLS].values.astype(np.float32)
    
    train_path = config.DATA_DIR / "train" / f"{symbol}_emat.csv"
    if train_path.exists():
        df_train = pd.read_csv(train_path)
        train_feats = df_train[config.EMAT_FEATURE_COLS].values.astype(np.float32)
        mean = np.nanmean(train_feats, axis=0)
        std = np.nanstd(train_feats, axis=0) + 1e-8
    else:
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0) + 1e-8
        
    features = (features - mean) / std

    window = config.EMAT_WINDOW_SIZE
    all_returns = []  # Mỗi phần tử là dict {horizon: return_predicted}

    model.eval()
    with torch.no_grad():
        for i in range(len(features) - window + 1):
            x = torch.tensor(features[i:i + window], dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(x)
            returns = {}
            for h in config.PREDICTION_HORIZONS:
                r = outputs[f"{h}d"].cpu().numpy()[0]  # (1,)
                returns[h] = r
            all_returns.append(returns)

    # Padding: window-1 hàng đầu tiên không có prediction → dùng 0.0
    padding = [
        {h: 0.0 for h in config.PREDICTION_HORIZONS}
        for _ in range(window - 1)
    ]
    all_returns = padding + all_returns

    # Flatten thành array: [Return_1d, Return_3d, Return_5d]
    result = np.zeros((len(all_returns), len(config.PREDICTION_HORIZONS)))
    for i, rets in enumerate(all_returns):
        for j, h in enumerate(config.PREDICTION_HORIZONS):
            result[i, j] = rets[h]
    return result  # shape: (N, 3)


def generate_hmm_features(hmm_models, df_prices, symbol):
    """Chạy HMM inference → regime one-hot + transition probs."""
    if symbol not in hmm_models:
        n = len(df_prices)
        return np.zeros((n, 6))  # 3 one-hot + 3 transition

    model_data = hmm_models[symbol]
    model = model_data["model"]
    regime_map = model_data["regime_map"]

    df = df_prices.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    
    # Dynamic Smoothing
    if symbol == "VNM":
        df["Return_Smooth"] = df["Return"]
        df["Vol_Smooth"] = df["Volatility"]
    else:
        df["Return_Smooth"] = df["Return"].ewm(span=config.HMM_SMOOTHING_WINDOW).mean()
        df["Vol_Smooth"] = df["Volatility"].ewm(span=config.HMM_SMOOTHING_WINDOW).mean()

    # Các hàng đầu không có đủ data → padding
    result = np.zeros((len(df_prices), 6))
    valid_mask = ~(df["Return_Smooth"].isna() | df["Vol_Smooth"].isna())
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return result

    X = np.column_stack([df.loc[valid_mask, "Return_Smooth"].values, df.loc[valid_mask, "Vol_Smooth"].values])
    states = model.predict(X)
    state_probs = model.predict_proba(X)

    for idx_out, idx_in in enumerate(valid_indices):
        # One-hot regime (3 dims)
        n_comp = model.n_components
        one_hot = np.zeros(3)
        state = states[idx_out]
        regime_label = regime_map.get(state, "Sideways")
        regime_idx = {"Bearish": 0, "Sideways": 1, "Bullish": 2}.get(regime_label, 1)
        one_hot[regime_idx] = 1.0
        result[idx_in, :3] = one_hot

        # Transition probabilities (top 3)
        if idx_out < len(state_probs):
            probs = state_probs[idx_out]
            # Aggregate sang 3 regime categories
            regime_probs = np.zeros(3)
            for s_idx in range(n_comp):
                r_label = regime_map.get(s_idx, "Sideways")
                r_idx = {"Bearish": 0, "Sideways": 1, "Bullish": 2}.get(r_label, 1)
                regime_probs[r_idx] += probs[s_idx] if s_idx < len(probs) else 0
            result[idx_in, 3:6] = regime_probs

    return result  # shape: (N, 6)


def generate_garch_features(garch_models, df_prices, symbol):
    """Chạy GARCH inference → conditional volatility + category one-hot (Rolling Z-score)."""
    from arch import arch_model as arch_m
    result = np.zeros((len(df_prices), 4))  # 1 vol_norm + 3 one-hot (Thấp/TB/Cao)

    if symbol not in garch_models:
        return result

    model_data = garch_models[symbol]

    df = df_prices.copy()
    df["Return"] = df["Close"].pct_change() * config.GARCH_RETURNS_SCALE
    df = df.dropna(subset=["Return"])

    if len(df) < 10:
        return result

    try:
        am = arch_m(df["Return"], vol="Garch", p=config.GARCH_P, q=config.GARCH_Q, dist=config.GARCH_DIST)
        res = am.fit(disp="off", starting_values=model_data["result"].params.values)
        cond_vol = res.conditional_volatility

        # Chuẩn hóa volatility về [0, 1]
        vol_min, vol_max = cond_vol.min(), cond_vol.max()
        vol_norm = (cond_vol - vol_min) / (vol_max - vol_min + 1e-8)

        # Rolling Z-score — tự tính trên chính chuỗi hiện tại
        window = config.GARCH_VOL_ROLLING_WINDOW
        rolling_mean = cond_vol.rolling(window=window, min_periods=10).mean()
        rolling_std = cond_vol.rolling(window=window, min_periods=10).std()
        zscore = (cond_vol - rolling_mean) / (rolling_std + 1e-8)

        # Map lại vào index gốc
        valid_indices = df.index
        for i, idx in enumerate(valid_indices):
            if idx < len(result):
                result[idx, 0] = vol_norm.iloc[i]
                z = zscore.iloc[i] if not np.isnan(zscore.iloc[i]) else 0.0
                if z > config.GARCH_VOL_ZSCORE_HIGH:
                    result[idx, 3] = 1.0  # Cao
                elif z < config.GARCH_VOL_ZSCORE_LOW:
                    result[idx, 1] = 1.0  # Thấp
                else:
                    result[idx, 2] = 1.0  # Trung bình
    except Exception as e:
        print(f"    GARCH inference error for {symbol}: {e}")

    return result  # shape: (N, 4)


def generate_sentiment_features(model_data, df_prices, symbol, split):
    """Chạy Sentiment inference → 3 probability logits (Tích cực, Tiêu cực, Trung tính)."""
    device = config.DEVICE
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    n = len(df_prices)
    result = np.zeros((n, 3))
    result.fill(1/3)  # Mặc định trung lập
    
    # Đọc tin tức
    vi_path = config.DATA_DIR / split / f"{symbol}_sentiment_vi.csv"
    if not vi_path.exists():
        return result
        
    df_news = pd.read_csv(vi_path)
    text_col = "Content" if "Content" in df_news.columns else "Header"
    if text_col not in df_news.columns:
        return result
        
    df_news["Date"] = pd.to_datetime(df_news["Date" if "Date" in df_news.columns else "date"]).dt.date
    df_news = df_news.dropna(subset=[text_col])
    
    if len(df_news) == 0:
        return result
        
    # Chạy inference theo batch để tránh OOM
    batch_size = config.SENTIMENT_BATCH_SIZE
    texts = df_news[text_col].astype(str).str[:1000].tolist()
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=config.SENTIMENT_MAX_LENGTH, return_tensors="pt"
            )
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            
    df_news["prob_0"] = [p[0] for p in all_probs]
    df_news["prob_1"] = [p[1] for p in all_probs]
    df_news["prob_2"] = [p[2] for p in all_probs]
    
    # Tính trung bình theo ngày
    daily_probs = df_news.groupby("Date")[["prob_0", "prob_1", "prob_2"]].mean()
    prob_dict = daily_probs.to_dict(orient="index")
    
    # Map vào df_prices
    for idx, date in enumerate(df_prices["Date"].dt.date):
        if date in prob_dict:
            p = prob_dict[date]
            result[idx] = [p["prob_0"], p["prob_1"], p["prob_2"]]
            
    return result


def generate_states_for_split(split, models):
    """Sinh state vectors cho một split (train/val/test)."""
    device = config.DEVICE
    all_observations = []
    all_prices = []
    all_dates = []
    all_symbols = []

    for symbol in config.SYMBOLS:
        print(f"    {symbol}...", end=" ")

        # Load data
        prices_path = config.DATA_DIR / split / f"{symbol}_prices.csv"
        emat_path = config.DATA_DIR / split / f"{symbol}_emat.csv"

        if not prices_path.exists() or not emat_path.exists():
            print("SKIP (no data)")
            continue

        df_prices = pd.read_csv(prices_path)
        df_prices["Date"] = pd.to_datetime(df_prices["Date"])
        df_emat = pd.read_csv(emat_path)
        df_emat["Date"] = pd.to_datetime(df_emat["Date"])

        n = len(df_prices)

        # 1. OHLCV normalized (5 dims)
        ohlcv = df_prices[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
        # Lấy mean/std từ tập Train để tránh Data Leakage
        train_prices_path = config.DATA_DIR / "train" / f"{symbol}_prices.csv"
        if train_prices_path.exists():
            df_prices_train = pd.read_csv(train_prices_path)
            ohlcv_train = df_prices_train[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
            mean_ohlcv = np.nanmean(ohlcv_train, axis=0)
            std_ohlcv = np.nanstd(ohlcv_train, axis=0) + 1e-8
        else:
            mean_ohlcv = np.nanmean(ohlcv, axis=0)
            std_ohlcv = np.nanstd(ohlcv, axis=0) + 1e-8
            
        ohlcv_norm = (ohlcv - mean_ohlcv) / std_ohlcv

        # 2. Technical Indicators from EMAT dataset (7 dims)
        tech_cols = ["RSI", "MACD", "Signal_Line", "BB_High", "BB_Low", "ATR", "K_Line"]
        # Align emat dates with prices dates
        emat_merged = pd.merge(
            df_prices[["Date"]], df_emat[["Date"] + tech_cols],
            on="Date", how="left"
        )
        tech_features = emat_merged[tech_cols].fillna(0).values.astype(np.float32)
        
        train_emat_path = config.DATA_DIR / "train" / f"{symbol}_emat.csv"
        if train_emat_path.exists():
            df_emat_train = pd.read_csv(train_emat_path)
            tech_train = df_emat_train[tech_cols].fillna(0).values.astype(np.float32)
            mean_tech = np.nanmean(tech_train, axis=0)
            std_tech = np.nanstd(tech_train, axis=0) + 1e-8
        else:
            mean_tech = np.nanmean(tech_features, axis=0)
            std_tech = np.nanstd(tech_features, axis=0) + 1e-8
            
        tech_norm = (tech_features - mean_tech) / std_tech

        # 3. EMAT features (3 dims: return 1d, 3d, 5d)
        emat_feats = np.zeros((n, 3))
        if "emat" in models:
            # Align emat dataset
            df_emat_aligned = pd.merge(df_prices[["Date"]], df_emat, on="Date", how="left").ffill()
            if len(df_emat_aligned) >= config.EMAT_WINDOW_SIZE:
                emat_feats_raw = generate_emat_features(models["emat"], df_emat_aligned, config.DEVICE, symbol)
                emat_feats[:len(emat_feats_raw)] = emat_feats_raw[:n]

        # 4. HMM features (6 dims: 3 one-hot + 3 transition probs)
        hmm_feats = np.zeros((n, 6))
        if "hmm" in models:
            hmm_feats = generate_hmm_features(models["hmm"], df_prices, symbol)

        # 5. GARCH features (4 dims: 1 vol + 3 one-hot)
        garch_feats = np.zeros((n, 4))
        if "garch" in models:
            garch_feats = generate_garch_features(models["garch"], df_prices, symbol)

        # 6. Sentiment features (3 dims: 3 probabilities)
        sentiment_feats = np.zeros((n, 3))
        sentiment_feats.fill(1/3)
        if "sentiment" in models:
            sentiment_feats = generate_sentiment_features(models["sentiment"], df_prices, symbol, split)

        # Tổng hợp state vector: 5 + 7 + 3 + 6 + 4 + 3 = 28 dims (không tính portfolio)
        state = np.hstack([ohlcv_norm, tech_norm, emat_feats, hmm_feats, garch_feats, sentiment_feats])

        # Xử lý NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        all_observations.append(state)
        all_prices.append(df_prices["Close"].values)
        all_dates.append(df_prices["Date"].values)
        all_symbols.extend([symbol] * n)
        print(f"OK ({n} days, state_dim={state.shape[1]})")

    if not all_observations:
        print("    [ERROR] Không có observations nào được tạo!")
        return

    # Lưu observations
    observations = np.concatenate(all_observations, axis=0)
    prices = np.concatenate(all_prices, axis=0)
    dates = np.concatenate(all_dates, axis=0)

    np.save(config.STATES_DIR / f"{split}_observations.npy", observations)
    np.save(config.STATES_DIR / f"{split}_prices.npy", prices)
    np.save(config.STATES_DIR / f"{split}_dates.npy", dates)

    # Lưu symbol mapping
    symbol_arr = np.array(all_symbols)
    np.save(config.STATES_DIR / f"{split}_symbols.npy", symbol_arr)

    print(f"    Tổng: {observations.shape[0]} samples, {observations.shape[1]} dims")
    return observations


def run():
    print("\n" + "=" * 60)
    print("  PHASE 3: SINH STATE VECTORS")
    print("=" * 60)

    print("\n  Loading models...")
    models = load_all_models()

    for split in ["train", "val", "test"]:
        print(f"\n  --- {split.upper()} ---")
        generate_states_for_split(split, models)

    print(f"\n  ✔ States đã lưu tại: {config.STATES_DIR}")
    # Verify
    for split in ["train", "val", "test"]:
        path = config.STATES_DIR / f"{split}_observations.npy"
        if path.exists():
            obs = np.load(path)
            has_nan = np.any(np.isnan(obs))
            has_inf = np.any(np.isinf(obs))
            print(f"    {split}: shape={obs.shape}, NaN={has_nan}, Inf={has_inf}")


if __name__ == "__main__":
    run()
