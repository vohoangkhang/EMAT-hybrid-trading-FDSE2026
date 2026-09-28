"""
Phase 2 - Task 2.2: Huấn luyện HMM (Hidden Markov Model)
=========================================================
Phân cụm không giám sát thành các Market Regime (Bull/Bear/Sideways).
Auto-select số regime tốt nhất bằng BIC.
Input: Returns + Volatility (2D).
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
import pickle
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import advanced_metrics as am


def prepare_hmm_features(df_prices: pd.DataFrame, symbol: str) -> np.ndarray:
    """
    Tạo features cho HMM: daily returns + rolling volatility.
    Trả về mảng 2D (N, 2).
    """
    df = df_prices.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=5).std()
    
    # Dynamic Smoothing
    if symbol == "VNM":
        # Tháo kính râm: Dùng dữ liệu thô hoàn toàn cho VNM
        df["Return_Smooth"] = df["Return"]
        df["Vol_Smooth"] = df["Volatility"]
    else:
        # GOOGL và META giữ nguyên công thức chuẩn
        df["Return_Smooth"] = df["Return"].ewm(span=config.HMM_SMOOTHING_WINDOW).mean()
        df["Vol_Smooth"] = df["Volatility"].ewm(span=config.HMM_SMOOTHING_WINDOW).mean()
    
    df = df.dropna()
    X = np.column_stack([df["Return_Smooth"].values, df["Vol_Smooth"].values])
    return X, df


def select_best_n_components(X: np.ndarray, symbol: str) -> tuple:
    """
    Thử n_components từ MIN đến MAX, chọn theo BIC thấp nhất.
    BIC = -2 * log_likelihood + k * ln(n)
    """
    results = []
    for n in range(config.HMM_MIN_COMPONENTS, config.HMM_MAX_COMPONENTS + 1):
        try:
            model = hmm.GaussianHMM(
                n_components=n,
                covariance_type=config.HMM_COVARIANCE_TYPE,
                n_iter=config.HMM_N_ITER,
                random_state=config.HMM_RANDOM_STATE,
                init_params="mc" # Chỉ khởi tạo mean và covariance tự động
            )
            # Khởi tạo thủ công start probabilities
            model.startprob_ = np.full(n, 1.0 / n)
            # Khởi tạo thủ công ma trận chuyển đổi ép buộc persistence
            persistence_weight = 0.50 if symbol == "VNM" else config.HMM_PERSISTENCE_PRIOR
            transmat = np.full((n, n), (1.0 - persistence_weight) / (n - 1))
            np.fill_diagonal(transmat, persistence_weight)
            model.transmat_ = transmat

            model.fit(X)
            log_likelihood = model.score(X)

            # Tính số parameters: n*(n-1) transition + n*d means + n*d*(d+1)/2 covs + (n-1) start
            d = X.shape[1]
            n_params = n * (n - 1) + n * d + n * d * (d + 1) // 2 + (n - 1)
            bic = -2 * log_likelihood + n_params * np.log(len(X))
            aic = -2 * log_likelihood + 2 * n_params

            results.append({
                "n_components": n,
                "log_likelihood": log_likelihood,
                "bic": bic,
                "aic": aic,
                "model": model,
            })
            print(f"    n={n}: LogLik={log_likelihood:.2f}, "
                  f"BIC={bic:.2f}, AIC={aic:.2f}")
        except Exception as e:
            print(f"    n={n}: FAILED - {e}")

    # Chọn BIC thấp nhất
    best = min(results, key=lambda x: x["bic"])
    return best["model"], best["n_components"], results


def map_regimes(model, means: np.ndarray) -> dict:
    """
    Map hidden states sang Bear/Side/Bull dựa trên mean return.
    State có mean return cao nhất → Bullish, thấp nhất → Bearish.

    Khi BIC chọn n > 3 (ví dụ META n=4), các states thừa được gộp
    về 3 regime chuẩn theo phân vị mean return. Đảm bảo output luôn
    tương thích 3D one-hot cho State Vector của RL Agent.
    """
    mean_returns = means[:, 0]
    sorted_indices = np.argsort(mean_returns)

    n = len(sorted_indices)
    regime_map = {}
    if n == 2:
        regime_map[sorted_indices[0]] = "Bearish"
        regime_map[sorted_indices[1]] = "Bullish"
    elif n == 3:
        regime_map[sorted_indices[0]] = "Bearish"
        regime_map[sorted_indices[1]] = "Sideways"
        regime_map[sorted_indices[2]] = "Bullish"
    else:
        # n > 3: gộp states về 3 regime chuẩn theo phân vị 1/3
        # VD: n=4 → 1 Bearish, 1 Sideways, 2 Bullish (hoặc tùy phân phối)
        for i, idx in enumerate(sorted_indices):
            if i < n // 3:
                regime_map[idx] = "Bearish"
            elif i < 2 * n // 3:
                regime_map[idx] = "Sideways"
            else:
                regime_map[idx] = "Bullish"

    return regime_map


def train():
    """Huấn luyện HMM trên tập train cho tất cả mã cổ phiếu."""
    print("\n" + "=" * 60)
    print("  PHASE 2.2: HUẤN LUYỆN HMM (Market Regime Detection)")
    print("=" * 60)

    models = {}

    for symbol in config.SYMBOLS:
        print(f"\n  --- {symbol} ---")

        # Load dữ liệu train
        path = config.DATA_DIR / "train" / f"{symbol}_prices.csv"
        if not path.exists():
            print(f"  [WARN] Không tìm thấy {path}")
            continue

        df_prices = pd.read_csv(path)
        df_prices["Date"] = pd.to_datetime(df_prices["Date"])
        df_prices = df_prices.sort_values("Date").reset_index(drop=True)

        # Chuẩn bị features
        X_train, df_train = prepare_hmm_features(df_prices, symbol)
        print(f"  Train samples: {len(X_train)}")

        # Auto-select n_components
        print(f"  Đang tìm số regime tốt nhất (BIC)...")
        best_model, best_n, all_results = select_best_n_components(X_train, symbol)
        print(f"  ✔ Chọn n_components={best_n} (BIC thấp nhất)")

        # Map regimes
        regime_map = map_regimes(best_model, best_model.means_)
        print(f"  Regime mapping: {regime_map}")

        # Lưu model + metadata
        model_data = {
            "model": best_model,
            "regime_map": regime_map,
            "n_components": best_n,
            "feature_names": ["Return", "Volatility"],
        }
        models[symbol] = model_data

        # Predict trên train set để kiểm tra
        states = best_model.predict(X_train)
        regime_labels = [regime_map[s] for s in states]
        counts = pd.Series(regime_labels).value_counts()
        print(f"  Phân phối regime (train):")
        for regime, count in counts.items():
            print(f"    {regime}: {count} ({count/len(states)*100:.1f}%)")

    # Lưu toàn bộ models
    save_path = config.MODELS_DIR / "hmm_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n  ✔ HMM models đã lưu tại: {save_path}")

    return models


def evaluate():
    """Đánh giá HMM trên tập val và test."""
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ HMM MODEL")
    print("=" * 60)

    # Load models
    with open(config.MODELS_DIR / "hmm_model.pkl", "rb") as f:
        models = pickle.load(f)

    for split in ["val", "test"]:
        print(f"\n  === Tập {split.upper()} ===")

        for symbol in config.SYMBOLS:
            if symbol not in models:
                continue

            path = config.DATA_DIR / split / f"{symbol}_prices.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            X, df_clean = prepare_hmm_features(df, symbol)

            model_data = models[symbol]
            model = model_data["model"]
            regime_map = model_data["regime_map"]

            # Log-likelihood trên tập đánh giá
            log_lik = model.score(X)
            states = model.predict(X)
            regime_labels = [regime_map[s] for s in states]
            counts = pd.Series(regime_labels).value_counts()

            print(f"\n  {symbol} ({split}):")
            print(f"    Log-Likelihood: {log_lik:.2f}")
            print(f"    Samples: {len(X)}")
            for regime, count in counts.items():
                print(f"    {regime}: {count} ({count/len(states)*100:.1f}%)")

            # === ADVANCED: Log-Likelihood + Tail-Risk + Stability ===
            ll_metrics = am.compute_hmm_log_likelihood_per_sample(model, X)
            tail_metrics = am.compute_tail_risk_analysis(model, X, regime_map)
            stability = am.compute_regime_transition_stability(model)

            print(am.format_hmm_tail_risk_table(tail_metrics, ll_metrics, stability, symbol))

            # Lưu metrics
            metrics = {
                "symbol": symbol,
                "split": split,
                "log_likelihood": float(log_lik),
                "n_components": model_data["n_components"],
                "regime_distribution": counts.to_dict(),
                "log_likelihood_advanced": ll_metrics,
                "tail_risk": tail_metrics,
                "regime_stability": stability,
            }
            metrics_path = config.METRICS_DIR / f"hmm_{split}_{symbol}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)


def run():
    """Chạy train + evaluate."""
    train()
    evaluate()


if __name__ == "__main__":
    run()
