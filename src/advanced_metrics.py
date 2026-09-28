"""
Advanced Metrics Module
========================
Các hàm tính metrics chuyên sâu cho đánh giá pipeline:
  1. Rank IC (Information Coefficient) — cho EMAT và Sentiment
  2. Log-Likelihood + Tail-Risk — cho HMM
  3. Sharpe, MaxDD, Calmar, Expectancy — cho RL Agent
"""
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================
# 1. RANK IC (Information Coefficient)
# =============================================================
def compute_rank_ic(predicted_scores: np.ndarray, actual_returns: np.ndarray) -> dict:
    """
    Tính Rank IC (Spearman Rank Correlation) giữa điểm dự đoán
    và lợi nhuận thực tế. Đây là thước đo tiêu chuẩn trong quant finance
    để đánh giá khả năng xếp hạng (ranking ability) của mô hình.

    Args:
        predicted_scores: Điểm dự đoán (ví dụ: probability of Up class)
        actual_returns: Forward returns thực tế

    Returns:
        dict chứa rank_ic, p_value, ic_mean, ic_std, icir
    """
    # Loại bỏ NaN
    mask = ~(np.isnan(predicted_scores) | np.isnan(actual_returns))
    pred = predicted_scores[mask]
    actual = actual_returns[mask]

    if len(pred) < 10:
        return {"rank_ic": 0.0, "p_value": 1.0, "ic_mean": 0.0,
                "ic_std": 0.0, "icir": 0.0, "n_samples": 0}

    # Spearman Rank Correlation
    rank_ic, p_value = stats.spearmanr(pred, actual)

    return {
        "rank_ic": float(rank_ic),
        "p_value": float(p_value),
        "n_samples": int(len(pred)),
    }


def compute_rolling_rank_ic(
    predicted_scores: np.ndarray,
    actual_returns: np.ndarray,
    window: int = 60,
) -> dict:
    """
    Tính Rolling Rank IC theo cửa sổ lăn để đánh giá tính ổn định
    của khả năng xếp hạng theo thời gian.

    Returns:
        dict chứa ic_mean, ic_std, icir (Information Coefficient IR),
        hit_rate (tỷ lệ IC > 0)
    """
    mask = ~(np.isnan(predicted_scores) | np.isnan(actual_returns))
    pred = predicted_scores[mask]
    actual = actual_returns[mask]

    if len(pred) < window:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
                "hit_rate": 0.0, "n_windows": 0}

    ic_values = []
    for i in range(len(pred) - window + 1):
        p_win = pred[i:i + window]
        a_win = actual[i:i + window]
        ic, _ = stats.spearmanr(p_win, a_win)
        if not np.isnan(ic):
            ic_values.append(ic)

    if not ic_values:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
                "hit_rate": 0.0, "n_windows": 0}

    ic_arr = np.array(ic_values)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    # ICIR = IC_mean / IC_std — thước đo ổn định của IC
    icir = ic_mean / (ic_std + 1e-8)
    hit_rate = float(np.mean(ic_arr > 0))

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "hit_rate": hit_rate,
        "n_windows": len(ic_values),
    }


# =============================================================
# 2. HMM: LOG-LIKELIHOOD + TAIL-RISK
# =============================================================
def compute_hmm_log_likelihood_per_sample(model, X: np.ndarray) -> dict:
    """
    Tính log-likelihood trung bình trên mỗi sample (normalized)
    và thông tin AIC/BIC cho HMM.
    """
    n_samples = len(X)
    total_ll = model.score(X)
    ll_per_sample = total_ll / n_samples

    # Tính số parameters
    n = model.n_components
    d = X.shape[1]
    n_params = n * (n - 1) + n * d + n * d * (d + 1) // 2 + (n - 1)
    bic = -2 * total_ll + n_params * np.log(n_samples)
    aic = -2 * total_ll + 2 * n_params

    return {
        "total_log_likelihood": float(total_ll),
        "log_likelihood_per_sample": float(ll_per_sample),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": int(n_params),
        "n_samples": int(n_samples),
    }


def compute_tail_risk_analysis(
    model,
    X: np.ndarray,
    regime_map: dict,
    tail_percentile: float = 5.0,
) -> dict:
    """
    Phân tích Tail-Risk: kiểm tra xem HMM có phát hiện đúng các ngày
    biến động cực đoan (tail events) hay không.

    Cụ thể:
      - Lọc ra các ngày return nằm ngoài percentile 5% và 95%
      - Kiểm tra xem HMM gán chúng vào regime nào
      - Tail-Risk Detection Rate = tỷ lệ tail events nằm đúng Bearish/Bullish

    Args:
        model: GaussianHMM đã fit
        X: Mảng features (Return, Volatility) shape (N, 2)
        regime_map: dict mapping state_id → regime_name
        tail_percentile: Ngưỡng phần trăm cho tail (mặc định 5%)

    Returns:
        dict chứa thống kê tail-risk detection
    """
    returns = X[:, 0]  # Cột đầu tiên = daily return
    states = model.predict(X)
    regime_labels = np.array([regime_map[s] for s in states])

    # Xác định tail events
    lower_threshold = np.percentile(returns, tail_percentile)
    upper_threshold = np.percentile(returns, 100 - tail_percentile)

    # Left tail (crash days)
    left_tail_mask = returns <= lower_threshold
    left_tail_regimes = regime_labels[left_tail_mask]
    left_tail_n = int(left_tail_mask.sum())
    # HMM nên gán crash days vào "Bearish"
    left_detected = int((left_tail_regimes == "Bearish").sum()) if left_tail_n > 0 else 0
    left_detection_rate = left_detected / max(left_tail_n, 1)

    # Right tail (surge days)
    right_tail_mask = returns >= upper_threshold
    right_tail_regimes = regime_labels[right_tail_mask]
    right_tail_n = int(right_tail_mask.sum())
    # HMM nên gán surge days vào "Bullish"
    right_detected = int((right_tail_regimes == "Bullish").sum()) if right_tail_n > 0 else 0
    right_detection_rate = right_detected / max(right_tail_n, 1)

    # Phân phối regime trong tail events
    left_tail_dist = {}
    right_tail_dist = {}
    if left_tail_n > 0:
        for regime in set(regime_map.values()):
            left_tail_dist[regime] = int((left_tail_regimes == regime).sum())
    if right_tail_n > 0:
        for regime in set(regime_map.values()):
            right_tail_dist[regime] = int((right_tail_regimes == regime).sum())

    # Tổng hợp tail detection rate
    overall_detection = (left_detected + right_detected) / max(left_tail_n + right_tail_n, 1)

    return {
        "tail_percentile": tail_percentile,
        "lower_threshold_return": float(lower_threshold),
        "upper_threshold_return": float(upper_threshold),
        "left_tail": {
            "n_events": left_tail_n,
            "bearish_detected": left_detected,
            "detection_rate": float(left_detection_rate),
            "regime_distribution": left_tail_dist,
        },
        "right_tail": {
            "n_events": right_tail_n,
            "bullish_detected": right_detected,
            "detection_rate": float(right_detection_rate),
            "regime_distribution": right_tail_dist,
        },
        "overall_tail_detection_rate": float(overall_detection),
    }


def compute_regime_transition_stability(model) -> dict:
    """
    Phân tích ma trận chuyển trạng thái (Transition Matrix) của HMM
    để đánh giá tính ổn định của regime.
    Diagonal values cao → Regime ổn định, ít chuyển đổi ngẫu nhiên.
    """
    transmat = model.transmat_
    n = model.n_components

    # Persistence (diagonal) — xác suất ở lại cùng regime
    persistence = np.diag(transmat)

    return {
        "transition_matrix": transmat.tolist(),
        "regime_persistence": persistence.tolist(),
        "avg_persistence": float(np.mean(persistence)),
        "min_persistence": float(np.min(persistence)),
    }


# =============================================================
# 3. RL AGENT: SHARPE, MAXDD, CALMAR, EXPECTANCY
# =============================================================
def compute_agent_advanced_metrics(trade_history: list, initial_balance: float) -> dict:
    """
    Tính bảng tổng hợp metrics chuyên sâu cho RL Agent từ trade history.

    Args:
        trade_history: List of dicts từ env.trade_history
        initial_balance: Số vốn ban đầu

    Returns:
        dict chứa Sharpe, Sortino, MaxDD, Calmar, Expectancy,
        Profit Factor, Avg Win/Loss, và thống kê giao dịch.
    """
    if not trade_history or len(trade_history) < 2:
        return _empty_agent_metrics()

    values = np.array([t["portfolio_value"] for t in trade_history])
    daily_returns = np.diff(values) / values[:-1]

    # --- Sharpe Ratio (annualized) ---
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252))

    # --- Sortino Ratio (chỉ phạt downside volatility) ---
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-8
    sortino = float(np.mean(daily_returns) / (downside_std + 1e-8) * np.sqrt(252))

    # --- Max Drawdown ---
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    max_drawdown = float(np.max(drawdowns))

    # Tìm vị trí và thời lượng drawdown dài nhất
    max_dd_end = int(np.argmax(drawdowns))
    max_dd_start = int(np.argmax(values[:max_dd_end + 1])) if max_dd_end > 0 else 0
    max_dd_duration = max_dd_end - max_dd_start

    # --- Calmar Ratio (annualized return / max drawdown) ---
    total_return = (values[-1] - initial_balance) / initial_balance
    n_days = len(values)
    annualized_return = total_return * (252 / max(n_days, 1))
    calmar = float(annualized_return / (max_drawdown + 1e-8))

    # --- Expectancy (kỳ vọng lợi nhuận mỗi giao dịch) ---
    # Tách winning và losing trades
    winning_returns = daily_returns[daily_returns > 0]
    losing_returns = daily_returns[daily_returns < 0]

    n_wins = len(winning_returns)
    n_losses = len(losing_returns)
    n_active = n_wins + n_losses  # Chỉ đếm ngày có giao dịch thực sự
    n_total = len(daily_returns)

    # Win rate tính trên active days (có return != 0)
    win_rate = n_wins / max(n_active, 1)
    loss_rate = n_losses / max(n_active, 1)
    active_ratio = n_active / max(n_total, 1)  # Tỷ lệ ngày active
    avg_win = float(np.mean(winning_returns)) if n_wins > 0 else 0.0
    avg_loss = float(np.mean(losing_returns)) if n_losses > 0 else 0.0

    # Expectancy tính trên active days
    expectancy = win_rate * avg_win + loss_rate * avg_loss

    # --- Profit Factor = Gross Profit / Gross Loss ---
    gross_profit = float(np.sum(winning_returns)) if n_wins > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losing_returns))) if n_losses > 0 else 1e-8
    profit_factor = gross_profit / (gross_loss + 1e-8)

    # --- Risk-Reward Ratio ---
    risk_reward = abs(avg_win) / (abs(avg_loss) + 1e-8)

    # --- Underwater Analysis (thời gian dưới đỉnh) ---
    underwater = drawdowns > 0
    underwater_pct = float(np.mean(underwater)) * 100

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_dd_duration_days": max_dd_duration,
        "calmar_ratio": calmar,
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward_ratio": float(risk_reward),
        "n_winning_days": int(n_wins),
        "n_losing_days": int(n_losses),
        "n_active_days": int(n_active),
        "n_trading_days": int(n_total),
        "active_ratio": float(active_ratio),
        "underwater_pct": underwater_pct,
        "final_value": float(values[-1]),
    }


def _empty_agent_metrics() -> dict:
    """Trả về metrics rỗng khi không có dữ liệu."""
    return {
        "total_return": 0.0, "annualized_return": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown": 0.0, "max_dd_duration_days": 0,
        "calmar_ratio": 0.0, "expectancy": 0.0,
        "profit_factor": 0.0, "win_rate": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0,
        "risk_reward_ratio": 0.0, "n_winning_days": 0,
        "n_losing_days": 0, "n_trading_days": 0,
        "underwater_pct": 0.0, "final_value": 0.0,
    }


def format_agent_metrics_table(metrics: dict, label: str = "Agent") -> str:
    """
    Format metrics thành bảng đẹp cho console output.
    """
    lines = [
        f"\n  {'─' * 50}",
        f"  BẢNG TỔNG HỢP HIỆU SUẤT: {label}",
        f"  {'─' * 50}",
        f"  {'Metric':<30} {'Giá trị':>15}",
        f"  {'─' * 50}",
        f"  {'Total Return':<30} {metrics['total_return']*100:>14.2f}%",
        f"  {'Annualized Return':<30} {metrics['annualized_return']*100:>14.2f}%",
        f"  {'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>15.4f}",
        f"  {'Sortino Ratio':<30} {metrics['sortino_ratio']:>15.4f}",
        f"  {'Max Drawdown':<30} {metrics['max_drawdown']*100:>14.2f}%",
        f"  {'Max DD Duration (days)':<30} {metrics['max_dd_duration_days']:>15d}",
        f"  {'Calmar Ratio':<30} {metrics['calmar_ratio']:>15.4f}",
        f"  {'─' * 50}",
        f"  {'Expectancy (per step)':<30} {metrics['expectancy']*100:>14.6f}%",
        f"  {'Profit Factor':<30} {metrics['profit_factor']:>15.4f}",
        f"  {'Win Rate (Active)':<30} {metrics['win_rate']*100:>14.2f}%",
        f"  {'Avg Win':<30} {metrics['avg_win']*100:>14.4f}%",
        f"  {'Avg Loss':<30} {metrics['avg_loss']*100:>14.4f}%",
        f"  {'Risk/Reward Ratio':<30} {metrics['risk_reward_ratio']:>15.4f}",
        f"  {'─' * 50}",
        f"  {'Winning Days':<30} {metrics['n_winning_days']:>15d}",
        f"  {'Losing Days':<30} {metrics['n_losing_days']:>15d}",
        f"  {'Active Ratio':<30} {metrics.get('active_ratio', 0)*100:>14.2f}%",
        f"  {'Underwater %':<30} {metrics['underwater_pct']:>14.2f}%",
        f"  {'Final Value':<30} ${metrics['final_value']:>14,.2f}",
        f"  {'─' * 50}",
    ]
    return "\n".join(lines)


def format_rank_ic_table(rank_ic_results: dict, model_name: str = "Model") -> str:
    """
    Format bảng Rank IC cho EMAT hoặc Sentiment.
    rank_ic_results: dict mapping horizon/label → ic metrics
    """
    lines = [
        f"\n  {'─' * 65}",
        f"  BẢNG RANK IC: {model_name}",
        f"  {'─' * 65}",
        f"  {'Horizon':<12} {'Rank IC':>10} {'p-value':>10} {'IC Mean':>10} "
        f"{'ICIR':>8} {'Hit%':>8} {'N':>6}",
        f"  {'─' * 65}",
    ]
    for key, m in rank_ic_results.items():
        lines.append(
            f"  {key:<12} {m.get('rank_ic', m.get('ic_mean', 0)):>10.4f} "
            f"{m.get('p_value', 0):>10.4f} "
            f"{m.get('ic_mean', 0):>10.4f} "
            f"{m.get('icir', 0):>8.4f} "
            f"{m.get('hit_rate', 0)*100:>7.1f}% "
            f"{m.get('n_samples', m.get('n_windows', 0)):>6d}"
        )
    lines.append(f"  {'─' * 65}")
    return "\n".join(lines)


def format_hmm_tail_risk_table(tail_results: dict, ll_results: dict,
                                stability: dict, symbol: str) -> str:
    """Format bảng HMM Log-Likelihood + Tail-Risk."""
    lines = [
        f"\n  {'─' * 55}",
        f"  HMM DIAGNOSTICS: {symbol}",
        f"  {'─' * 55}",
        f"  Log-Likelihood (total)   : {ll_results['total_log_likelihood']:>12.2f}",
        f"  Log-Likelihood (per obs) : {ll_results['log_likelihood_per_sample']:>12.4f}",
        f"  AIC                      : {ll_results['aic']:>12.2f}",
        f"  BIC                      : {ll_results['bic']:>12.2f}",
        f"  Num Parameters           : {ll_results['n_params']:>12d}",
        f"  {'─' * 55}",
        f"  REGIME PERSISTENCE (diagonal of Transition Matrix):",
    ]
    for i, p in enumerate(stability['regime_persistence']):
        lines.append(f"    State {i}: {p:.4f}")
    lines.append(f"  Avg Persistence          : {stability['avg_persistence']:.4f}")
    lines.append(f"  {'─' * 55}")
    lines.append(f"  TAIL-RISK ANALYSIS (ngưỡng {tail_results['tail_percentile']}%)")
    lines.append(f"  Left tail  (crash):")
    lt = tail_results['left_tail']
    lines.append(f"    Events: {lt['n_events']} | Bearish detected: "
                 f"{lt['bearish_detected']} | Rate: {lt['detection_rate']*100:.1f}%")
    if lt['regime_distribution']:
        dist_str = ", ".join(f"{k}: {v}" for k, v in lt['regime_distribution'].items())
        lines.append(f"    Regime dist: {dist_str}")

    lines.append(f"  Right tail (surge):")
    rt = tail_results['right_tail']
    lines.append(f"    Events: {rt['n_events']} | Bullish detected: "
                 f"{rt['bullish_detected']} | Rate: {rt['detection_rate']*100:.1f}%")
    if rt['regime_distribution']:
        dist_str = ", ".join(f"{k}: {v}" for k, v in rt['regime_distribution'].items())
        lines.append(f"    Regime dist: {dist_str}")

    lines.append(f"  Overall Tail Detection   : "
                 f"{tail_results['overall_tail_detection_rate']*100:.1f}%")
    lines.append(f"  {'─' * 55}")
    return "\n".join(lines)
