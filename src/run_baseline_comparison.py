"""
Baseline Comparison cho tất cả 10 mã (7 mới + 3 cũ).
=====================================================
Chạy 5 baselines: Buy&Hold, Random, SMA Cross, RSI, MACD
trên test set price data.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Đường dẫn data cho 2 nhóm
NEW_DATA_DIR = config.DATA_DIR  # 7 mã mới
OLD_DATA_DIR = Path(r"D:\HUTECH\thayKhang\EMAT\data")  # 3 mã cũ

ALL_SYMBOLS = {
    # 7 mã mới (từ project hiện tại)
    "AAPL": NEW_DATA_DIR, "MSFT": NEW_DATA_DIR, "JPM": NEW_DATA_DIR,
    "XOM": NEW_DATA_DIR, "VCB": NEW_DATA_DIR, "HPG": NEW_DATA_DIR, "FPT": NEW_DATA_DIR,
    # 3 mã cũ (từ project gốc)
    "GOOGL": OLD_DATA_DIR, "META": OLD_DATA_DIR, "VNM": OLD_DATA_DIR,
}

INITIAL_BALANCE = 100_000.0
TRANSACTION_COST = 0.001


def load_test_prices(symbol, data_dir):
    """Load test prices cho 1 mã."""
    price_path = data_dir / "test" / f"{symbol}_prices.csv"
    if not price_path.exists():
        return None
    df = pd.read_csv(price_path)
    # Tìm cột Close
    close_col = "Close"
    if close_col not in df.columns:
        for c in df.columns:
            if "close" in c.lower():
                close_col = c
                break
    return df[close_col].values


def compute_metrics(values, initial_balance):
    """Tính metrics từ portfolio values series."""
    if len(values) < 2:
        return {}
    
    total_return = (values[-1] - initial_balance) / initial_balance
    returns = np.diff(values) / values[:-1]
    
    # Sharpe (annualized)
    if np.std(returns) > 1e-8:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max Drawdown
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Calmar
    ann_return = (1 + total_return) ** (252 / max(len(values), 1)) - 1
    calmar = ann_return / (max_dd + 1e-8)
    
    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "final_value": float(values[-1]),
    }


# =====================================================================
# BASELINE STRATEGIES
# =====================================================================

def buy_and_hold(prices):
    """Mua toàn bộ ngày 1, giữ đến cuối."""
    shares = (INITIAL_BALANCE * (1 - TRANSACTION_COST)) / prices[0]
    values = [shares * p for p in prices]
    metrics = compute_metrics(values, INITIAL_BALANCE)
    metrics["total_trades"] = 1
    metrics["win_rate"] = 0.0
    return metrics


def random_strategy(prices, seed=42):
    """Random Buy/Sell."""
    np.random.seed(seed)
    balance = INITIAL_BALANCE
    shares = 0.0
    values = []
    trades = 0
    wins = 0
    
    for i, p in enumerate(prices):
        action = np.random.choice(["hold", "buy", "sell"], p=[0.7, 0.15, 0.15])
        
        if action == "buy" and balance > 100:
            invest = balance * 0.5
            cost = invest * TRANSACTION_COST
            shares += (invest - cost) / p
            balance -= invest
            trades += 1
        elif action == "sell" and shares > 0:
            revenue = shares * p
            cost = revenue * TRANSACTION_COST
            prev_value = shares * prices[max(0, i-1)]
            if revenue > prev_value:
                wins += 1
            balance += revenue - cost
            shares = 0.0
            trades += 1
        
        values.append(balance + shares * p)
    
    metrics = compute_metrics(values, INITIAL_BALANCE)
    metrics["total_trades"] = trades
    metrics["win_rate"] = wins / max(trades, 1)
    return metrics


def sma_cross(prices, short=10, long=30):
    """SMA Crossover strategy."""
    balance = INITIAL_BALANCE
    shares = 0.0
    values = []
    trades = 0
    wins = 0
    
    sma_short = pd.Series(prices).rolling(short).mean().values
    sma_long = pd.Series(prices).rolling(long).mean().values
    
    for i in range(len(prices)):
        p = prices[i]
        
        if i >= long:
            # Golden cross: short > long → Buy
            if sma_short[i] > sma_long[i] and sma_short[i-1] <= sma_long[i-1]:
                if balance > 100 and shares == 0:
                    invest = balance
                    cost = invest * TRANSACTION_COST
                    shares = (invest - cost) / p
                    balance -= invest
                    trades += 1
            
            # Death cross: short < long → Sell
            elif sma_short[i] < sma_long[i] and sma_short[i-1] >= sma_long[i-1]:
                if shares > 0:
                    revenue = shares * p
                    cost = revenue * TRANSACTION_COST
                    prev_cost = shares * prices[max(0, i-1)]
                    if revenue > prev_cost:
                        wins += 1
                    balance += revenue - cost
                    shares = 0.0
                    trades += 1
        
        values.append(balance + shares * p)
    
    metrics = compute_metrics(values, INITIAL_BALANCE)
    metrics["total_trades"] = trades
    metrics["win_rate"] = wins / max(trades, 1)
    return metrics


def rsi_strategy(prices, period=14, oversold=30, overbought=70):
    """RSI (30/70) strategy."""
    balance = INITIAL_BALANCE
    shares = 0.0
    values = []
    trades = 0
    wins = 0
    
    # Tính RSI
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).rolling(period).mean().values
    avg_loss = pd.Series(losses).rolling(period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    
    for i in range(len(prices)):
        p = prices[i]
        
        if i >= period:
            if rsi[i] < oversold and balance > 100 and shares == 0:
                invest = balance
                cost = invest * TRANSACTION_COST
                shares = (invest - cost) / p
                balance -= invest
                trades += 1
            
            elif rsi[i] > overbought and shares > 0:
                revenue = shares * p
                cost = revenue * TRANSACTION_COST
                prev_cost = shares * prices[max(0, i-1)]
                if revenue > prev_cost:
                    wins += 1
                balance += revenue - cost
                shares = 0.0
                trades += 1
        
        values.append(balance + shares * p)
    
    metrics = compute_metrics(values, INITIAL_BALANCE)
    metrics["total_trades"] = trades
    metrics["win_rate"] = wins / max(trades, 1)
    return metrics


def macd_strategy(prices, fast=12, slow=26, signal=9):
    """MACD (12/26/9) strategy."""
    balance = INITIAL_BALANCE
    shares = 0.0
    values = []
    trades = 0
    wins = 0
    
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    
    for i in range(1, len(prices)):
        p = prices[i]
        
        if i >= slow:
            # MACD cross above signal → Buy
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                if balance > 100 and shares == 0:
                    invest = balance
                    cost = invest * TRANSACTION_COST
                    shares = (invest - cost) / p
                    balance -= invest
                    trades += 1
            
            # MACD cross below signal → Sell
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                if shares > 0:
                    revenue = shares * p
                    cost = revenue * TRANSACTION_COST
                    prev_cost = shares * prices[max(0, i-1)]
                    if revenue > prev_cost:
                        wins += 1
                    balance += revenue - cost
                    shares = 0.0
                    trades += 1
        
        values.append(balance + shares * p)
    
    metrics = compute_metrics(values, INITIAL_BALANCE)
    metrics["total_trades"] = trades
    metrics["win_rate"] = wins / max(trades, 1)
    return metrics


# =====================================================================
# MAIN
# =====================================================================
def run():
    print("=" * 70)
    print("  BASELINE COMPARISON — 10 MÃ (7 mới + 3 cũ)")
    print("=" * 70)
    
    strategies = {
        "Buy & Hold": buy_and_hold,
        "Random": random_strategy,
        "SMA Cross (10/30)": sma_cross,
        "RSI (30/70)": rsi_strategy,
        "MACD (12/26/9)": macd_strategy,
    }
    
    results = {name: {} for name in strategies}
    
    for symbol, data_dir in ALL_SYMBOLS.items():
        prices = load_test_prices(symbol, data_dir)
        if prices is None:
            print(f"\n  [{symbol}] SKIP — không tìm thấy test prices tại {data_dir}")
            continue
        
        print(f"\n  [{symbol}] {len(prices)} ngày test")
        
        for strat_name, strat_func in strategies.items():
            try:
                metrics = strat_func(prices)
                results[strat_name][symbol] = metrics
                ret = metrics["total_return"] * 100
                sharpe = metrics["sharpe_ratio"]
                mdd = metrics["max_drawdown"] * 100
                print(f"    {strat_name:<20} | Return: {ret:>+8.2f}% | Sharpe: {sharpe:>6.2f} | MDD: {mdd:>6.2f}%")
            except Exception as e:
                print(f"    {strat_name:<20} | ERROR: {e}")
    
    # Lưu kết quả
    output_path = config.METRICS_DIR / "baseline_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✔ Đã lưu: {output_path}")
    
    # CSV cho dễ đọc
    csv_path = config.METRICS_DIR / "baseline_comparison.csv"
    rows = []
    for strat_name, strat_results in results.items():
        for sym, m in strat_results.items():
            rows.append({
                "Strategy": strat_name,
                "Symbol": sym,
                "Return": m["total_return"],
                "Sharpe": m["sharpe_ratio"],
                "MaxDD": m["max_drawdown"],
                "Calmar": m["calmar_ratio"],
                "Trades": m.get("total_trades", 0),
                "WinRate": m.get("win_rate", 0),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✔ Đã lưu: {csv_path}")
    
    # Tổng hợp trung bình
    print(f"\n{'=' * 70}")
    print(f"  TONG HOP TRUNG BINH")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<20} | {'Avg Return':>10} | {'Avg Sharpe':>10} | {'Avg MDD':>10} | {'Avg Calmar':>10}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for strat_name, strat_results in results.items():
        if not strat_results:
            continue
        avg_ret = np.mean([m["total_return"] for m in strat_results.values()])
        avg_sharpe = np.mean([m["sharpe_ratio"] for m in strat_results.values()])
        avg_mdd = np.mean([m["max_drawdown"] for m in strat_results.values()])
        avg_calmar = np.mean([m["calmar_ratio"] for m in strat_results.values()])
        print(f"  {strat_name:<20} | {avg_ret*100:>+9.2f}% | {avg_sharpe:>10.2f} | {avg_mdd*100:>9.2f}% | {avg_calmar:>10.2f}")


if __name__ == "__main__":
    run()
