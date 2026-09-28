"""
Baseline Comparison: So sánh Hybrid Agent với các chiến lược baseline
======================================================================
Các baseline:
  1. Buy & Hold                — Mua đầu kỳ, giữ đến cuối
  2. Random Agent              — Chọn hành động ngẫu nhiên
  3. SMA Crossover (10/30)     — Mua khi SMA10 cắt lên SMA30, bán ngược lại
  4. RSI Strategy              — Mua khi RSI < 30 (oversold), bán khi RSI > 70
  5. MACD Strategy             — Mua khi MACD > Signal, bán ngược lại
  6. DQN-only Agent            — Chỉ dùng DQN head (không qua PPO fine-tune)
  7. Hybrid DQN-PPO Agent      — Mô hình đề xuất (PPO fine-tune sau DQN)

Tất cả baseline đều chạy trên cùng HybridTradingEnv (7 actions) để đảm bảo
công bằng về transaction cost, cooldown, stop-loss.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config
from phase3_trading_env import HybridTradingEnv
from phase4_hybrid_agent import HybridAgent


# =====================================================================
# TIỆN ÍCH: Chạy 1 chiến lược trên env, trả về metrics PER SYMBOL
# =====================================================================
def run_strategy_per_symbol(strategy_fn, split="test"):
    """
    Chạy chiến lược trên từng symbol riêng biệt.
    strategy_fn(obs, step, env) -> action (int 0-6)
    Trả về dict {symbol: metrics_dict}
    """
    results = {}
    for symbol in config.SYMBOLS:
        env = HybridTradingEnv(split=split, symbol_filter=symbol)
        state, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = strategy_fn(state, step, env)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        metrics = env.get_performance_metrics()
        results[symbol] = metrics
    return results


# =====================================================================
# 1. BUY & HOLD: Mua 100% ngay bước đầu, giữ đến cuối
# =====================================================================
def strategy_buy_hold(state, step, env):
    if step == 0:
        return 3  # Buy 100%
    return 0  # Hold


# =====================================================================
# 2. RANDOM AGENT: Ngẫu nhiên đều
# =====================================================================
def strategy_random(state, step, env):
    return np.random.randint(0, config.NUM_ACTIONS)


# =====================================================================
# 3. SMA CROSSOVER (10/30): Dựa trên OHLCV trong state
# =====================================================================
class SMAStrategy:
    """SMA Crossover dùng giá Close từ env.prices."""
    def __init__(self, short=10, long=30):
        self.short_w = short
        self.long_w = long
        self.price_history = []
        self.position = False  # Đang giữ cổ phiếu hay không

    def reset(self):
        self.price_history = []
        self.position = False

    def __call__(self, state, step, env):
        price = env.prices[env.current_step - 1] if env.current_step > 0 else env.prices[0]
        self.price_history.append(price)

        if len(self.price_history) < self.long_w:
            return 0  # Chưa đủ data → Hold

        sma_short = np.mean(self.price_history[-self.short_w:])
        sma_long = np.mean(self.price_history[-self.long_w:])

        if sma_short > sma_long and not self.position:
            self.position = True
            return 3  # Buy 100% (Golden Cross)
        elif sma_short < sma_long and self.position:
            self.position = False
            return 6  # Sell 100% (Death Cross)
        return 0  # Hold


# =====================================================================
# 4. RSI STRATEGY: Dựa trên RSI trong state vector [index 5]
# =====================================================================
class RSIStrategy:
    """Mua khi RSI < 30, bán khi RSI > 70."""
    def __init__(self, buy_threshold=30, sell_threshold=70):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position = False

    def reset(self):
        self.position = False

    def __call__(self, state, step, env):
        # RSI nằm ở index 5 trong state vector (đã normalized)
        # Cần ước lượng RSI thực từ giá
        prices = env.prices[:env.current_step + 1]
        if len(prices) < 15:
            return 0

        # Tính RSI 14 ngày từ giá thực
        deltas = np.diff(prices[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) + 1e-8
        avg_loss = np.mean(losses) + 1e-8
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if rsi < self.buy_threshold and not self.position:
            self.position = True
            return 3  # Buy 100%
        elif rsi > self.sell_threshold and self.position:
            self.position = False
            return 6  # Sell 100%
        return 0


# =====================================================================
# 5. MACD STRATEGY: Dựa trên MACD vs Signal Line
# =====================================================================
class MACDStrategy:
    """Mua khi MACD > Signal, bán khi MACD < Signal."""
    def __init__(self):
        self.position = False
        self.prev_cross = None  # 'above' or 'below'

    def reset(self):
        self.position = False
        self.prev_cross = None

    def __call__(self, state, step, env):
        prices = env.prices[:env.current_step + 1]
        if len(prices) < 35:
            return 0

        # Tính MACD (12, 26, 9) từ giá thực
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = ema12 - ema26

        # Signal line = EMA(9) của MACD
        if len(prices) < 35:
            return 0
        macd_series = []
        for i in range(26, len(prices)):
            e12 = self._ema(prices[:i+1], 12)
            e26 = self._ema(prices[:i+1], 26)
            macd_series.append(e12 - e26)
        if len(macd_series) < 9:
            return 0

        signal = self._ema(np.array(macd_series), 9)
        current_macd = macd_series[-1]

        if current_macd > signal and not self.position:
            self.position = True
            return 3  # Buy 100%
        elif current_macd < signal and self.position:
            self.position = False
            return 6  # Sell 100%
        return 0

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return data[-1]
        alpha = 2.0 / (period + 1)
        ema = data[-period]
        for price in data[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema


# =====================================================================
# 6. DQN-ONLY AGENT: Load checkpoint sau giai đoạn DQN (chưa PPO)
# =====================================================================
def create_dqn_only_strategy():
    """Load agent chỉ qua DQN training."""
    device = config.DEVICE
    dqn_path = config.MODELS_DIR / "agent_after_dqn.pth"
    if not dqn_path.exists():
        print(f"  [SKIP] Không tìm thấy {dqn_path}")
        return None

    state_dim = config.STATE_DIM_TOTAL
    agent = HybridAgent(state_dim).to(device)
    agent.load_state_dict(torch.load(dqn_path, map_location=device))
    agent.eval()

    def strategy(state, step, env):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_vals = agent.get_q_values(s)
            return q_vals.argmax(dim=1).item()

    return strategy


# =====================================================================
# 7. HYBRID AGENT: Load final model (DQN + PPO)
# =====================================================================
def create_hybrid_strategy():
    """Load agent cuối cùng sau cả DQN + PPO."""
    device = config.DEVICE
    model_path = config.MODELS_DIR / "hybrid_agent_final.pth"
    if not model_path.exists():
        print(f"  [SKIP] Không tìm thấy {model_path}")
        return None

    state_dim = config.STATE_DIM_TOTAL
    agent = HybridAgent(state_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    def strategy(state, step, env):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, _, _ = agent.select_action_ppo(s)
            return action

    return strategy


# =====================================================================
# MAIN: Chạy tất cả baselines và tổng hợp bảng
# =====================================================================
def main():
    print("=" * 80)
    print("  BASELINE COMPARISON: Hybrid Agent vs Baselines (Test Set)")
    print("=" * 80)

    np.random.seed(config.RANDOM_SEED)

    # Định nghĩa tất cả chiến lược
    sma = SMAStrategy(short=10, long=30)
    rsi = RSIStrategy()
    macd = MACDStrategy()

    strategies = {
        "Buy & Hold": strategy_buy_hold,
        "Random": strategy_random,
        "SMA Cross (10/30)": sma,
        "RSI (30/70)": rsi,
        "MACD (12/26/9)": macd,
    }

    # Thêm DQN-only nếu có checkpoint
    dqn_strategy = create_dqn_only_strategy()
    if dqn_strategy:
        strategies["DQN-only"] = dqn_strategy

    # Thêm Hybrid Agent
    hybrid_strategy = create_hybrid_strategy()
    if hybrid_strategy:
        strategies["Hybrid DQN-PPO"] = hybrid_strategy

    # Chạy benchmark
    all_results = {}
    for name, strategy_fn in strategies.items():
        print(f"\n  [{name}] Đang chạy backtest...")

        # Reset state cho chiến lược có trạng thái
        if hasattr(strategy_fn, 'reset'):
            strategy_fn.reset()

        # Chạy per-symbol
        sym_results = {}
        for symbol in config.SYMBOLS:
            # Reset strategy mỗi symbol
            if hasattr(strategy_fn, 'reset'):
                strategy_fn.reset()

            env = HybridTradingEnv(split="test", symbol_filter=symbol)
            state, _ = env.reset()
            done = False
            step = 0

            while not done:
                action = strategy_fn(state, step, env)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            metrics = env.get_performance_metrics()
            sym_results[symbol] = metrics
            ret = metrics.get('total_return', 0) * 100
            dd = metrics.get('max_drawdown', 0) * 100
            sr = metrics.get('sharpe_ratio', 0)
            trades = metrics.get('total_trades', 0)
            print(f"    {symbol}: Return={ret:+.2f}% | MaxDD={dd:.2f}% | "
                  f"Sharpe={sr:.3f} | Trades={trades}")

        all_results[name] = sym_results

    # =====================================================================
    # TẠO BẢNG TỔNG HỢP
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  BẢNG SO SÁNH TỔNG HỢP — TEST SET")
    print("=" * 100)

    # Tính trung bình qua 3 symbols
    summary_rows = []
    for name in strategies.keys():
        sym_results = all_results[name]
        avg_return = np.mean([sym_results[s].get('total_return', 0) for s in config.SYMBOLS])
        avg_dd = np.mean([sym_results[s].get('max_drawdown', 0) for s in config.SYMBOLS])
        avg_sharpe = np.mean([sym_results[s].get('sharpe_ratio', 0) for s in config.SYMBOLS])
        avg_calmar = np.mean([sym_results[s].get('calmar_ratio', 0) for s in config.SYMBOLS])
        avg_wr = np.mean([sym_results[s].get('win_rate', 0) for s in config.SYMBOLS])
        total_trades = sum([sym_results[s].get('total_trades', 0) for s in config.SYMBOLS])

        # Per-symbol returns
        per_sym = {s: sym_results[s].get('total_return', 0) * 100 for s in config.SYMBOLS}

        summary_rows.append({
            'Strategy': name,
            'Avg Return (%)': avg_return * 100,
            'GOOGL (%)': per_sym.get('GOOGL', 0),
            'META (%)': per_sym.get('META', 0),
            'VNM (%)': per_sym.get('VNM', 0),
            'Avg MaxDD (%)': avg_dd * 100,
            'Avg Sharpe': avg_sharpe,
            'Avg Calmar': avg_calmar,
            'Avg Win Rate': avg_wr,
            'Total Trades': total_trades,
        })

    df = pd.DataFrame(summary_rows)
    df = df.sort_values('Avg Return (%)', ascending=False)

    # In bảng đẹp
    print(f"\n{'Strategy':<22} | {'Avg Ret%':>9} | {'GOOGL%':>8} | {'META%':>8} | "
          f"{'VNM%':>8} | {'MaxDD%':>8} | {'Sharpe':>7} | {'Calmar':>8} | "
          f"{'WinRate':>8} | {'Trades':>7}")
    print("-" * 130)
    for _, row in df.iterrows():
        print(f"{row['Strategy']:<22} | {row['Avg Return (%)']:>+9.2f} | "
              f"{row['GOOGL (%)']:>+8.2f} | {row['META (%)']:>+8.2f} | "
              f"{row['VNM (%)']:>+8.2f} | {row['Avg MaxDD (%)']:>8.2f} | "
              f"{row['Avg Sharpe']:>7.3f} | {row['Avg Calmar']:>8.3f} | "
              f"{row['Avg Win Rate']:>8.3f} | {row['Total Trades']:>7.0f}")

    # =====================================================================
    # LƯU KẾT QUẢ
    # =====================================================================
    # Lưu JSON
    json_out = {}
    for name in strategies.keys():
        json_out[name] = {}
        for s in config.SYMBOLS:
            json_out[name][s] = all_results[name][s]

    out_path = config.METRICS_DIR / "baseline_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n  Kết quả đã lưu: {out_path}")

    # Lưu CSV
    csv_path = config.METRICS_DIR / "baseline_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV đã lưu: {csv_path}")

    # =====================================================================
    # TẠO BIỂU ĐỒ SO SÁNH
    # =====================================================================
    create_comparison_chart(df)

    return df


def create_comparison_chart(df):
    """Tạo biểu đồ so sánh các baseline."""
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 13,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    strategies = df['Strategy'].tolist()
    x = np.arange(len(strategies))

    # Màu sắc: highlight Hybrid Agent
    colors = []
    for s in strategies:
        if 'Hybrid' in s:
            colors.append('#6A1B9A')  # Tím nổi bật
        elif 'DQN' in s:
            colors.append('#1565C0')  # Xanh dương
        elif 'Buy' in s:
            colors.append('#757575')  # Xám
        elif 'Random' in s:
            colors.append('#BDBDBD')  # Xám nhạt
        else:
            colors.append('#E65100')  # Cam cho Technical Analysis

    # Chart 1: Average Return
    ax = axes[0]
    bars = ax.barh(x, df['Avg Return (%)'].values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Avg Return (%)')
    ax.set_title('Average Return', fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    for bar, val in zip(bars, df['Avg Return (%)'].values):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right',
                fontsize=8, fontweight='bold')

    # Chart 2: Max Drawdown (nhỏ hơn = tốt hơn)
    ax = axes[1]
    bars = ax.barh(x, df['Avg MaxDD (%)'].values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Avg Max Drawdown (%)')
    ax.set_title('Max Drawdown (lower = better)', fontweight='bold')
    for bar, val in zip(bars, df['Avg MaxDD (%)'].values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')

    # Chart 3: Sharpe Ratio
    ax = axes[2]
    bars = ax.barh(x, df['Avg Sharpe'].values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Avg Sharpe Ratio')
    ax.set_title('Sharpe Ratio (higher = better)', fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    for bar, val in zip(bars, df['Avg Sharpe'].values):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=8, fontweight='bold')

    fig.suptitle('Baseline Comparison — Test Set Performance', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    chart_path = config.PLOTS_DIR / 'fig14_baseline_comparison.png'
    plt.savefig(chart_path)
    plt.close()
    print(f"  Biểu đồ đã lưu: {chart_path}")

    # =========================================================
    # BIỂU ĐỒ PER-SYMBOL RETURN
    # =========================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.12
    symbols = config.SYMBOLS
    n_strat = len(strategies)

    for i, sym in enumerate(symbols):
        col = f'{sym} (%)'
        vals = df[col].values
        offset = (i - len(symbols)/2 + 0.5) * width * 2.5
        bars = ax.bar(x + offset, vals, width * 2, label=sym, alpha=0.85)
        for bar, val in zip(bars, vals):
            y_pos = val if val >= 0 else val
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:+.1f}', ha='center',
                    va='bottom' if val >= 0 else 'top',
                    fontsize=6, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Return (%)')
    ax.set_title('Per-Symbol Return Comparison — Test Set', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(0, color='black', lw=0.8)
    plt.tight_layout()
    chart_path2 = config.PLOTS_DIR / 'fig15_per_symbol_returns.png'
    plt.savefig(chart_path2)
    plt.close()
    print(f"  Biểu đồ đã lưu: {chart_path2}")


if __name__ == "__main__":
    main()
