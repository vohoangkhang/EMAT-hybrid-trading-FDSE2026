import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from phase3_trading_env import HybridTradingEnv
from phase4_hybrid_agent import HybridAgent
from phase2_train_hmm import prepare_hmm_features

# =====================================================================
# CẤU HÌNH ĐỊNH DẠNG MATPLOTLIB CHUẨN IEEE (Scientific Paper Style)
# =====================================================================
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times Newrepr', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,        # Chuẩn xuất bản
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# =====================================================================
# 1. HMM REGIME VISUALIZATION
# =====================================================================
def plot_hmm_regime(symbol="META", split="test"):
    print(f"Generating HMM Regime Plot for {symbol} ({split})...")
    
    hmm_path = config.MODELS_DIR / "hmm_model.pkl"
    if not hmm_path.exists():
        print("HMM Model not found.")
        return

    with open(hmm_path, "rb") as f:
        models = pickle.load(f)
        
    if symbol not in models:
        print(f"Symbol {symbol} not found in HMM models.")
        return
        
    model_data = models[symbol]
    hmm_model = model_data["model"]
    regime_map = model_data["regime_map"]
    
    # Load prices
    price_path = config.DATA_DIR / split / f"{symbol}_prices.csv"
    if not price_path.exists():
        print(f"Price data for {symbol} not found.")
        return
        
    df = pd.read_csv(price_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    X, df_clean = prepare_hmm_features(df, symbol)
    states = hmm_model.predict(X)
    
    # Align lengths
    df_clean = df_clean.iloc[:len(states)].copy()
    df_clean['State'] = states
    df_clean['Regime'] = df_clean['State'].map(regime_map)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_clean['Date'], df_clean['Close'], color='black', linewidth=1.2, label='Price')
    
    colors = {'Bullish': '#2ca02c', 'Bearish': '#d62728', 'Sideways': '#ff7f0e'}
    
    # Fill between for regimes
    prev_state = None
    start_idx = 0
    
    for i in range(1, len(df_clean)):
        current_regime = df_clean['Regime'].iloc[i]
        if current_regime != df_clean['Regime'].iloc[i-1] or i == len(df_clean) - 1:
            end_idx = i
            regime_label = df_clean['Regime'].iloc[i-1]
            ax.axvspan(df_clean['Date'].iloc[start_idx], df_clean['Date'].iloc[end_idx], 
                       color=colors.get(regime_label, 'grey'), alpha=0.3, lw=0)
            start_idx = i
            
    # Add dummy lines for legend
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    patch_bull = mpatches.Patch(color=colors['Bullish'], alpha=0.3, label='Bullish Regime')
    patch_bear = mpatches.Patch(color=colors['Bearish'], alpha=0.3, label='Bearish Regime')
    patch_side = mpatches.Patch(color=colors['Sideways'], alpha=0.3, label='Sideways Regime')
    handles.extend([patch_bull, patch_bear, patch_side])
    
    ax.legend(handles=handles, loc='best')
    ax.set_title(f'Hidden Markov Model - Market Regimes ({symbol})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    out_path = config.PLOTS_DIR / f'fig1_hmm_regime_{symbol}.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# =====================================================================
# 2. EVALUATE AGENT AND GET TRADE HISTORY
# =====================================================================
def get_agent_history(split="test"):
    agent_path = config.MODELS_DIR / "hybrid_agent_final.pth"
    if not agent_path.exists():
        print("Agent model not found.")
        return None, None
        
    env = HybridTradingEnv(split=split)
    agent = HybridAgent(env.state_dim).to(config.DEVICE)
    agent.load_state_dict(torch.load(agent_path, map_location=config.DEVICE))
    agent.eval()
    
    state, _ = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            # For testing, we use the deterministic Q-values or PPO policy max prob.
            # We'll use Q-values here as in phase4 evaluate.
            q_values = agent.get_q_values(state_tensor)
            action = q_values.argmax(dim=-1).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    return env.trade_history, env.prices

# =====================================================================
# 3. CUMULATIVE RETURN COMPARISON
# =====================================================================
def plot_cumulative_returns(trade_history, prices):
    if not trade_history:
        return
        
    print("Generating Cumulative Return Plot...")
    
    steps = [t["step"] for t in trade_history]
    agent_values = [t["portfolio_value"] for t in trade_history]
    
    # Calculate Buy & Hold from prices
    initial_balance = agent_values[0]
    bh_shares = initial_balance / prices[0]
    bh_values = [bh_shares * prices[i] for i in steps]
    
    # Normalize to cumulative returns
    agent_returns = (np.array(agent_values) - initial_balance) / initial_balance * 100
    bh_returns = (np.array(bh_values) - initial_balance) / initial_balance * 100
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(steps, agent_returns, color='#1f77b4', linewidth=2, label='RL Agent (PPO/DQN)')
    ax.plot(steps, bh_returns, color='#7f7f7f', linewidth=1.5, linestyle='--', label='Buy & Hold Baseline')
    
    ax.set_title('Cumulative Return Comparison (Test Set)')
    ax.set_xlabel('Trading Steps')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(loc='upper left')
    
    # Highlight max drawdown areas or 0 line
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    
    out_path = config.PLOTS_DIR / 'fig2_cumulative_returns.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# =====================================================================
# 4. TRADING ACTION SCATTER PLOT
# =====================================================================
def plot_trading_actions(trade_history, symbol_filter=None):
    if not trade_history:
        return
        
    print("Generating Trading Action Plot...")
    
    # If we have multiple symbols concatenated in the test env, 
    # we just plot a chunk of steps to make the scatter plot readable.
    # Let's plot the first 500 steps for clarity.
    
    plot_history = trade_history[:400]
    
    steps = [t["step"] for t in plot_history]
    prices = [t["price"] for t in plot_history]
    actions = [t["action"] for t in plot_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, prices, color='black', alpha=0.6, linewidth=1.5, label='Price')
    
    buy_steps, buy_prices = [], []
    sell_steps, sell_prices = [], []
    
    for i, a in enumerate(actions):
        if a in [1, 2, 3]:  # Buy actions
            buy_steps.append(steps[i])
            buy_prices.append(prices[i])
        elif a in [4, 5, 6]:  # Sell actions
            sell_steps.append(steps[i])
            sell_prices.append(prices[i])
            
    ax.scatter(buy_steps, buy_prices, marker='^', color='green', s=100, label='Buy Action', zorder=5)
    ax.scatter(sell_steps, sell_prices, marker='v', color='red', s=100, label='Sell Action', zorder=5)
    
    ax.set_title('RL Agent Trading Actions Over Price')
    ax.set_xlabel('Trading Steps')
    ax.set_ylabel('Price')
    ax.legend()
    
    out_path = config.PLOTS_DIR / 'fig3_trading_actions.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def run():
    print("="*60)
    print(" GENERATING PAPER FIGURES (IEEE FORMAT)")
    print("="*60)
    
    plot_hmm_regime(symbol="META")
    plot_hmm_regime(symbol="GOOGL")
    
    history, prices = get_agent_history("test")
    if history:
        plot_cumulative_returns(history, prices)
        plot_trading_actions(history)
        
    print("\nAll figures generated successfully in results/plots/.")

if __name__ == "__main__":
    run()
