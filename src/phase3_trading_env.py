"""
Phase 3 - Trading Environment: HybridTradingEnv(gym.Env)
========================================================
Gym environment với 7 hành động rời rạc và Reward Shaping 5 thành phần.
Bao gồm Cooldown Period và Churning Penalty chống overtrading.
Đọc observations.npy (không chạy inference trong env).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


class HybridTradingEnv(gym.Env):
    """
    Môi trường giao dịch cho Hybrid DQN-PPO Agent.

    Action Space (Discrete 7):
        0: Hold       - Không làm gì
        1: Buy 25%    - Mua 25% balance
        2: Buy 50%    - Mua 50% balance
        3: Buy 100%   - Mua 100% balance (All-in)
        4: Sell 25%   - Bán 25% holdings
        5: Sell 50%   - Bán 50% holdings
        6: Sell 100%  - Bán hết (Close position)

    Reward Shaping:
        1. Base PnL reward
        2. Action-Intensity (thưởng thuận xu hướng)
        3. HMM Regime Penalty (phạt ngược sóng)
        4. Risk Penalty (drawdown + volatility)
        5. Churning Penalty (phạt đổi chiều Buy↔Sell liên tục)
    """
    metadata = {"render_modes": ["human"]}

    # Tỷ lệ mua/bán cho mỗi action
    ACTION_MAP = {
        0: ("hold", 0.0),
        1: ("buy", 0.25),
        2: ("buy", 0.50),
        3: ("buy", 1.00),
        4: ("sell", 0.25),
        5: ("sell", 0.50),
        6: ("sell", 1.00),
    }

    def __init__(self, split="train", symbol_filter=None):
        super().__init__()

        # Load pre-computed observations
        obs_path = config.STATES_DIR / f"{split}_observations.npy"
        prices_path = config.STATES_DIR / f"{split}_prices.npy"

        self.observations = np.load(obs_path).astype(np.float32)
        self.prices = np.load(prices_path).astype(np.float64)

        # Always load symbols to track boundaries
        self.symbols = np.load(config.STATES_DIR / f"{split}_symbols.npy", allow_pickle=True)

        # Symbol filtering (nếu cần train riêng từng mã)
        if symbol_filter:
            mask = self.symbols == symbol_filter
            self.observations = self.observations[mask]
            self.prices = self.prices[mask]
            self.symbols = self.symbols[mask]

        self.n_steps = len(self.observations)
        self.market_dim = self.observations.shape[1]

        # State = market features (31) + portfolio state (3)
        self.state_dim = self.market_dim + config.STATE_DIM_PORTFOLIO
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.NUM_ACTIONS)

        # Portfolio state
        self.initial_balance = config.INITIAL_BALANCE
        self.transaction_cost = config.TRANSACTION_COST

        # Penalty scale: normalize penalties theo episode length
        # 6000 steps là baseline (Quick Mode), episode dài hơn sẽ giảm penalties
        _BASELINE_STEPS = 6000
        self.penalty_scale = min(1.0, _BASELINE_STEPS / max(self.n_steps, 1))

        self.reset()

    def _get_state(self):
        """Ghép market features + portfolio state."""
        market_state = self.observations[self.current_step]

        # Portfolio state: normalized
        portfolio_state = np.array([
            self.balance / self.initial_balance,         # Normalized balance
            self.shares * self.prices[self.current_step] / self.initial_balance,  # Holdings value
            (self._portfolio_value() - self.initial_balance) / self.initial_balance,  # Unrealized PnL
        ], dtype=np.float32)

        return np.concatenate([market_state, portfolio_state])

    def _portfolio_value(self):
        """Tổng giá trị portfolio: cash + stock."""
        return self.balance + self.shares * self.prices[self.current_step]

    def _get_hmm_regime(self):
        """Lấy regime từ HMM one-hot trong state (index 21:24)."""
        # OHLCV(5) + Tech(7) + EMAT(9) = 21 → HMM one-hot ở [21:24]
        hmm_onehot = self.observations[self.current_step, 21:24]
        regime = np.argmax(hmm_onehot)  # 0=Bearish, 1=Sideways, 2=Bullish
        return regime

    def _get_emat_trend(self):
        """Lấy trend dự đoán 1d từ EMAT probs (index 12:15)."""
        # OHLCV(5) + Tech(7) = 12 → EMAT 1d probs ở [12:15]
        emat_1d = self.observations[self.current_step, 12:15]
        trend = np.argmax(emat_1d)  # 0=Giảm, 1=Tăng, 2=Đi ngang
        return trend

    def _get_garch_volatility(self):
        """Lấy normalized conditional volatility (index 27)."""
        # OHLCV(5) + Tech(7) + EMAT(9) + HMM(6) = 27
        return self.observations[self.current_step, 27]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares = 0.0
        self.current_step = 0
        self.peak_value = self.initial_balance
        self.total_trades = 0
        self.profitable_trades = 0
        self.trade_history = []
        self.cooldown_counter = 0
        self.last_trade_type = None  # "buy" / "sell" / None
        self.entry_price = None      # Dùng cho Dynamic Exit (Trailing stop)
        self.idle_cash_steps = 0     # Đếm số step liên tiếp giữ 100% cash

        return self._get_state(), {}

    def step(self, action):
        """Step với discrete action (0-6). Backward compat cho baseline."""
        action_type, fraction = self.ACTION_MAP[action]
        return self._execute_step(action_type, fraction, action_idx=action)

    def step_hierarchical(self, direction, fraction=0.5):
        """
        Step với Hierarchical action: DQN chọn direction, PPO chọn fraction.

        Args:
            direction: int — 0=Buy, 1=Hold, 2=Sell
            fraction: float ∈ (0, 1) — tỷ lệ vốn (continuous từ PPO)
        """
        DIRECTION_MAP = {0: "buy", 1: "hold", 2: "sell"}
        action_type = DIRECTION_MAP[direction]

        if action_type == "hold":
            fraction = 0.0

        # Clamp fraction vào khoảng hợp lệ
        fraction = max(config.PPO_ALLOCATION_MIN, min(config.PPO_ALLOCATION_MAX, fraction))

        # Map ngược lại action_idx gần nhất để ghi log (chỉ dùng cho trade_history)
        if action_type == "hold":
            action_idx = 0
        elif action_type == "buy":
            if fraction <= 0.375:
                action_idx = 1
            elif fraction <= 0.75:
                action_idx = 2
            else:
                action_idx = 3
        else:  # sell
            if fraction <= 0.375:
                action_idx = 4
            elif fraction <= 0.75:
                action_idx = 5
            else:
                action_idx = 6

        return self._execute_step(action_type, fraction, action_idx=action_idx)

    def _execute_step(self, action_type, fraction, action_idx=0):
        """Logic thực thi giao dịch dùng chung cho step() và step_hierarchical()."""
        current_price = self.prices[self.current_step]
        prev_value = self._portfolio_value()

        trade_value = 0.0
        trade_executed = False

        # 1. Bắt sự kiện chuyển mã cổ phiếu (Boundary Bug Fix)
        is_boundary = False
        if self.current_step < self.n_steps - 1:
            if self.symbols[self.current_step] != self.symbols[self.current_step + 1]:
                is_boundary = True

        # 2. Dynamic Exit (Trailing Stop qua GARCH Volatility hoặc Stop Loss cứng 12%)
        garch_vol = self._get_garch_volatility()
        is_force_close = False
        if self.shares > 0 and self.entry_price is not None:
            loss_pct = (self.entry_price - current_price) / self.entry_price
            if loss_pct > 1.0 * (garch_vol + 1e-5) or loss_pct > 0.12:
                action_type = "sell"
                fraction = 1.0  # Force close
                is_force_close = True

        # Nếu là boundary, bắt buộc phải force close toàn bộ danh mục để tránh nhảy giá
        if is_boundary and self.shares > 0:
            action_type = "sell"
            fraction = 1.0
            is_force_close = True

        # Cooldown: ép hành động thành Hold nếu đang trong thời gian "làm mát"
        if self.cooldown_counter > 0:
            if action_type != "hold" and not is_force_close:
                action_type = "hold"
                fraction = 0.0
            self.cooldown_counter -= 1

        # Guard: Chặn Buy khi cash không đủ (<=0 hoặc quá nhỏ để giao dịch)
        min_trade_threshold = 1.0  # Tối thiểu $1 để tránh lệnh vô nghĩa
        if action_type == "buy" and fraction > 0 and self.balance > min_trade_threshold:
            invest_amount = self.balance * fraction
            cost = invest_amount * self.transaction_cost
            shares_bought = (invest_amount - cost) / current_price
            self.balance -= invest_amount
            self.shares += shares_bought
            trade_value = invest_amount
            self.total_trades += 1
            trade_executed = True

            # Cập nhật giá vốn bình quân (Entry price)
            if self.entry_price is None:
                self.entry_price = current_price
            else:
                total_value = (self.shares - shares_bought) * self.entry_price + invest_amount
                self.entry_price = total_value / self.shares

        elif action_type == "sell" and fraction > 0 and self.shares > 0:
            shares_to_sell = self.shares * fraction
            revenue = shares_to_sell * current_price
            cost = revenue * self.transaction_cost
            self.balance += revenue - cost
            self.shares -= shares_to_sell
            trade_value = revenue
            self.total_trades += 1
            trade_executed = True
            # Kiểm tra lời/lỗ
            if revenue > shares_to_sell * self.prices[max(0, self.current_step - 1)]:
                self.profitable_trades += 1

            if self.shares < 1e-5:  # Close position
                self.shares = 0.0
                self.entry_price = None

        # Kích hoạt cooldown sau khi trade thành công
        if trade_executed:
            self.cooldown_counter = config.TRADE_COOLDOWN_STEPS

        # Reset tài khoản nếu qua boundary để tránh mang tiền/cooldown sang mã mới
        if is_boundary:
            # Tính PnL của step cuối cùng TRƯỚC khi reset tài khoản
            new_value_before_reset = self.balance + self.shares * current_price
            base_pnl = (new_value_before_reset - prev_value) / prev_value

            self.balance = self.initial_balance
            self.shares = 0.0
            self.entry_price = None
            self.cooldown_counter = 0
            self.peak_value = self.initial_balance

            self.current_step += 1
            terminated = self.current_step >= self.n_steps - 1
            truncated = False
            new_value = self.balance + self.shares * self.prices[self.current_step]
        else:
            self.current_step += 1
            terminated = self.current_step >= self.n_steps - 1
            truncated = False
            new_value = self.balance + self.shares * self.prices[self.current_step]
            base_pnl = (new_value - prev_value) / prev_value

        # 2. Action-Intensity Reward (thưởng thuận xu hướng EMAT)
        action_intensity = 0.0
        emat_trend = self._get_emat_trend()
        if not is_force_close:
            if action_type == "buy" and emat_trend == 1:    # Mua khi EMAT dự đoán Tăng
                action_intensity = 0.005 * fraction
            elif action_type == "sell" and emat_trend == 0:  # Bán khi EMAT dự đoán Giảm
                action_intensity = 0.005 * fraction
            elif action_type == "buy" and emat_trend == 0:   # Mua khi Giảm → phạt
                action_intensity = -0.003 * fraction
            elif action_type == "sell" and emat_trend == 1:  # Bán khi Tăng → phạt
                action_intensity = -0.003 * fraction

        # 3. HMM Regime Penalty
        regime_penalty = 0.0
        regime = self._get_hmm_regime()
        if not is_force_close:
            if action_type == "buy" and regime == 0:     # Mua trong Bearish
                regime_penalty = -0.005
            elif action_type != "hold" and regime == 1:  # Phạt giao dịch trong Sideways
                regime_penalty = -0.008
            # Bỏ phạt bán trong Bullish — HMM regime detection chưa chính xác 100%

        # 4. Risk Penalty (drawdown + volatility + position sizing)
        self.peak_value = max(self.peak_value, new_value)
        drawdown = (self.peak_value - new_value) / self.peak_value
        
        # Drawdown Penalty giảm nhẹ (v3): multiplier 8 thay vì 20
        # DD 8% -> phạt 0.41, DD 15% -> phạt 1.44, DD 30% -> phạt 5.76
        if drawdown > config.REWARD_DRAWDOWN_THRESHOLD:
            drawdown_penalty = - ((drawdown * 8.0) ** 2)
        else:
            drawdown_penalty = 0.0
        
        vol_penalty = -garch_vol * config.REWARD_VOLATILITY_LAMBDA
        
        # Position Sizing Penalty: phạt All-in (Buy 100%) khi volatility cao
        position_penalty = 0.0
        if action_type == "buy" and fraction >= 1.0 and garch_vol > 0.5:
            position_penalty = -0.010 * garch_vol
        
        risk_penalty = drawdown_penalty + vol_penalty + position_penalty

        # 5. Churning Penalty — phạt đổi chiều vị thế liên tục (Buy→Sell hoặc Sell→Buy)
        churning_penalty = 0.0
        if trade_executed and not is_force_close:
            current_trade_type = action_type
            if (self.last_trade_type == "buy" and current_trade_type == "sell") or \
               (self.last_trade_type == "sell" and current_trade_type == "buy"):
                churning_penalty = -0.020  # Raw value × CHURNING_PENALTY_W = -0.001
            self.last_trade_type = current_trade_type

        # 6. Hold Bonus — thưởng giữ vị thế khi đúng lúc
        hold_bonus = 0.0
        if action_type == "hold" and self.shares > 0:
            # Bonus 1: EMAT dự đoán tăng (không cần HMM đồng thuận)
            if emat_trend == 1:
                hold_bonus += config.REWARD_HOLD_UPTREND_BONUS
            # Bonus 2: Portfolio đang lãi → thưởng giữ
            if new_value > self.initial_balance:
                hold_bonus += config.REWARD_HOLD_UPTREND_BONUS * 0.5
        elif action_type == "hold" and self.shares == 0:
            # Luôn phạt nhẹ khi Hold mà không có position → khuyến khích hành động
            hold_bonus = -config.REWARD_HOLD_UPTREND_BONUS * 0.5
            # Phạt mạnh hơn nếu EMAT dự đoán tăng (bỏ lỡ cơ hội rõ ràng)
            if emat_trend == 1:
                hold_bonus = -config.REWARD_HOLD_UPTREND_BONUS * 1.0

        # 7. Idle Penalty — phạt tích lũy khi giữ 100% cash quá lâu trong thị trường thuận lợi
        # Ablation Study cho thấy MSFT, FPT, XOM bị "Decision Freezing" ở Full Pipeline
        # do xung đột tín hiệu khiến PPO chọn không hành động.
        idle_penalty = 0.0
        if self.shares < 1e-5 and not trade_executed:
            self.idle_cash_steps += 1
        else:
            self.idle_cash_steps = 0

        # Phạt tăng dần (progressive) sau 30 steps liên tiếp giữ cash
        _IDLE_GRACE_PERIOD = 30
        if self.idle_cash_steps > _IDLE_GRACE_PERIOD:
            excess = self.idle_cash_steps - _IDLE_GRACE_PERIOD
            # Phạt nặng hơn nếu thị trường đang Bullish (HMM) hoặc EMAT dự đoán Tăng
            if regime == 2 or emat_trend == 1:  # Bullish hoặc EMAT tăng
                idle_penalty = -config.REWARD_IDLE_PENALTY_W * min(excess / 50.0, 1.0)
            else:
                # Vẫn phạt nhẹ dù thị trường không Bullish để tránh ngủ đông hoàn toàn
                idle_penalty = -config.REWARD_IDLE_PENALTY_W * 0.3 * min(excess / 50.0, 1.0)

        # Tổng hợp reward — penalties được scale theo episode length
        penalty_sum = (
            config.REWARD_ACTION_INTENSITY_W * action_intensity
            + config.REWARD_HMM_PENALTY_W * regime_penalty
            + config.REWARD_RISK_PENALTY_W * risk_penalty
            + config.REWARD_CHURNING_PENALTY_W * churning_penalty
        )
        reward = (
            config.REWARD_BASE_PNL_W * base_pnl
            + self.penalty_scale * penalty_sum  # Scale penalties theo n_steps
            + hold_bonus
            + idle_penalty
        )

        # Ghi lại trade
        self.trade_history.append({
            "step": self.current_step,
            "action": action_idx,
            "price": current_price,
            "balance": self.balance,
            "shares": self.shares,
            "portfolio_value": new_value,
            "reward": reward,
        })

        info = {
            "portfolio_value": new_value,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
        }

        return self._get_state(), reward, terminated, truncated, info

    def get_performance_metrics(self):
        """Tính toán metrics đánh giá toàn bộ episode."""
        if not self.trade_history:
            return {}

        values = [t["portfolio_value"] for t in self.trade_history]
        returns = np.diff(values) / values[:-1] if len(values) > 1 else [0]

        total_return = (values[-1] - self.initial_balance) / self.initial_balance
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        annualized_return = total_return * (252 / max(len(values), 1))
        calmar = annualized_return / (max_drawdown + 1e-8)
        win_rate = self.profitable_trades / max(self.total_trades, 1)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "total_trades": self.total_trades,
            "final_value": values[-1],
        }


if __name__ == "__main__":
    # Test nhanh environment
    print("Testing HybridTradingEnv...")
    env = HybridTradingEnv(split="train")
    state, _ = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State dim: {env.state_dim}")

    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    metrics = env.get_performance_metrics()
    print(f"Total reward: {total_reward:.4f}")
    print(f"Metrics: {metrics}")
