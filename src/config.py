"""
Cấu hình tập trung cho toàn bộ MLOps Training Pipeline.
Mọi hyperparameters, đường dẫn, và hằng số đều được quản lý tại đây.
"""
import os
import sys
from pathlib import Path

# Fix encoding cho Windows terminal (cp1258 không hỗ trợ đầy đủ Unicode)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Tránh xung đột OpenMP giữa PyTorch và NumPy/MKL trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Giới hạn threads OpenMP → tránh rò rỉ bộ nhớ với KMeans/sklearn trên Windows
os.environ["OMP_NUM_THREADS"] = "5"

# =============================================================
# ĐƯỜNG DẪN GỐC
# =============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SRC = PROJECT_ROOT / "data_src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATES_DIR = PROJECT_ROOT / "states"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

# Tạo thư mục nếu chưa tồn tại
for d in [DATA_DIR / "train", DATA_DIR / "val", DATA_DIR / "test",
          MODELS_DIR, STATES_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================
# NGUỒN DỮ LIỆU
# =============================================================
# Đường dẫn tới các loại dữ liệu trong data_src
PRICES_SRC = DATA_SRC / "data" / "prices"
EMAT_DATASETS_SRC = DATA_SRC / "data" / "emat_datasets"
NEWS_EN_SRC = DATA_SRC / "data" / "news_en"       # raw_clean cũng có
NEWS_VI_SRC = DATA_SRC / "data" / "news_vi"        # raw_clean cũng có
SENTIMENT_VI_SRC = DATA_SRC / "data" / "sentiment_vi"

# Nếu cần fallback sang raw_clean (đầy đủ hơn)
RAW_CLEAN_PRICES = DATA_SRC / "raw_clean" / "prices"
RAW_CLEAN_NEWS_EN = DATA_SRC / "raw_clean" / "news_en"
RAW_CLEAN_NEWS_VI = DATA_SRC / "raw_clean" / "news_vi"
RAW_CLEAN_SENTIMENT_VI = DATA_SRC / "raw_clean" / "sentiment_vi"

# =============================================================
# MÃ CỔ PHIẾU
# =============================================================
SYMBOLS = ["AAPL", "MSFT", "JPM", "XOM", "VCB", "HPG", "FPT", "GOOGL", "META", "VNM"]

# =============================================================
# QUICK MODE — Bật để test nhanh trước khi chạy full pipeline
# Train 2 mã (AAPL, JPM) × ít episodes → validate kiến trúc trong ~5-10 phút
# =============================================================
QUICK_MODE = False
QUICK_SYMBOLS = ["AAPL", "JPM"]
QUICK_DQN_EPISODES = 100
QUICK_PPO_EPISODES = 50
QUICK_EPSILON_DECAY = 100_000

# =============================================================
# PER-SYMBOL MODE — Mỗi mã train agent riêng, tránh conflict
# =============================================================
PER_SYMBOL_MODE = True  # ← ĐỔI THÀNH False để train combined
PER_SYMBOL_DQN_EPISODES = 150    # ~3000 steps/mã × 150 eps = 450k steps
PER_SYMBOL_PPO_EPISODES = 80     # PPO converge nhanh hơn DQN
PER_SYMBOL_EPSILON_DECAY = 200_000

# =============================================================
# THAM SỐ CHIA DỮ LIỆU (PHASE 1)
# =============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# =============================================================
# EMAT TARGET SCALE (REGRESSION)
# =============================================================
EMAT_TARGET_SCALE = 100.0   # Nhân với 100 để tránh vanishing gradient cho phần trăm return
PREDICTION_HORIZONS = [1, 3, 5]

# =============================================================
# EMAT HYPERPARAMETERS (Time2Vec + Transformer)
# Tham khảo: Bui et al. (2025), EUSIPCO 2025
# =============================================================
EMAT_INPUT_DIM = 12          # OHLCV(5) + RSI, MACD, Signal, BB_H, BB_L, ATR, K(7)
EMAT_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "MACD", "Signal_Line", "BB_High", "BB_Low", "ATR", "K_Line"
]
EMAT_D_MODEL = 64           # Embedding dimension
EMAT_NHEAD = 8               # Attention heads (128/8=16 per head)
EMAT_NUM_LAYERS = 1          # Transformer encoder layers
EMAT_DIM_FF = 256            # Feed-forward dimension (2×d_model)
EMAT_DROPOUT = 0.35          # Tăng mạnh dropout để chống overfitting (461K params vs 6420 samples)
EMAT_WINDOW_SIZE = 15        # Giảm xuống 15 ngày để EMAT phản ứng nhanh với ngắn hạn
EMAT_BATCH_SIZE = 64
EMAT_EPOCHS = 100            # Tăng gấp đôi cho model hội tụ tốt hơn
EMAT_LR = 5e-4
EMAT_WEIGHT_DECAY = 1e-2     # Tăng L2 regularization mạnh hơn để ép generalization
EMAT_PATIENCE = 15           # Early stopping patience (tương ứng epoch dài hơn)

# =============================================================
# HMM HYPERPARAMETERS
# =============================================================
HMM_MIN_COMPONENTS = 3       # Ép tối thiểu 3 states (Bearish/Sideways/Bullish) cho nhất quán
HMM_MAX_COMPONENTS = 3
HMM_COVARIANCE_TYPE = "diag" # Dùng diag để dễ hội tụ, giảm thiểu lỗi kẹt EM
HMM_N_ITER = 1000
HMM_RANDOM_STATE = 42
HMM_SMOOTHING_WINDOW = 5     # Dùng EMA(5) để làm mượt nhiễu trước khi đưa vào HMM
HMM_PERSISTENCE_PRIOR = 0.95 # Ép ma trận chuyển đổi ưu tiên duy trì trạng thái cũ

# =============================================================
# GARCH HYPERPARAMETERS
# =============================================================
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "Normal"
GARCH_RETURNS_SCALE = 100    # Nhân 100 cho optimization tốt hơn
GARCH_VOL_ROLLING_WINDOW = 60   # Cửa sổ lăn 60 ngày cho Z-score volatility
GARCH_VOL_ZSCORE_HIGH = 1.0     # Z > 1 → Volatility Cao
GARCH_VOL_ZSCORE_LOW = -1.0     # Z < -1 → Volatility Thấp

# =============================================================
# SENTIMENT HYPERPARAMETERS (XLM-RoBERTa + CNN + BiGRU)
# Tham khảo: XLM-RoBERTa cross-lingual transfer papers
# =============================================================
SENTIMENT_BACKBONE = "xlm-roberta-base"
SENTIMENT_NUM_CLASSES = 3     # Tích cực / Tiêu cực / Trung tính
SENTIMENT_FREEZE_LAYERS = 6   # Freeze 6/12 layers đầu
SENTIMENT_CNN_OUT = 256
SENTIMENT_CNN_KERNEL = 3
SENTIMENT_GRU_HIDDEN = 128
SENTIMENT_MAX_LENGTH = 256    # Dài hơn để capture đủ context tài chính
SENTIMENT_LR_BACKBONE = 2e-5  # Thấp cho pre-trained
SENTIMENT_LR_HEAD = 5e-4      # Cao cho CNN+GRU head
SENTIMENT_BATCH_SIZE = 16     # Nhỏ vì XLM-RoBERTa nặng
SENTIMENT_EPOCHS = 10
# Class weights: nhấn mạnh Tiêu cực (tin xấu ảnh hưởng giá mạnh hơn)
SENTIMENT_CLASS_WEIGHTS = [1.0, 2.0, 0.8]  # [Tích cực, Tiêu cực, Trung tính]
SENTIMENT_SCORE_THRESHOLD = 0.2  # Ngưỡng phân loại sentiment_score

# Cột dữ liệu sentiment VI
SENTIMENT_VI_COLS = {
    "date": "Date",        # raw_clean dùng "Date", data dùng "date"
    "header": "Header",    # raw_clean dùng "Header", data dùng "header"
    "content": "Content",  # raw_clean dùng "Content", data dùng "content"
    "score": "sentiment_score",
}

# Cột dữ liệu news EN
NEWS_EN_COLS = {
    "title": "title",
    "date": "date",
    "stock": "stock",
}

# =============================================================
# STATE VECTOR DIMENSIONS
# =============================================================
# OHLCV normalized (5) + Tech Indicators (7) + EMAT returns 3 heads×1 (3)
# + HMM regime one-hot (3) + GARCH vol (1) + GARCH cat (3)
# + Sentiment probs (3) + Portfolio state (3) = 28
STATE_DIM_MARKET = 28        # Không tính portfolio state
STATE_DIM_PORTFOLIO = 3      # balance, holdings, unrealized_pnl
STATE_DIM_TOTAL = STATE_DIM_MARKET + STATE_DIM_PORTFOLIO

# =============================================================
# TRADING ENVIRONMENT (Phase 3)
# =============================================================
NUM_ACTIONS = 7               # Backward compat: Hold, Buy25, Buy50, Buy100, Sell25, Sell50, Sell100
INITIAL_BALANCE = 100_000.0
TRANSACTION_COST = 0.001      # 0.1% phí giao dịch
TRADE_COOLDOWN_STEPS = 20     # Tăng cooldown → giảm overtrading (trước: 10)

# =============================================================
# HIERARCHICAL ACTION SPACE (Phase 4 — DQN Strategic + PPO Tactical)
# =============================================================
# DQN chọn direction: Buy=0, Hold=1, Sell=2
DQN_NUM_DIRECTIONS = 3
DIRECTION_BUY = 0
DIRECTION_HOLD = 1
DIRECTION_SELL = 2
# PPO chọn allocation liên tục qua Beta distribution
PPO_ALLOCATION_MIN = 0.05    # Tối thiểu 5% để tránh lệnh quá nhỏ vô nghĩa
PPO_ALLOCATION_MAX = 1.0     # Tối đa 100%
# Allocation mặc định khi PPO chưa train (giai đoạn DQN)
DQN_DEFAULT_FRACTION = 0.50

# Reward shaping weights (v3.4 — Best of all iterations)
# PnL=3.0 (proven Buy 60%) + penalties thấp + hold bonus mạnh
REWARD_BASE_PNL_W = 3.0               # Proven ở v3.0 → Buy 60%
REWARD_ACTION_INTENSITY_W = 0.01       # Giữ nguyên
REWARD_HMM_PENALTY_W = 0.003           # Nhẹ — HMM chỉ gợi ý
REWARD_RISK_PENALTY_W = 0.02           # Giữa v3.0 (0.03) và v3.3 (0.01)
REWARD_DRAWDOWN_THRESHOLD = 0.10       # Phạt DD > 10%
REWARD_VOLATILITY_LAMBDA = 0.005       # Nhẹ
REWARD_CHURNING_PENALTY_W = 0.03       # Nhẹ
REWARD_HOLD_UPTREND_BONUS = 0.020      # Giữ mạnh
REWARD_IDLE_PENALTY_W = 0.015           # Phạt tích lũy khi giữ 100% cash quá lâu (chống Freezing)

# =============================================================
# DQN HYPERPARAMETERS (Phase 4 - Giai đoạn 1: Strategic)
# DQN học KHI NÀO mua/bán: Buy(0) / Hold(1) / Sell(2)
# =============================================================
DQN_LR = 5e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY_STEPS = 800_000
DQN_REPLAY_BUFFER_SIZE = 100_000
DQN_TARGET_UPDATE_FREQ = 500
DQN_BATCH_SIZE = 128
DQN_EPISODES = 500                  # DQN học Buy/Hold/Sell strategy

# =============================================================
# PPO HYPERPARAMETERS (Phase 4 - Giai đoạn 2: Tactical)
# PPO học BAO NHIÊU % vốn qua Beta distribution → allocation ∈ (0,1)
# =============================================================
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RATIO = 0.15                # Clip chặt → cập nhật conservative
PPO_ENTROPY_COEFF = 0.02             # Exploration cho Beta distribution
PPO_VALUE_COEFF = 0.5
PPO_BATCH_SIZE = 256
PPO_EPISODES = 300                   # PPO tối ưu continuous allocation
PPO_EPOCHS_PER_UPDATE = 4

# =============================================================
# SHARED ENCODER (Phase 4)
# =============================================================
ENCODER_HIDDEN_1 = 256
ENCODER_HIDDEN_2 = 128        # Output dimension
DQN_HEAD_HIDDEN = 128
PPO_HEAD_HIDDEN = 128

# =============================================================
# DEVICE (lazy-load torch để Phase 1 không cần cài torch)
# =============================================================
RANDOM_SEED = 42

def get_device():
    """Lazy-load torch và trả về device."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Property-like access: import khi cần
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"


def print_config():
    """In toàn bộ cấu hình để xác nhận."""
    print("=" * 60)
    print("CẤU HÌNH PIPELINE")
    print("=" * 60)
    print(f"  Mã cổ phiếu     : {SYMBOLS}")
    print(f"  Device           : {DEVICE}")
    print(f"  Split ratio      : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  EMAT input_dim   : {EMAT_INPUT_DIM}")
    print(f"  EMAT d_model     : {EMAT_D_MODEL}")
    print(f"  EMAT window      : {EMAT_WINDOW_SIZE}")
    print(f"  Prediction heads : {PREDICTION_HORIZONS}")
    print(f"  State dim        : {STATE_DIM_TOTAL}")
    print(f"  Sentiment model  : {SENTIMENT_BACKBONE}")
    print(f"  --- Hierarchical Agent ---")
    print(f"  Quick Mode       : {'⚡ ON' if QUICK_MODE else '🏋️ OFF'}")
    if QUICK_MODE:
        print(f"  Quick Symbols    : {QUICK_SYMBOLS}")
    print(f"  DQN directions   : {DQN_NUM_DIRECTIONS} (Buy/Hold/Sell)")
    print(f"  PPO allocation   : continuous Beta(α,β) ∈ (0,1)")
    print(f"  DQN episodes     : {QUICK_DQN_EPISODES if QUICK_MODE else DQN_EPISODES}")
    print(f"  PPO episodes     : {QUICK_PPO_EPISODES if QUICK_MODE else PPO_EPISODES}")
    print(f"  Default fraction : {DQN_DEFAULT_FRACTION*100:.0f}%")
    print(f"  Reward version   : v3 (PnL×{REWARD_BASE_PNL_W}, DD threshold={REWARD_DRAWDOWN_THRESHOLD})")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
