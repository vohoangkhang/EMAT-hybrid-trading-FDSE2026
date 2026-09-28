"""
Phase 2 - Task 2.1: Huấn luyện EMAT (Enhanced Multi-Aspect Attention Transformer)
==================================================================================
Kiến trúc: Time2Vec → TransformerEncoder → Multi-Head Attention → Multi-Task Output
Multi-task: Dự đoán xu hướng 1d, 3d, 5d đồng thời.
Input: 12 features (OHLCV + 7 Technical Indicators), window_size=20.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import advanced_metrics as am


# =============================================================
# KIẾN TRÚC MÔ HÌNH
# =============================================================
class Time2Vec(nn.Module):
    """
    Time2Vec: Học biểu diễn thời gian với cả thành phần tuyến tính (trend)
    và tuần hoàn (periodic patterns) thông qua hàm sin.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Thành phần tuyến tính (1 chiều)
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        # Thành phần tuần hoàn (out_features - 1 chiều)
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau):
        # tau: (batch, seq_len, in_features)
        linear = torch.matmul(tau, self.w0) + self.b0       # (batch, seq, 1)
        periodic = torch.sin(torch.matmul(tau, self.w) + self.b)  # (batch, seq, out-1)
        return torch.cat([linear, periodic], dim=-1)          # (batch, seq, out_features)


class EMAT(nn.Module):
    """
    Enhanced Multi-Aspect Attention Transformer (Multi-Task Version)
    Dự đoán đồng thời xu hướng 1d, 3d, 5d.
    """
    def __init__(
        self,
        input_dim: int = config.EMAT_INPUT_DIM,
        d_model: int = config.EMAT_D_MODEL,
        nhead: int = config.EMAT_NHEAD,
        num_layers: int = config.EMAT_NUM_LAYERS,
        dim_feedforward: int = config.EMAT_DIM_FF,
        dropout: float = config.EMAT_DROPOUT,
        horizons: list = None,
    ):
        super().__init__()
        self.horizons = horizons or config.PREDICTION_HORIZONS

        # 1. Time2Vec Embedding
        self.time_embedding = Time2Vec(input_dim, d_model)

        # 2. Positional Encoding (learnable)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.EMAT_WINDOW_SIZE, d_model) * 0.02
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Multi-Aspect Attention (Global Attention Pooling)
        self.attention_query = nn.Linear(d_model, d_model)
        self.attention_key = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)  # Dropout cho attention weights

        # 5. Layer Normalization + Dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.context_dropout = nn.Dropout(dropout)  # Dropout sau context vector

        # 6. Multi-Task Regression Heads (1 head per horizon)
        self.classifiers = nn.ModuleDict()
        for h in self.horizons:
            self.classifiers[f"head_{h}d"] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        # 1. Time2Vec Embedding
        embedded = self.time_embedding(x)  # (batch, seq, d_model)

        # 2. Positional Encoding
        embedded = embedded + self.pos_encoding[:, :x.size(1), :]

        # 3. Transformer
        encoded = self.transformer(embedded)  # (batch, seq, d_model)

        # 4. Multi-Aspect Attention Pooling
        Q = self.attention_query(encoded[:, -1:, :])  # Dùng step cuối làm query
        K = self.attention_key(encoded)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (encoded.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, seq)
        attn_weights = self.attn_dropout(attn_weights)  # Regularize attention
        context = torch.bmm(attn_weights, encoded).squeeze(1)  # (batch, d_model)

        # 5. Normalize + Dropout
        context = self.layer_norm(context)
        context = self.context_dropout(context)

        # 6. Multi-task heads
        outputs = {}
        for h in self.horizons:
            outputs[f"{h}d"] = self.classifiers[f"head_{h}d"](context).squeeze(-1)

        return outputs


# =============================================================
# CHUẨN BỊ DỮ LIỆU
# =============================================================
def prepare_data(split: str = "train") -> tuple:
    """
    Load và chuẩn bị dữ liệu cho EMAT từ tập split đã chia.
    Trả về: (X_all, y_all, dates_all) cho tất cả mã cổ phiếu gộp lại.
    y_all là dict {horizon: labels}
    """
    all_X, all_dates = [], []
    all_y = {h: [] for h in config.PREDICTION_HORIZONS}

    for symbol in config.SYMBOLS:
        path = config.DATA_DIR / split / f"{symbol}_emat.csv"
        if not path.exists():
            print(f"  [WARN] Không tìm thấy {path}")
            continue

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])

        # Lấy features
        features = df[config.EMAT_FEATURE_COLS].values.astype(np.float32)

        # Lấy mean/std từ tập Train để tránh Data Leakage
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

        # Tạo nhãn cho mỗi horizon (Lợi nhuận % tương lai)
        labels = {}
        for h in config.PREDICTION_HORIZONS:
            future_returns = df["Close"].pct_change(periods=h).shift(-h).values
            labels[h] = future_returns * config.EMAT_TARGET_SCALE

        # Tạo sliding windows
        window = config.EMAT_WINDOW_SIZE
        for i in range(len(features) - window):
            end_idx = i + window - 1  # Index cuối của window

            # Kiểm tra nhãn hợp lệ cho TẤT CẢ horizons
            valid = True
            for h in config.PREDICTION_HORIZONS:
                if end_idx >= len(labels[h]) or np.isnan(
                    df["Close"].pct_change(periods=h).shift(-h).iloc[end_idx]
                    if end_idx < len(df) else np.nan
                ):
                    valid = False
                    break

            if not valid:
                continue

            all_X.append(features[i:i + window])
            all_dates.append(df["Date"].iloc[end_idx])
            for h in config.PREDICTION_HORIZONS:
                all_y[h].append(labels[h][end_idx])

    X = torch.tensor(np.array(all_X), dtype=torch.float32)
    y = {h: torch.tensor(np.array(all_y[h]), dtype=torch.float32)
         for h in config.PREDICTION_HORIZONS}
    return X, y, all_dates


# =============================================================
# HÀM HUẤN LUYỆN
# =============================================================
def train():
    """Huấn luyện EMAT model trên tập train với early stopping trên tập val."""
    print("\n" + "=" * 60)
    print("  PHASE 2.1: HUẤN LUYỆN EMAT (Multi-Task Transformer)")
    print("=" * 60)

    device = config.DEVICE
    print(f"  Device: {device}")

    # 1. Load dữ liệu
    print("\n  Đang load dữ liệu train...")
    X_train, y_train, _ = prepare_data("train")
    print(f"  Train: {X_train.shape[0]} samples, window={X_train.shape[1]}, "
          f"features={X_train.shape[2]}")

    print("  Đang load dữ liệu val...")
    X_val, y_val, _ = prepare_data("val")
    print(f"  Val  : {X_val.shape[0]} samples")

    # DataLoader
    train_tensors = [X_train] + [y_train[h] for h in config.PREDICTION_HORIZONS]
    val_tensors = [X_val] + [y_val[h] for h in config.PREDICTION_HORIZONS]

    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    train_loader = DataLoader(
        train_dataset, batch_size=config.EMAT_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.EMAT_BATCH_SIZE, shuffle=False
    )

    # 3. Model + Optimizer + Scheduler
    model = EMAT().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Tổng parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.EMAT_LR,
        weight_decay=config.EMAT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss functions: Huber Loss chống nhiễu
    criteria = {h: nn.HuberLoss(delta=1.0) for h in config.PREDICTION_HORIZONS}

    # 4. Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.EMAT_EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0
        epoch_total = 0

        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            X_batch = batch[0]
            y_batch = {h: batch[i + 1] for i, h in enumerate(config.PREDICTION_HORIZONS)}

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Multi-task loss: trung bình loss các heads
            loss = sum(
                criteria[h](outputs[f"{h}d"], y_batch[h])
                for h in config.PREDICTION_HORIZONS
            ) / len(config.PREDICTION_HORIZONS)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            epoch_total += X_batch.size(0)

        scheduler.step()

        # --- Validate ---
        val_metrics = evaluate_model(model, val_loader, criteria, device)

        # Avg Loss across horizons
        avg_val_loss = np.mean([val_metrics[h]["loss"] for h in config.PREDICTION_HORIZONS])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{config.EMAT_EPOCHS} | "
                  f"Train Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping (Minimize Validation Loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODELS_DIR / "emat_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.EMAT_PATIENCE:
                print(f"\n  Early stopping tại epoch {epoch+1}. "
                      f"Best Val Loss: {best_val_loss:.4f}")
                break

    print(f"\n  ✔ Model đã lưu tại: {config.MODELS_DIR / 'emat_model.pth'}")
    print(f"  ✔ Best Val Loss (avg): {best_val_loss:.4f}")
    return model


def evaluate_model(model, loader, criteria, device) -> dict:
    """Đánh giá model trên một DataLoader. Trả về metrics cho mỗi horizon."""
    model.eval()
    all_preds = {h: [] for h in config.PREDICTION_HORIZONS}
    all_labels = {h: [] for h in config.PREDICTION_HORIZONS}
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            X_batch = batch[0]
            y_batch = {h: batch[i + 1] for i, h in enumerate(config.PREDICTION_HORIZONS)}

            outputs = model(X_batch)

            loss = sum(
                criteria[h](outputs[f"{h}d"], y_batch[h])
                for h in config.PREDICTION_HORIZONS
            ) / len(config.PREDICTION_HORIZONS)
            total_loss += loss.item()

            for h in config.PREDICTION_HORIZONS:
                preds = outputs[f"{h}d"].cpu().numpy()
                all_preds[h].extend(preds)
                all_labels[h].extend(y_batch[h].cpu().numpy())

    metrics = {}
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    for h in config.PREDICTION_HORIZONS:
        preds_arr = np.array(all_preds[h])
        labels_arr = np.array(all_labels[h])
        metrics[h] = {
            "rmse": np.sqrt(mean_squared_error(labels_arr, preds_arr)),
            "mae": mean_absolute_error(labels_arr, preds_arr),
            "loss": total_loss / max(len(loader), 1),
        }
    return metrics


def _compute_emat_rank_ic(model, split, device):
    """
    Tính Rank IC cho EMAT: dùng composite score (P_Up - P_Down)
    tương quan với forward returns thực tế.
    """
    rank_ic_results = {}

    for symbol in config.SYMBOLS:
        path = config.DATA_DIR / split / f"{symbol}_emat.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        features = df[config.EMAT_FEATURE_COLS].values.astype(np.float32)
        
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

        for h in config.PREDICTION_HORIZONS:
            future_returns = df["Close"].pct_change(periods=h).shift(-h).values
            pred_scores, actual_rets = [], []

            for i in range(len(features) - window):
                end_idx = i + window - 1
                if end_idx >= len(future_returns) or np.isnan(future_returns[end_idx]):
                    continue

                x = torch.tensor(features[i:i + window], dtype=torch.float32)
                x = x.unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    score = out[f"{h}d"].cpu().numpy()[0]

                pred_scores.append(score)
                actual_rets.append(future_returns[end_idx])

            key = f"{symbol}_{h}d"
            pred_arr = np.array(pred_scores)
            actual_arr = np.array(actual_rets)

            # Point IC
            ic = am.compute_rank_ic(pred_arr, actual_arr)
            # Rolling IC
            rolling = am.compute_rolling_rank_ic(pred_arr, actual_arr, window=60)

            rank_ic_results[key] = {**ic, **rolling}

    return rank_ic_results


def evaluate():
    """Đánh giá chi tiết trên tập val và test, bao gồm Rank IC."""
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ EMAT MODEL")
    print("=" * 60)

    device = config.DEVICE
    model = EMAT().to(device)
    model.load_state_dict(
        torch.load(config.MODELS_DIR / "emat_model.pth", map_location=device)
    )
    model.eval()

    for split in ["val", "test"]:
        print(f"\n  --- Tập {split.upper()} ---")
        X, y, _ = prepare_data(split)
        if X.shape[0] == 0:
            print(f"  Không có dữ liệu {split}.")
            continue

        tensors = [X] + [y[h] for h in config.PREDICTION_HORIZONS]
        loader = DataLoader(TensorDataset(*tensors), batch_size=config.EMAT_BATCH_SIZE)

        all_preds = {h: [] for h in config.PREDICTION_HORIZONS}
        all_labels = {h: [] for h in config.PREDICTION_HORIZONS}

        with torch.no_grad():
            for batch in loader:
                batch = [b.to(device) for b in batch]
                outputs = model(batch[0])
                for i, h in enumerate(config.PREDICTION_HORIZONS):
                    all_preds[h].extend(outputs[f"{h}d"].cpu().numpy())
                    all_labels[h].extend(batch[i + 1].cpu().numpy())

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        for h in config.PREDICTION_HORIZONS:
            p = np.array(all_preds[h])
            l = np.array(all_labels[h])
            rmse = np.sqrt(mean_squared_error(l, p))
            mae = mean_absolute_error(l, p)

            print(f"\n  Horizon {h}d:")
            print(f"    RMSE : {rmse:.4f}")
            print(f"    MAE  : {mae:.4f}")

            # Lưu metrics
            metrics_path = config.METRICS_DIR / f"emat_{split}_{h}d.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rmse": float(rmse),
                    "mae": float(mae),
                }, f, indent=2, ensure_ascii=False)

        # === RANK IC ===
        print(f"\n  Đang tính Rank IC ({split})...")
        rank_ic = _compute_emat_rank_ic(model, split, device)
        print(am.format_rank_ic_table(rank_ic, model_name=f"EMAT ({split.upper()})"))

        # Lưu Rank IC
        with open(config.METRICS_DIR / f"emat_rank_ic_{split}.json", "w", encoding="utf-8") as f:
            json.dump(rank_ic, f, indent=2, ensure_ascii=False)


def run():
    """Chạy train + evaluate."""
    model = train()
    evaluate()


if __name__ == "__main__":
    run()
