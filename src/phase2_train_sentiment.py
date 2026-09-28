"""
Phase 2 - Task 2.4: Sentiment Analysis (XLM-RoBERTa + CNN + BiGRU)
===================================================================
Backbone đa ngôn ngữ XLM-RoBERTa hiểu cả EN/VI. Freeze 6 layers đầu,
fine-tune 6 layers cuối + CNN + BiGRU head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import advanced_metrics as am


class SentimentDataset(Dataset):
    """Dataset cho sentiment analysis, hỗ trợ cả text EN và VI."""
    def __init__(self, texts, labels, tokenizer, max_length=config.SENTIMENT_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class XLMRoBERTa_CNN_BiGRU(nn.Module):
    """
    XLM-RoBERTa (partial freeze) + CNN + BiGRU cho phân tích sentiment đa ngôn ngữ.
    Freeze 6/12 layers đầu → Fine-tune 6 layers cuối + CNN + BiGRU head.
    """
    def __init__(self, model_name=config.SENTIMENT_BACKBONE, num_classes=config.SENTIMENT_NUM_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768

        # Freeze 6 layers đầu (giữ embeddings + 6 encoder layers đầu)
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for i in range(config.SENTIMENT_FREEZE_LAYERS):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = False

        # CNN: quét n-gram features cục bộ
        self.conv = nn.Conv1d(hidden_size, config.SENTIMENT_CNN_OUT, kernel_size=config.SENTIMENT_CNN_KERNEL, padding=1)
        self.bn_conv = nn.BatchNorm1d(config.SENTIMENT_CNN_OUT)

        # BiGRU: học ngữ cảnh tuần tự
        self.gru = nn.GRU(
            input_size=config.SENTIMENT_CNN_OUT,
            hidden_size=config.SENTIMENT_GRU_HIDDEN,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(config.SENTIMENT_GRU_HIDDEN * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # Backbone: trích xuất contextual embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq, 768)

        # CNN
        x = sequence_output.permute(0, 2, 1)  # (batch, 768, seq)
        x = F.relu(self.bn_conv(self.conv(x)))  # (batch, 256, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, 256)

        # BiGRU
        gru_out, _ = self.gru(x)  # (batch, seq, 256)

        # Max pooling toàn câu
        pooled = torch.max(gru_out, dim=1)[0]  # (batch, 256)

        return self.fc(self.dropout(pooled))


def prepare_sentiment_data(split="train"):
    """
    Load dữ liệu sentiment từ cả nguồn VI (có nhãn) và EN (sẽ được pseudo-label).
    Trả về (texts, labels).
    """
    texts, labels = [], []

    for symbol in config.SYMBOLS:
        # 1. Dữ liệu VI có nhãn thật (sentiment_score)
        vi_path = config.DATA_DIR / split / f"{symbol}_sentiment_vi.csv"
        if vi_path.exists():
            df_vi = pd.read_csv(vi_path)
            # Dùng Content hoặc Header
            text_col = "Content" if "Content" in df_vi.columns else "Header"
            if text_col in df_vi.columns and "sentiment_score" in df_vi.columns:
                df_vi = df_vi.dropna(subset=[text_col, "sentiment_score"])
                for _, row in df_vi.iterrows():
                    text = str(row[text_col])[:1000]  # Giới hạn độ dài
                    score = row["sentiment_score"]
                    # Map score → label
                    if score > config.SENTIMENT_SCORE_THRESHOLD:
                        label = 0  # Tích cực
                    elif score < -config.SENTIMENT_SCORE_THRESHOLD:
                        label = 1  # Tiêu cực
                    else:
                        label = 2  # Trung tính
                    texts.append(text)
                    labels.append(label)

        # 2. Dữ liệu EN (title ngắn) - dùng như data bổ sung
        en_path = config.DATA_DIR / split / f"{symbol}_news_en.csv"
        if en_path.exists():
            df_en = pd.read_csv(en_path)
            if "title" in df_en.columns:
                df_en = df_en.dropna(subset=["title"])
                # Gán nhãn trung tính cho EN (sẽ được model tự học pattern)
                for _, row in df_en.iterrows():
                    texts.append(str(row["title"]))
                    labels.append(2)  # Trung tính (pseudo-label)

    return texts, labels


def train():
    print("\n" + "=" * 60)
    print("  PHASE 2.4: HUẤN LUYỆN SENTIMENT (XLM-RoBERTa + CNN + BiGRU)")
    print("=" * 60)

    device = config.DEVICE
    print(f"  Device: {device}")
    print(f"  Backbone: {config.SENTIMENT_BACKBONE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_BACKBONE)

    # Chuẩn bị data
    print("  Đang load dữ liệu train...")
    texts_train, labels_train = prepare_sentiment_data("train")
    print(f"  Train: {len(texts_train)} samples")

    texts_val, labels_val = prepare_sentiment_data("val")
    print(f"  Val  : {len(texts_val)} samples")

    if len(texts_train) == 0:
        print("  [ERROR] Không có dữ liệu sentiment để train!")
        return None

    # Phân phối nhãn
    dist = np.bincount(labels_train, minlength=3)
    print(f"  Phân phối: Tích cực={dist[0]}, Tiêu cực={dist[1]}, Trung tính={dist[2]}")

    # DataLoader
    train_ds = SentimentDataset(texts_train, labels_train, tokenizer)
    val_ds = SentimentDataset(texts_val, labels_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.SENTIMENT_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.SENTIMENT_BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = XLMRoBERTa_CNN_BiGRU().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer với learning rate khác nhau cho backbone vs head
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.SENTIMENT_LR_BACKBONE},
        {"params": head_params, "lr": config.SENTIMENT_LR_HEAD},
    ])

    # Class weights
    class_weights = torch.tensor(config.SENTIMENT_CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_f1 = 0.0
    for epoch in range(config.SENTIMENT_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)

        # Validate
        val_preds, val_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(batch["label"].numpy())

        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"  Epoch {epoch+1}/{config.SENTIMENT_EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.MODELS_DIR / "sentiment_model.pth")

    print(f"\n  ✔ Model đã lưu: {config.MODELS_DIR / 'sentiment_model.pth'}")
    print(f"  ✔ Best Val F1: {best_val_f1:.4f}")
    return model


def evaluate():
    print("\n  ĐÁNH GIÁ SENTIMENT MODEL")
    device = config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_BACKBONE)
    model = XLMRoBERTa_CNN_BiGRU().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "sentiment_model.pth", map_location=device))
    model.eval()

    for split in ["val", "test"]:
        texts, labels = prepare_sentiment_data(split)
        if not texts:
            continue
        ds = SentimentDataset(texts, labels, tokenizer)
        loader = DataLoader(ds, batch_size=config.SENTIMENT_BATCH_SIZE, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(batch["label"].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        print(f"\n  {split.upper()}: Acc={acc:.4f} | F1={f1:.4f}")
        print(f"  CM (Pos/Neg/Neu):\n{cm}")

        # Đặc biệt report Recall lớp Tiêu cực (index=1)
        report = classification_report(all_labels, all_preds, target_names=["Tích cực", "Tiêu cực", "Trung tính"], zero_division=0, output_dict=True)
        print(f"  Recall Tiêu cực: {report['Tiêu cực']['recall']:.4f}")

        with open(config.METRICS_DIR / f"sentiment_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"acc": acc, "f1": f1, "cm": cm.tolist(), "report": report}, f, indent=2, ensure_ascii=False)

    # === RANK IC cho Sentiment ===
    _compute_sentiment_rank_ic(model, tokenizer, device)


def _compute_sentiment_rank_ic(model, tokenizer, device):
    """
    Tính Rank IC cho Sentiment: dùng P(Tích cực) - P(Tiêu cực) làm score,
    tương quan với forward returns 1d của cổ phiếu được đề cập.
    """
    print("\n  Đang tính Rank IC cho Sentiment...")
    model.eval()

    for split in ["val", "test"]:
        rank_ic_results = {}

        for symbol in config.SYMBOLS:
            # Load sentiment data
            vi_path = config.DATA_DIR / split / f"{symbol}_sentiment_vi.csv"
            prices_path = config.DATA_DIR / split / f"{symbol}_prices.csv"
            if not vi_path.exists() or not prices_path.exists():
                continue

            df_vi = pd.read_csv(vi_path)
            df_prices = pd.read_csv(prices_path)
            df_prices["Date"] = pd.to_datetime(df_prices["Date"])

            # Tính forward returns 1d
            df_prices["fwd_return"] = df_prices["Close"].pct_change().shift(-1)
            date_to_return = dict(zip(df_prices["Date"].dt.date, df_prices["fwd_return"]))

            text_col = "Content" if "Content" in df_vi.columns else "Header"
            if text_col not in df_vi.columns:
                continue

            df_vi["Date"] = pd.to_datetime(df_vi["Date" if "Date" in df_vi.columns else "date"])
            df_vi = df_vi.dropna(subset=[text_col])

            pred_scores, actual_rets = [], []

            for _, row in df_vi.iterrows():
                text = str(row[text_col])[:1000]
                news_date = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
                fwd_ret = date_to_return.get(news_date)
                if fwd_ret is None or np.isnan(fwd_ret):
                    continue

                encoded = tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=config.SENTIMENT_MAX_LENGTH, return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(
                        encoded["input_ids"].to(device),
                        encoded["attention_mask"].to(device),
                    )
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                # Score: P(Tích cực) - P(Tiêu cực)
                score = probs[0] - probs[1]
                pred_scores.append(score)
                actual_rets.append(fwd_ret)

            if len(pred_scores) > 10:
                pred_arr = np.array(pred_scores)
                actual_arr = np.array(actual_rets)
                ic = am.compute_rank_ic(pred_arr, actual_arr)
                rolling = am.compute_rolling_rank_ic(pred_arr, actual_arr, window=30)
                rank_ic_results[symbol] = {**ic, **rolling}

        if rank_ic_results:
            print(am.format_rank_ic_table(rank_ic_results, model_name=f"Sentiment ({split.upper()})"))
            with open(config.METRICS_DIR / f"sentiment_rank_ic_{split}.json", "w", encoding="utf-8") as f:
                json.dump(rank_ic_results, f, indent=2, ensure_ascii=False)
        else:
            print(f"  [INFO] Không đủ dữ liệu để tính Rank IC ({split})")


def run():
    train()
    evaluate()

if __name__ == "__main__":
    run()
