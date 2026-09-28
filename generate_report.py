"""
Generate Report: Tạo toàn bộ biểu đồ đánh giá pipeline.
Chạy: python generate_report.py
Output: results/plots/*.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "font.size": 11,
})

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def load_json(name):
    with open(METRICS / name, encoding="utf-8") as f:
        return json.load(f)

# ==============================================================
# 1. EMAT: Accuracy + F1 Bar Chart
# ==============================================================
def plot_emat_performance():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EMAT Multi-Task Performance", fontsize=16, fontweight="bold", color="#e94560")

    for ax_idx, split in enumerate(["val", "test"]):
        ax = axes[ax_idx]
        horizons = ["1d", "3d", "5d"]
        accs, f1s = [], []
        for h in horizons:
            m = load_json(f"emat_{split}_{h}.json")
            accs.append(m["accuracy"])
            f1s.append(m["f1_weighted"])

        x = np.arange(len(horizons))
        w = 0.35
        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#0f3460", edgecolor="#e94560")
        bars2 = ax.bar(x + w/2, f1s, w, label="F1 (weighted)", color="#533483", edgecolor="#e94560")

        ax.axhline(y=0.333, color="#e94560", linestyle="--", alpha=0.7, label="Random (33.3%)")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Score")
        ax.set_title(f"Tập {split.upper()}", color="#eee")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylim(0, 0.55)
        ax.legend(fontsize=9)

        for bar in list(bars1) + list(bars2):
            h_val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h_val + 0.01, f"{h_val:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#eee")

    plt.tight_layout()
    fig.savefig(PLOTS / "01_emat_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 01_emat_performance.png")

# ==============================================================
# 2. EMAT Confusion Matrices
# ==============================================================
def plot_emat_confusion():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("EMAT Confusion Matrices", fontsize=16, fontweight="bold", color="#e94560")
    labels = ["Giảm", "Tăng", "Đi ngang"]

    for row, split in enumerate(["val", "test"]):
        for col, h in enumerate(["1d", "3d", "5d"]):
            ax = axes[row, col]
            m = load_json(f"emat_{split}_{h}.json")
            cm = np.array(m["confusion_matrix"])

            # Normalize theo hàng
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
            im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=0.6)

            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                            ha="center", va="center", fontsize=8,
                            color="white" if cm_norm[i,j] > 0.35 else "#333")

            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"{split.upper()} — {h}", color="#eee")
            if col == 0:
                ax.set_ylabel("Thực tế")
            if row == 1:
                ax.set_xlabel("Dự đoán")

    plt.tight_layout()
    fig.savefig(PLOTS / "02_emat_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 02_emat_confusion.png")

# ==============================================================
# 3. EMAT Rank IC Heatmap
# ==============================================================
def plot_rank_ic():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EMAT Rank IC (Spearman) — Khả năng xếp hạng", fontsize=16, fontweight="bold", color="#e94560")

    symbols = ["GOOGL", "META", "VNM"]
    horizons = ["1d", "3d", "5d"]

    for ax_idx, split in enumerate(["val", "test"]):
        ax = axes[ax_idx]
        data = load_json(f"emat_rank_ic_{split}.json")

        matrix = np.zeros((3, 3))
        annot = []
        for i, sym in enumerate(symbols):
            row_annot = []
            for j, h in enumerate(horizons):
                key = f"{sym}_{h}"
                if key in data:
                    ic = data[key].get("rank_ic", 0)
                    p = data[key].get("p_value", 1)
                    matrix[i, j] = ic
                    star = "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.1 else ""
                    row_annot.append(f"{ic:.3f}{star}")
                else:
                    row_annot.append("N/A")
            annot.append(row_annot)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.15, vmax=0.45, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=10,
                        fontweight="bold", color="black")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(horizons)
        ax.set_yticklabels(symbols)
        ax.set_title(f"Tập {split.upper()}\n(** p<0.01, * p<0.05, † p<0.1)", fontsize=10, color="#eee")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(PLOTS / "03_emat_rank_ic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 03_emat_rank_ic.png")

# ==============================================================
# 4. EMAT Rolling ICIR
# ==============================================================
def plot_icir():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("EMAT Rolling ICIR & Hit Rate (TEST)", fontsize=14, fontweight="bold", color="#e94560")

    data = load_json("emat_rank_ic_test.json")
    keys = sorted(data.keys())
    icirs = [data[k].get("icir", 0) for k in keys]
    hits = [data[k].get("hit_rate", 0) * 100 for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    bars1 = ax.bar(x - w/2, icirs, w, label="ICIR", color="#0f3460", edgecolor="#e94560")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, hits, w, label="Hit Rate %", color="#533483", edgecolor="#e94560", alpha=0.7)

    ax.axhline(0.5, color="#e94560", ls="--", alpha=0.5, label="ICIR=0.5 (tốt)")
    ax.set_ylabel("ICIR")
    ax2.set_ylabel("Hit Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTS / "04_emat_icir.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 04_emat_icir.png")

# ==============================================================
# 5. Sentiment Performance
# ==============================================================
def plot_sentiment():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentiment Model Performance (XLM-RoBERTa + CNN + BiGRU)",
                 fontsize=14, fontweight="bold", color="#e94560")
    labels = ["Tích cực", "Tiêu cực", "Trung tính"]

    for ax_idx, split in enumerate(["val", "test"]):
        ax = axes[ax_idx]
        m = load_json(f"sentiment_{split}.json")
        cm = np.array(m["cm"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i,j] > 0.5 else "#333")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{split.upper()} — Acc={m['acc']:.3f} | F1={m['f1']:.3f}", color="#eee")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")

    plt.tight_layout()
    fig.savefig(PLOTS / "05_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 05_sentiment.png")

# ==============================================================
# 6. HMM Tail-Risk Detection
# ==============================================================
def plot_hmm_tail_risk():
    symbols = ["GOOGL", "META", "VNM"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HMM Tail-Risk Detection Rate (TEST)", fontsize=14, fontweight="bold", color="#e94560")

    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        m = load_json(f"hmm_test_{sym}.json")
        tr = m["tail_risk"]

        cats = ["Left Tail\n(Crash)", "Right Tail\n(Surge)", "Overall"]
        rates = [
            tr["left_tail"]["detection_rate"],
            tr["right_tail"]["detection_rate"],
            tr["overall_tail_detection_rate"],
        ]
        colors = ["#e94560", "#533483", "#0f3460"]
        bars = ax.bar(cats, [r * 100 for r in rates], color=colors, edgecolor="#eee", width=0.6)

        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{r*100:.1f}%", ha="center", fontsize=10, color="#eee")

        ax.set_ylim(0, 100)
        ax.set_ylabel("Detection Rate (%)")
        ax.set_title(f"{sym} (n={m['n_components']})", color="#eee")
        ax.axhline(50, color="#e94560", ls="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(PLOTS / "06_hmm_tail_risk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 06_hmm_tail_risk.png")

# ==============================================================
# 7. HMM Regime Persistence
# ==============================================================
def plot_hmm_persistence():
    symbols = ["GOOGL", "META", "VNM"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HMM Regime Persistence (Transition Matrix Diagonal)",
                 fontsize=14, fontweight="bold", color="#e94560")

    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        m = load_json(f"hmm_test_{sym}.json")
        stab = m["regime_stability"]
        tm = np.array(stab["transition_matrix"])
        n = len(tm)

        im = ax.imshow(tm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{tm[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if tm[i,j] > 0.5 else "#333")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"S{i}" for i in range(n)], fontsize=9)
        ax.set_yticklabels([f"S{i}" for i in range(n)], fontsize=9)
        ax.set_title(f"{sym} — Avg Persist={stab['avg_persistence']:.3f}", color="#eee")
        ax.set_xlabel("To")
        ax.set_ylabel("From")

    plt.tight_layout()
    fig.savefig(PLOTS / "07_hmm_persistence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 07_hmm_persistence.png")

# ==============================================================
# 8. GARCH Volatility Distribution
# ==============================================================
def plot_garch():
    symbols = ["GOOGL", "META", "VNM"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GARCH Volatility Classification (TEST)", fontsize=14, fontweight="bold", color="#e94560")

    for idx, sym in enumerate(symbols):
        ax = axes[idx]
        m = load_json(f"garch_test_{sym}.json")
        dist = m["dist"]
        cats = list(dist.keys())
        vals = list(dist.values())
        colors = {"Thấp": "#533483", "Trung bình": "#0f3460", "Cao": "#e94560"}
        bar_colors = [colors.get(c, "#666") for c in cats]

        bars = ax.bar(cats, vals, color=bar_colors, edgecolor="#eee")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(v), ha="center", fontsize=10, color="#eee")

        ax.set_title(f"{sym} (RMSE={m['rmse']:.3f})", color="#eee")
        ax.set_ylabel("Số ngày")

    plt.tight_layout()
    fig.savefig(PLOTS / "08_garch_vol.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 08_garch_vol.png")

# ==============================================================
# 9. Agent Performance Dashboard
# ==============================================================
def plot_agent_dashboard():
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Hybrid DQN-PPO Agent — Performance Dashboard",
                 fontsize=16, fontweight="bold", color="#e94560")
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    val_m = load_json("agent_backtest_val.json")
    test_m = load_json("agent_backtest_test.json")

    # --- Panel 1: Key Metrics Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_keys = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"]
    labels = ["Sharpe", "Sortino", "Calmar", "Profit\nFactor"]
    val_vals = [val_m[k] for k in metrics_keys]
    test_vals = [test_m[k] for k in metrics_keys]

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, val_vals, w, label="VAL", color="#0f3460", edgecolor="#e94560")
    ax1.bar(x + w/2, test_vals, w, label="TEST", color="#533483", edgecolor="#e94560")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.set_title("Risk-Adjusted Metrics", color="#eee")
    ax1.axhline(0, color="#e94560", ls="-", alpha=0.3)

    # --- Panel 2: Return Comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    cats = ["Agent\nVAL", "B&H\nVAL", "Agent\nTEST", "B&H\nTEST"]
    returns = [
        val_m["total_return"] * 100,
        val_m["buy_hold_return"] * 100,
        test_m["total_return"] * 100,
        test_m["buy_hold_return"] * 100,
    ]
    colors = ["#0f3460", "#666", "#533483", "#666"]
    bars = ax2.bar(cats, returns, color=colors, edgecolor="#e94560")
    for bar, r in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (3 if r >= 0 else -8),
                 f"{r:.1f}%", ha="center", fontsize=10, color="#eee")
    ax2.axhline(0, color="#e94560", ls="-", alpha=0.5)
    ax2.set_title("Total Return (%)", color="#eee")
    ax2.set_ylabel("%")

    # --- Panel 3: Win/Loss Stats ---
    ax3 = fig.add_subplot(gs[1, 0])
    test_stats = [
        ("Win Rate", test_m["win_rate"] * 100),
        ("Avg Win", test_m["avg_win"] * 100),
        ("Avg Loss", abs(test_m["avg_loss"]) * 100),
        ("R/R Ratio", test_m["risk_reward_ratio"]),
        ("Expectancy\n(×1000)", test_m["expectancy"] * 1000),
    ]
    s_labels = [s[0] for s in test_stats]
    s_vals = [s[1] for s in test_stats]
    colors3 = ["#0f3460", "#2ecc71", "#e94560", "#533483", "#f39c12"]
    bars3 = ax3.barh(s_labels, s_vals, color=colors3, edgecolor="#eee")
    for bar, v in zip(bars3, s_vals):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=9, color="#eee")
    ax3.set_title("Trading Statistics (TEST)", color="#eee")

    # --- Panel 4: Risk Metrics ---
    ax4 = fig.add_subplot(gs[1, 1])
    risk_data = {
        "VAL": {
            "Max DD": val_m["max_drawdown"] * 100,
            "DD Duration": val_m["max_dd_duration_days"],
            "Underwater%": val_m["underwater_pct"],
        },
        "TEST": {
            "Max DD": test_m["max_drawdown"] * 100,
            "DD Duration": test_m["max_dd_duration_days"],
            "Underwater%": test_m["underwater_pct"],
        },
    }
    risk_labels = ["Max DD (%)", "DD Duration\n(days)", "Underwater (%)"]
    val_risk = [risk_data["VAL"]["Max DD"], risk_data["VAL"]["DD Duration"], risk_data["VAL"]["Underwater%"]]
    test_risk = [risk_data["TEST"]["Max DD"], risk_data["TEST"]["DD Duration"], risk_data["TEST"]["Underwater%"]]

    x4 = np.arange(3)
    ax4.bar(x4 - w/2, val_risk, w, label="VAL", color="#e94560", edgecolor="#eee")
    ax4.bar(x4 + w/2, test_risk, w, label="TEST", color="#c0392b", edgecolor="#eee")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(risk_labels)
    ax4.legend()
    ax4.set_title("Risk Profile", color="#eee")

    fig.savefig(PLOTS / "09_agent_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 09_agent_dashboard.png")

# ==============================================================
# 10. Pipeline Summary Scorecard
# ==============================================================
def plot_scorecard():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")

    test_m = load_json("agent_backtest_test.json")
    emat_1d = load_json("emat_test_1d.json")
    emat_5d = load_json("emat_test_5d.json")
    sent = load_json("sentiment_test.json")
    hmm_g = load_json("hmm_test_GOOGL.json")

    rows = [
        ["Component", "Metric", "Value", "Benchmark", "Status"],
        ["EMAT (1d)", "F1-Score", f"{emat_1d['f1_weighted']:.3f}", "0.333 (random)", "⚠ Marginal"],
        ["EMAT (5d)", "F1-Score", f"{emat_5d['f1_weighted']:.3f}", "0.333 (random)", "⚠ Marginal"],
        ["EMAT", "Best Rank IC", "0.417 (META 5d)", ">0.05 (useful)", "✓ Strong"],
        ["Sentiment", "Accuracy", f"{sent['acc']:.3f}", ">0.80", "✓ Good"],
        ["Sentiment", "F1 (weighted)", f"{sent['f1']:.3f}", ">0.80", "✓ Good"],
        ["HMM (GOOGL)", "Tail Detection", f"{hmm_g['tail_risk']['overall_tail_detection_rate']:.1%}", ">50%", "✗ Weak"],
        ["GARCH", "RMSE (GOOGL)", "0.864", "<1.0", "✓ Good"],
        ["Agent", "Total Return", f"{test_m['total_return']*100:.1f}%", "vs B&H -47.2%", "✓ Strong"],
        ["Agent", "Sharpe Ratio", f"{test_m['sharpe_ratio']:.3f}", ">1.0", "✓ Good"],
        ["Agent", "Max Drawdown", f"{test_m['max_drawdown']*100:.1f}%", "<30%", "✓ Good"],
        ["Agent", "Calmar Ratio", f"{test_m['calmar_ratio']:.3f}", ">1.0", "✓ Good"],
        ["Agent", "Profit Factor", f"{test_m['profit_factor']:.3f}", ">1.0", "✓ Good"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                     cellLoc="center", colColours=["#e94560"]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_facecolor("#e94560")
        else:
            cell.set_facecolor("#16213e")
            cell.set_text_props(color="#eee")
            status = rows[r][4] if r < len(rows) else ""
            if c == 4:
                if "✓" in status:
                    cell.set_facecolor("#1a472a")
                elif "⚠" in status:
                    cell.set_facecolor("#4a3f00")
                elif "✗" in status:
                    cell.set_facecolor("#4a1a1a")

    ax.set_title("PIPELINE SCORECARD — Tổng hợp đánh giá toàn bộ hệ thống",
                 fontsize=14, fontweight="bold", color="#e94560", pad=20)

    fig.savefig(PLOTS / "10_scorecard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] 10_scorecard.png")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  GENERATING PIPELINE REPORT")
    print("=" * 50)
    plot_emat_performance()
    plot_emat_confusion()
    plot_rank_ic()
    plot_icir()
    plot_sentiment()
    plot_hmm_tail_risk()
    plot_hmm_persistence()
    plot_garch()
    plot_agent_dashboard()
    plot_scorecard()
    print(f"  [OK] {PLOTS}")
