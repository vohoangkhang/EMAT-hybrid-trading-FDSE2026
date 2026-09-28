"""
Run Pipeline: Orchestrator chạy tuần tự 4 Phase.
=================================================
Mỗi Phase là một mô-đun độc lập, chạy theo thứ tự:
Phase 1 → Phase 2 (EMAT, HMM, GARCH, Sentiment) → Phase 3 → Phase 4
"""
import time
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def run_phase(name, module_func):
    """Chạy một phase với error handling và timing."""
    print(f"\n{'#'*60}")
    print(f"# {name}")
    print(f"{'#'*60}")
    start = time.time()
    try:
        module_func()
        elapsed = time.time() - start
        print(f"\n  ⏱ {name} hoàn tất trong {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✘ {name} THẤT BẠI sau {elapsed:.1f}s")
        print(f"  Lỗi: {e}")
        traceback.print_exc()
        return False


def main():
    """Chạy toàn bộ pipeline."""
    config.print_config()
    overall_start = time.time()

    # Phase 1: Data Split
    from phase1_data_split import run as phase1_run
    if not run_phase("PHASE 1: CHUẨN BỊ DỮ LIỆU ĐỒNG BỘ", phase1_run):
        print("DỪNG PIPELINE DO LỖI PHASE 1")
        return

    # Phase 2: Feature Extractors
    from phase2_train_emat import run as emat_run
    if not run_phase("PHASE 2.1: HUẤN LUYỆN EMAT", emat_run):
        print("DỪNG PIPELINE DO LỖI EMAT")
        return

    from phase2_train_hmm import run as hmm_run
    if not run_phase("PHASE 2.2: HUẤN LUYỆN HMM", hmm_run):
        print("DỪNG PIPELINE DO LỖI HMM")
        return

    from phase2_train_garch import run as garch_run
    if not run_phase("PHASE 2.3: HUẤN LUYỆN GARCH", garch_run):
        print("DỪNG PIPELINE DO LỖI GARCH")
        return

    # Sentiment có thể bỏ qua nếu không có GPU/data đủ
    from phase2_train_sentiment import run as sentiment_run
    sentiment_ok = run_phase("PHASE 2.4: HUẤN LUYỆN SENTIMENT", sentiment_run)
    if not sentiment_ok:
        print("  [WARN] Sentiment training thất bại, tiếp tục không có sentiment...")

    # Phase 3: Generate States + Environment
    from phase3_generate_states import run as states_run
    if not run_phase("PHASE 3: SINH STATE VECTORS", states_run):
        print("DỪNG PIPELINE DO LỖI PHASE 3")
        return

    # Phase 4: Train Agent
    from phase4_train_agent import run as agent_run
    if not run_phase("PHASE 4: HUẤN LUYỆN HYBRID AGENT", agent_run):
        print("DỪNG PIPELINE DO LỖI PHASE 4")
        return

    # Tổng kết
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  ✔ PIPELINE HOÀN TẤT THÀNH CÔNG")
    print(f"  ⏱ Tổng thời gian: {total_time/60:.1f} phút")
    print(f"  📁 Models: {config.MODELS_DIR}")
    print(f"  📁 States: {config.STATES_DIR}")
    print(f"  📁 Results: {config.METRICS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
