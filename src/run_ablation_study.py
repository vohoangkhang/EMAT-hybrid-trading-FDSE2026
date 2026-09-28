import os
import shutil
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import phase4_train_agent

SCENARIOS = {
    "base": {"emat": False, "hmm": False, "garch": False, "sentiment": False},
    "base_emat": {"emat": True, "hmm": False, "garch": False, "sentiment": False},
    "base_emat_hmm_garch": {"emat": True, "hmm": True, "garch": True, "sentiment": False},
    "base_emat_sentiment": {"emat": True, "hmm": False, "garch": False, "sentiment": True},
    "full": {"emat": True, "hmm": True, "garch": True, "sentiment": True},
}

def backup_original_states():
    print("\n[Ablation] Sao lưu dữ liệu gốc...")
    backup_dir = config.STATES_DIR / "backup_full"
    backup_dir.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        file_name = f"{split}_observations.npy"
        src_path = config.STATES_DIR / file_name
        dest_path = backup_dir / file_name
        
        if not src_path.exists():
            print(f"[ERROR] Không tìm thấy {src_path}! Vui lòng chạy Phase 3 trước.")
            sys.exit(1)
            
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
            print(f"  Đã backup: {file_name}")
        else:
            print(f"  Backup đã tồn tại: {file_name}")

def apply_mask_to_states(scenario_name, mask_config):
    print(f"\n[Ablation] Áp dụng mặt nạ (Masking) cho kịch bản: {scenario_name.upper()}")
    backup_dir = config.STATES_DIR / "backup_full"
    
    for split in ["train", "val", "test"]:
        file_name = f"{split}_observations.npy"
        obs = np.load(backup_dir / file_name)
        
        # Format of state_dim=28:
        # 0-4: OHLCV (5 dims)
        # 5-11: Tech Indicators (7 dims)
        # 12-14: EMAT (3 dims)
        # 15-20: HMM (6 dims)
        # 21-24: GARCH (4 dims)
        # 25-27: Sentiment (3 dims)
        
        if not mask_config["emat"]:
            obs[:, 12:15] = 0.0
            
        if not mask_config["hmm"]:
            obs[:, 15:21] = 0.0
            
        if not mask_config["garch"]:
            obs[:, 21:25] = 0.0
            
        if not mask_config["sentiment"]:
            # Điền xác suất trung lập thay vì 0 để khỏi làm nhiễu neural net
            obs[:, 25:28] = 1/3
            
        # Lưu đè vào thư mục gốc để Phase 4 tự động lấy
        dest_path = config.STATES_DIR / file_name
        np.save(dest_path, obs)
        print(f"  Đã ghi đè {file_name} với cấu hình {scenario_name}")

def clean_old_training_data():
    """Xóa dữ liệu per_symbol cũ để buộc agent phải train lại từ đầu."""
    per_symbol_dir = config.MODELS_DIR / "per_symbol"
    if per_symbol_dir.exists():
        print("  Xóa thư mục models/per_symbol cũ để train lại...")
        shutil.rmtree(per_symbol_dir)

def run_ablation():
    print("==================================================")
    print(" BẮT ĐẦU ABLATION STUDY - KIỂM CHỨNG CÁC MODULE")
    print("==================================================")
    
    # 1. Backup dữ liệu gốc
    backup_original_states()
    
    ablation_results_dir = config.RESULTS_DIR / "ablation"
    ablation_results_dir.mkdir(exist_ok=True)
    
    # Ép chạy chế độ FULL THEO YÊU CẦU CỦA USER
    config.QUICK_MODE = False
    
    # Lưu lại file cấu hình để khôi phục sau này nếu cần
    for scenario_name, mask_config in SCENARIOS.items():
        print(f"\n\n{'='*60}")
        print(f" ĐANG CHẠY KỊCH BẢN: {scenario_name.upper()}")
        print(f" Cấu hình: {mask_config}")
        print(f"{'='*60}")
        
        scenario_out_dir = ablation_results_dir / scenario_name
        if scenario_out_dir.exists():
            print(f"Kịch bản {scenario_name} đã tồn tại trong results/ablation. Bỏ qua.")
            continue
            
        # 2. Áp dụng Mask
        apply_mask_to_states(scenario_name, mask_config)
        
        # 3. Xóa model cũ để ép train lại
        clean_old_training_data()
        
        # 4. Chạy Phase 4: Train + Evaluate
        try:
            phase4_train_agent.run_per_symbol()
        except Exception as e:
            print(f"\n[ERROR] Lỗi khi train kịch bản {scenario_name}: {e}")
            break
            
        # 5. Lưu kết quả ra thư mục riêng
        print(f"\n[Ablation] Lưu kết quả của kịch bản {scenario_name}...")
        scenario_out_dir.mkdir(exist_ok=True)
        
        # Copy file tổng hợp kết quả
        summary_file = config.METRICS_DIR / "agent_per_symbol_summary.json"
        if summary_file.exists():
            shutil.copy2(summary_file, scenario_out_dir / "agent_per_symbol_summary.json")
            
        # Move thư mục models per_symbol sang kết quả ablation (nếu muốn giữ lại model của từng kịch bản)
        # Tuy nhiên model rất nặng, nên ta chỉ copy các file JSON backtest thay vì copy model weights
        per_symbol_dir = config.MODELS_DIR / "per_symbol"
        if per_symbol_dir.exists():
            for sym_dir in per_symbol_dir.iterdir():
                if sym_dir.is_dir():
                    dest_sym_dir = scenario_out_dir / sym_dir.name
                    dest_sym_dir.mkdir(exist_ok=True)
                    # Chỉ copy các file backtest.json
                    for json_file in sym_dir.glob("*.json"):
                        shutil.copy2(json_file, dest_sym_dir / json_file.name)
                        
        print(f"Đã lưu kết quả của {scenario_name} vào {scenario_out_dir}")

    # 6. Phục hồi dữ liệu gốc
    print("\n[Ablation] Đang phục hồi dữ liệu observations gốc...")
    apply_mask_to_states("full", SCENARIOS["full"])
    print("🎉 HOÀN TẤT QUÁ TRÌNH ABLATION STUDY!")

if __name__ == "__main__":
    run_ablation()
