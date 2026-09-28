"""Script test de debug import config."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
print("Step 1: sys imported")
try:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    print("Step 2: path configured")
    import config
    print("Step 3: config imported")
    config.print_config()
    print("Step 4: config printed")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
