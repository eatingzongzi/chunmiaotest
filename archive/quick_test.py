"""Quick timing test for data loading."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_mit_stanford

data_dir = 'data/mit_stanford'
print("Starting data loading...", flush=True)
t0 = time.time()
train_ds, test_ds, _, _ = load_mit_stanford(data_dir, window_length=30)
t1 = time.time()
print(f"Done in {t1-t0:.1f}s", flush=True)
print(f"Train: {len(train_ds)} samples", flush=True)
print(f"Test:  {len(test_ds)} samples", flush=True)
