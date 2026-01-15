# Quick verification script
import json
import os

# UPDATE THESE PATHS TO MATCH YOUR SETUP
BASE_PATH = r"D:\Doku\doku-text2sql\data\raw"  # Update this!
TRAIN_JSON = os.path.join(BASE_PATH, "train", "train.json")
DEV_JSON = os.path.join(BASE_PATH, "dev", "dev.json")

print("="*80)
print("BIRD DATA CHECK")
print("="*80)

# Check files
train_exists = os.path.exists(TRAIN_JSON)
dev_exists = os.path.exists(DEV_JSON)

print(f"\nChecking: {TRAIN_JSON}")
print(f"  Exists: {train_exists}")
if train_exists:
    size_mb = os.path.getsize(TRAIN_JSON) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

print(f"\nChecking: {DEV_JSON}")
print(f"  Exists: {dev_exists}")
if dev_exists:
    size_mb = os.path.getsize(DEV_JSON) / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

if train_exists and dev_exists:
    print("\n✅ READY TO START ANALYSIS!")
    
    # Quick load
    with open(TRAIN_JSON, 'r') as f:
        train = json.load(f)
    with open(DEV_JSON, 'r') as f:
        dev = json.load(f)
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train)} examples")
    print(f"  Dev: {len(dev)} examples")
    print(f"\nExample fields: {list(train[0].keys())}")
else:
    print("\n❌ Files not found - please update BASE_PATH")