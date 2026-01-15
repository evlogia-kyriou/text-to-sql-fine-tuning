import json
import os

# Set paths to YOUR extracted folders
TRAIN_JSON = r"D:\Doku\doku-text2sql\data\raw\train\train.json"
DEV_JSON = r"D:\Doku\doku-text2sql\data\raw\dev\dev.json"

# Check if files exist
print("Checking files...")
print(f"train.json exists: {os.path.exists(TRAIN_JSON)}")
print(f"dev.json exists: {os.path.exists(DEV_JSON)}")

# Load train data
print("\nLoading train.json...")
with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"✅ SUCCESS! Loaded {len(train_data)} training examples")

# Load dev data
print("\nLoading dev.json...")
with open(DEV_JSON, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

print(f"✅ SUCCESS! Loaded {len(dev_data)} dev examples")

# Look at structure
print("\n" + "="*80)
print("EXAMPLE STRUCTURE")
print("="*80)
print(json.dumps(train_data[0], indent=2)[:600])
print("...")