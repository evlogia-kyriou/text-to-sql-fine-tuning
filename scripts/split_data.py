from sklearn.model_selection import train_test_split
import json

# Load formatted data
examples = []
with open('data/processed/formatted_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Total examples: {len(examples)}")

# 80/20 split
db_ids = [ex['db_id'] for ex in examples]
train, val = train_test_split(
    examples, 
    test_size=0.2, 
    random_state=42,
    stratify=db_ids  # Ensures similar DB distribution
)

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")

# Save
with open('data/processed/train.jsonl', 'w', encoding='utf-8') as f:
    for ex in train:
        f.write(json.dumps(ex) + '\n')

with open('data/processed/val.jsonl', 'w', encoding='utf-8') as f:
    for ex in val:
        f.write(json.dumps(ex) + '\n')

# Show statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

# Count unique databases in each split
train_dbs = set(ex['db_id'] for ex in train)
val_dbs = set(ex['db_id'] for ex in val)

print(f"\nUnique databases:")
print(f"  Train: {len(train_dbs)} databases")
print(f"  Val: {len(val_dbs)} databases")
print(f"  Overlap: {len(train_dbs & val_dbs)} databases")

# Show example from training set
print("\n" + "="*80)
print("EXAMPLE FROM TRAINING SET")
print("="*80)
print(f"\nDatabase: {train[0]['db_id']}")
print(f"Question: {train[0]['question']}")
print(f"SQL: {train[0]['SQL']}")
print("\nFormatted text (first 600 chars):")
print("-"*80)
print(train[0]['text'][:600])
print("...")

print("\n" + "="*80)
print("✅ Ready for training!")
print("="*80)