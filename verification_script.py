# Quick verification script
import json

# Load one example
with open('data/processed/formatted_train.jsonl', 'r') as f:
    first_example = json.loads(f.readline())

# Print it
print("="*80)
print("FORMATTED EXAMPLE:")
print("="*80)
print(first_example['text'])
print("\n" + "="*80)
print(f"Database: {first_example['db_id']}")
print(f"Question: {first_example['question']}")
print(f"SQL: {first_example['SQL']}")