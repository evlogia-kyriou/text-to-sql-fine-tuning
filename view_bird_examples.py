"""
View BIRD Dataset Examples
===========================
Load and display several examples from the BIRD dataset
"""

import json
import pandas as pd

# Load the processed data (already analyzed)
print("="*80)
print("BIRD DATASET SAMPLE EXAMPLES")
print("="*80)
print()

# Try to load from processed data first
try:
    train_df = pd.read_json('data/processed/train_analyzed.jsonl', lines=True)
    print(f"✓ Loaded from processed data: {len(train_df)} examples")
except:
    # Fallback to raw data if processed doesn't exist
    print("Loading from raw data...")
    with open('data/raw/train/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data)
    print(f"✓ Loaded from raw data: {len(train_df)} examples")

print()
print("="*80)
print("DATASET STRUCTURE")
print("="*80)
print(f"\nColumns: {list(train_df.columns)}")
print(f"\nDataset shape: {train_df.shape}")
print()

# Display 5 diverse examples
print("="*80)
print("EXAMPLE 1: Simple Query")
print("="*80)
example1 = train_df.iloc[0]
print(json.dumps({
    'db_id': example1['db_id'],
    'question': example1['question'],
    'evidence': example1.get('evidence', 'N/A'),
    'SQL': example1['SQL']
}, indent=2))
print()

print("="*80)
print("EXAMPLE 2: Different Database")
print("="*80)
example2 = train_df.iloc[100]
print(json.dumps({
    'db_id': example2['db_id'],
    'question': example2['question'],
    'evidence': example2.get('evidence', 'N/A'),
    'SQL': example2['SQL']
}, indent=2))
print()

print("="*80)
print("EXAMPLE 3: Mid-complexity Query")
print("="*80)
example3 = train_df.iloc[500]
print(json.dumps({
    'db_id': example3['db_id'],
    'question': example3['question'],
    'evidence': example3.get('evidence', 'N/A'),
    'SQL': example3['SQL']
}, indent=2))
print()

print("="*80)
print("EXAMPLE 4: Another Database")
print("="*80)
example4 = train_df.iloc[1000]
print(json.dumps({
    'db_id': example4['db_id'],
    'question': example4['question'],
    'evidence': example4.get('evidence', 'N/A'),
    'SQL': example4['SQL']
}, indent=2))
print()

print("="*80)
print("EXAMPLE 5: Later in Dataset")
print("="*80)
example5 = train_df.iloc[2000]
print(json.dumps({
    'db_id': example5['db_id'],
    'question': example5['question'],
    'evidence': example5.get('evidence', 'N/A'),
    'SQL': example5['SQL']
}, indent=2))
print()

# Show SQL complexity distribution
if 'num_joins' in train_df.columns:
    print("="*80)
    print("SQL COMPLEXITY OVERVIEW")
    print("="*80)
    print(f"\nJOIN distribution:")
    print(train_df['num_joins'].value_counts().sort_index().head(10))
    print()
    
    # Show one complex example
    complex_examples = train_df[train_df['num_joins'] >= 3]
    if len(complex_examples) > 0:
        print("="*80)
        print("EXAMPLE 6: Complex Query (3+ JOINs)")
        print("="*80)
        example6 = complex_examples.iloc[0]
        print(json.dumps({
            'db_id': example6['db_id'],
            'question': example6['question'],
            'evidence': example6.get('evidence', 'N/A'),
            'SQL': example6['SQL'],
            'num_joins': int(example6['num_joins'])
        }, indent=2))

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"\n✓ Dataset has {len(train_df)} training examples")
print(f"✓ {train_df['db_id'].nunique()} unique databases")
print(f"✓ Columns: {', '.join(train_df.columns[:4])}")  # Main columns
print()