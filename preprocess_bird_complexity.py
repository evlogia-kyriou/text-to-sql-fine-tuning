"""
BIRD Dataset Preprocessing - Filter by SQL Complexity
======================================================

Since BIRD dataset has no difficulty labels, we filter by SQL complexity:
- Simple: 0-1 JOINs
- Moderate: 0-2 JOINs  
- All: Everything

This creates subsets for different experiments.
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("="*80)
print("BIRD DATASET PREPROCESSING - COMPLEXITY-BASED FILTERING")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD ANALYZED DATA
# ============================================================================

print("STEP 1: Loading analyzed data...")
train_df = pd.read_json('data/processed/train_analyzed.jsonl', lines=True)
dev_df = pd.read_json('data/processed/dev_analyzed.jsonl', lines=True)

print(f"✓ Train: {len(train_df)} examples")
print(f"✓ Dev: {len(dev_df)} examples")
print()

# ============================================================================
# STEP 2: EXTRACT SQL FEATURES (from analysis)
# ============================================================================

print("STEP 2: Extracting SQL complexity features...")

def analyze_sql_complexity(sql):
    """Extract SQL features"""
    sql_upper = sql.upper()
    
    return {
        'length': len(sql),
        'num_joins': sql_upper.count('JOIN'),
        'has_join': 'JOIN' in sql_upper,
        'has_subquery': '(' in sql and sql_upper.count('SELECT') > 1,
        'has_group_by': 'GROUP BY' in sql_upper,
        'has_order_by': 'ORDER BY' in sql_upper,
    }

# Calculate features
train_df['sql_features'] = train_df['SQL'].apply(analyze_sql_complexity)
dev_df['sql_features'] = dev_df['SQL'].apply(analyze_sql_complexity)

# Extract into columns
feature_df = pd.DataFrame(train_df['sql_features'].tolist())
train_df = pd.concat([train_df, feature_df], axis=1)

dev_feature_df = pd.DataFrame(dev_df['sql_features'].tolist())
dev_df = pd.concat([dev_df, dev_feature_df], axis=1)

print(f"✓ Calculated complexity features")
print()

# ============================================================================
# STEP 3: CREATE COMPLEXITY-BASED SUBSETS
# ============================================================================

print("STEP 3: Creating complexity-based subsets...")
print()

# Subset 1: Simple (0-1 JOINs)
simple_train = train_df[train_df['num_joins'] <= 1].copy()
simple_dev = dev_df[dev_df['num_joins'] <= 1].copy()

print(f"SIMPLE (0-1 JOINs):")
print(f"  Train: {len(simple_train)} examples ({len(simple_train)/len(train_df)*100:.1f}%)")
print(f"  Dev: {len(simple_dev)} examples")
print(f"  Expected execution rate: 70-80%")
print()

# Subset 2: Moderate (0-2 JOINs)
moderate_train = train_df[train_df['num_joins'] <= 2].copy()
moderate_dev = dev_df[dev_df['num_joins'] <= 2].copy()

print(f"MODERATE (0-2 JOINs):")
print(f"  Train: {len(moderate_train)} examples ({len(moderate_train)/len(train_df)*100:.1f}%)")
print(f"  Dev: {len(moderate_dev)} examples")
print(f"  Expected execution rate: 60-70%")
print()

# Subset 3: All
print(f"ALL (no filter):")
print(f"  Train: {len(train_df)} examples (100.0%)")
print(f"  Dev: {len(dev_df)} examples")
print(f"  Expected execution rate: 50-60%")
print()

# ============================================================================
# STEP 4: FORMAT FOR TRAINING (Instruction Format)
# ============================================================================

print("STEP 4: Formatting for training...")
print()

def format_schema_simplified(db_id, example):
    """
    Create simplified schema text
    For now, just placeholder - will need actual schema from databases
    """
    # This is a simplified version - actual schema would come from train_tables.json
    # or database files
    return f"Database: {db_id}\n[Schema information would be loaded from train_tables.json]"


def format_training_example(row):
    """
    Format example into instruction template for fine-tuning
    
    Returns dict with 'text', 'prompt', 'completion' fields
    """
    # Simplified schema (you'll enhance this with actual schema data)
    schema_text = format_schema_simplified(row['db_id'], row)
    
    # ChatML format (Llama 3.2 uses this)
    prompt = f"""<|im_start|>system
You are a SQL expert assistant. Generate executable SQL queries based on the given database schema and user question.<|im_end|>
<|im_start|>user
Database: {row['db_id']}

Question: {row['question']}<|im_end|>
<|im_start|>assistant
"""
    
    completion = f"{row['SQL']}<|im_end|>"
    
    full_text = prompt + completion
    
    return {
        'text': full_text,
        'prompt': prompt,
        'completion': completion,
        'db_id': row['db_id'],
        'question': row['question'],
        'SQL': row['SQL'],
        'num_joins': row.get('num_joins', 0),
        'num_tables': row.get('num_tables', 0),
    }


# Format each subset
print("Formatting SIMPLE subset...")
simple_train_formatted = simple_train.apply(format_training_example, axis=1).tolist()
simple_dev_formatted = simple_dev.apply(format_training_example, axis=1).tolist()

print("Formatting MODERATE subset...")
moderate_train_formatted = moderate_train.apply(format_training_example, axis=1).tolist()
moderate_dev_formatted = moderate_dev.apply(format_training_example, axis=1).tolist()

print("Formatting ALL subset...")
all_train_formatted = train_df.apply(format_training_example, axis=1).tolist()
all_dev_formatted = dev_df.apply(format_training_example, axis=1).tolist()

print("✓ Formatting complete")
print()

# ============================================================================
# STEP 5: CREATE TRAIN/VAL SPLITS (80/20)
# ============================================================================

print("STEP 5: Creating train/val splits...")
print()

def create_split(train_data, split_name):
    """Create 80/20 train/val split"""
    train_examples = pd.DataFrame(train_data)
    
    # 80% train, 20% val
    train_split, val_split = train_test_split(
        train_examples,
        test_size=0.2,
        random_state=42
    )
    
    print(f"{split_name}:")
    print(f"  Train: {len(train_split)} examples")
    print(f"  Val: {len(val_split)} examples")
    
    return train_split, val_split


# Split each subset
simple_train_split, simple_val_split = create_split(simple_train_formatted, "SIMPLE")
moderate_train_split, moderate_val_split = create_split(moderate_train_formatted, "MODERATE")
all_train_split, all_val_split = create_split(all_train_formatted, "ALL")

print()

# ============================================================================
# STEP 6: SAVE PROCESSED DATA
# ============================================================================

print("STEP 6: Saving processed datasets...")
print()

# Create output directory
os.makedirs('data/processed', exist_ok=True)

# Save SIMPLE
simple_train_split.to_json('data/processed/simple_train.jsonl', orient='records', lines=True)
simple_val_split.to_json('data/processed/simple_val.jsonl', orient='records', lines=True)
pd.DataFrame(simple_dev_formatted).to_json('data/processed/simple_test.jsonl', orient='records', lines=True)
print("✓ Saved SIMPLE dataset (0-1 JOINs)")

# Save MODERATE
moderate_train_split.to_json('data/processed/moderate_train.jsonl', orient='records', lines=True)
moderate_val_split.to_json('data/processed/moderate_val.jsonl', orient='records', lines=True)
pd.DataFrame(moderate_dev_formatted).to_json('data/processed/moderate_test.jsonl', orient='records', lines=True)
print("✓ Saved MODERATE dataset (0-2 JOINs)")

# Save ALL
all_train_split.to_json('data/processed/all_train.jsonl', orient='records', lines=True)
all_val_split.to_json('data/processed/all_val.jsonl', orient='records', lines=True)
pd.DataFrame(all_dev_formatted).to_json('data/processed/all_test.jsonl', orient='records', lines=True)
print("✓ Saved ALL dataset (no filter)")

print()

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

print("="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print()

print("Dataset Summary:")
print()
print("SIMPLE (0-1 JOINs) - For Experiment 1:")
print(f"  Train: {len(simple_train_split)} | Val: {len(simple_val_split)} | Test: {len(simple_dev_formatted)}")
print(f"  Files: data/processed/simple_*.jsonl")
print()

print("MODERATE (0-2 JOINs) - For Experiment 2-3:")
print(f"  Train: {len(moderate_train_split)} | Val: {len(moderate_val_split)} | Test: {len(moderate_dev_formatted)}")
print(f"  Files: data/processed/moderate_*.jsonl")
print()

print("ALL (no filter) - For Experiment 3-5:")
print(f"  Train: {len(all_train_split)} | Val: {len(all_val_split)} | Test: {len(all_dev_formatted)}")
print(f"  Files: data/processed/all_*.jsonl")
print()

print("="*80)
print("Next Steps:")
print("1. Update QLoRA training configs to use these datasets")
print("2. Start training Experiment 1 (simple dataset)")
print("3. Evaluation will use test sets")
print("="*80)

# Save summary
summary = {
    "preprocessing_method": "complexity_based_filtering",
    "filter_criteria": "number_of_joins",
    "datasets": {
        "simple": {
            "filter": "num_joins <= 1",
            "train": len(simple_train_split),
            "val": len(simple_val_split),
            "test": len(simple_dev_formatted),
            "expected_execution_rate": "70-80%"
        },
        "moderate": {
            "filter": "num_joins <= 2",
            "train": len(moderate_train_split),
            "val": len(moderate_val_split),
            "test": len(moderate_dev_formatted),
            "expected_execution_rate": "60-70%"
        },
        "all": {
            "filter": "none",
            "train": len(all_train_split),
            "val": len(all_val_split),
            "test": len(all_dev_formatted),
            "expected_execution_rate": "50-60%"
        }
    }
}

with open('outputs/preprocessing_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Saved preprocessing summary to outputs/preprocessing_summary.json")