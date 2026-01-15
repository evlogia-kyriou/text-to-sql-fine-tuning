"""
BIRD Dataset Analysis - Day 1
==============================

Complete analysis of BIRD Text2SQL dataset for DOKU project
This script analyzes train and dev sets to understand:
- Dataset size and structure
- Difficulty distribution
- Database diversity
- SQL complexity patterns
- Token length statistics
- Schema characteristics

Run this in Jupyter notebook for best results!
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("BIRD DATASET ANALYSIS")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("STEP 1: LOADING DATA")
print("-"*80)

# Adjust these paths to where YOUR BIRD data is located
TRAIN_PATH = "data/raw/train/train.json"  # or "data/raw/train.json"
DEV_PATH = "data/raw/dev/dev.json"        # or "data/raw/dev.json"

# Also check if databases folder exists
DB_PATH = "data/raw/train/train_databases"  # Contains schema SQLite files

try:
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"✓ Loaded train data: {len(train_data)} examples")
except FileNotFoundError:
    print(f"❌ Train file not found at: {TRAIN_PATH}")
    print("   Please update TRAIN_PATH to correct location")
    train_data = None

try:
    with open(DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print(f"✓ Loaded dev data: {len(dev_data)} examples")
except FileNotFoundError:
    print(f"❌ Dev file not found at: {DEV_PATH}")
    print("   Please update DEV_PATH to correct location")
    dev_data = None

print()

# Stop if data not loaded
if train_data is None:
    print("Please fix the file paths and run again!")
    exit()

# ============================================================================
# STEP 2: EXAMINE STRUCTURE
# ============================================================================

print("STEP 2: DATA STRUCTURE")
print("-"*80)

# Look at first example
example = train_data[0]
print("Fields in each example:")
for key in example.keys():
    print(f"  - {key}: {type(example[key]).__name__}")

print("\nExample record:")
print(json.dumps(example, indent=2)[:500] + "...")
print()

# Convert to DataFrames
train_df = pd.DataFrame(train_data)
dev_df = pd.DataFrame(dev_data) if dev_data else pd.DataFrame()

print(f"Train DataFrame: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print()

# ============================================================================
# STEP 3: DIFFICULTY DISTRIBUTION
# ============================================================================

print("STEP 3: DIFFICULTY DISTRIBUTION")
print("-"*80)

if 'difficulty' in train_df.columns:
    difficulty_counts = train_df['difficulty'].value_counts().sort_index()
    difficulty_pct = (difficulty_counts / len(train_df) * 100).round(1)
    
    print("Difficulty breakdown:")
    for diff, count in difficulty_counts.items():
        pct = difficulty_pct[diff]
        print(f"  {diff:15s}: {count:5d} ({pct:5.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    difficulty_counts.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('BIRD Dataset - Difficulty Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Difficulty Level')
    axes[0].set_ylabel('Number of Examples')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for i, (diff, count) in enumerate(difficulty_counts.items()):
        axes[0].text(i, count + 50, str(count), ha='center', va='bottom')
    
    # Pie chart
    colors = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']
    axes[1].pie(difficulty_counts.values, labels=difficulty_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Difficulty Distribution %', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/difficulty_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/visualizations/difficulty_distribution.png")
    plt.show()
else:
    print("  No 'difficulty' field in data")

print()

# ============================================================================
# STEP 4: DATABASE DIVERSITY
# ============================================================================

print("STEP 4: DATABASE DIVERSITY")
print("-"*80)

if 'db_id' in train_df.columns:
    unique_dbs = train_df['db_id'].nunique()
    db_counts = train_df['db_id'].value_counts()
    
    print(f"Unique databases: {unique_dbs}")
    print(f"Examples per database: {len(train_df) / unique_dbs:.1f} average")
    print()
    
    print("Top 10 databases by example count:")
    for db, count in db_counts.head(10).items():
        print(f"  {db:25s}: {count:4d} examples")
    
    # Visualize top 20
    fig, ax = plt.subplots(figsize=(14, 6))
    db_counts.head(20).plot(kind='barh', ax=ax, color='teal')
    ax.set_title('Top 20 Databases by Example Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Examples')
    ax.set_ylabel('Database ID')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/database_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/visualizations/database_distribution.png")
    plt.show()
else:
    print("  No 'db_id' field in data")

print()

# ============================================================================
# STEP 5: SQL COMPLEXITY ANALYSIS
# ============================================================================

print("STEP 5: SQL COMPLEXITY ANALYSIS")
print("-"*80)

def analyze_sql_complexity(sql):
    """Extract SQL features"""
    sql_upper = sql.upper()
    
    # Count keywords
    features = {
        'length': len(sql),
        'has_join': 'JOIN' in sql_upper,
        'num_joins': sql_upper.count('JOIN'),
        'has_subquery': '(' in sql and sql_upper.count('SELECT') > 1,
        'num_subqueries': sql_upper.count('SELECT') - 1,
        'has_group_by': 'GROUP BY' in sql_upper,
        'has_order_by': 'ORDER BY' in sql_upper,
        'has_limit': 'LIMIT' in sql_upper,
        'has_having': 'HAVING' in sql_upper,
        'has_distinct': 'DISTINCT' in sql_upper,
        'has_union': 'UNION' in sql_upper,
        'has_intersect': 'INTERSECT' in sql_upper,
        'has_except': 'EXCEPT' in sql_upper,
    }
    
    # Count tables (approximate)
    from_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
    join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
    features['num_tables'] = len(set(from_matches + join_matches))
    
    return features

# Apply analysis
print("Analyzing SQL queries...")
if 'SQL' in train_df.columns:
    train_df['sql_features'] = train_df['SQL'].apply(analyze_sql_complexity)
    
    # Extract into columns
    feature_df = pd.DataFrame(train_df['sql_features'].tolist())
    
    print("\nSQL Statistics:")
    print(f"  Average query length: {feature_df['length'].mean():.0f} characters")
    print(f"  Median query length: {feature_df['length'].median():.0f} characters")
    print(f"  Max query length: {feature_df['length'].max():.0f} characters")
    print()
    
    print("Feature frequency:")
    bool_features = [col for col in feature_df.columns if col.startswith('has_')]
    for feature in bool_features:
        count = feature_df[feature].sum()
        pct = count / len(feature_df) * 100
        print(f"  {feature:20s}: {count:5d} ({pct:5.1f}%)")
    
    print()
    print("Join distribution:")
    join_dist = feature_df['num_joins'].value_counts().sort_index()
    for num_joins, count in join_dist.items():
        pct = count / len(feature_df) * 100
        print(f"  {num_joins} joins: {count:5d} ({pct:5.1f}%)")
    
    # Visualize complexity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Query length distribution
    axes[0, 0].hist(feature_df['length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('SQL Query Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Query Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(feature_df['length'].median(), color='red', linestyle='--', label='Median')
    axes[0, 0].legend()
    
    # Number of joins
    join_dist.plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Number of JOINs Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of JOINs')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Number of tables
    table_dist = feature_df['num_tables'].value_counts().sort_index()
    table_dist.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Number of Tables Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Tables')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Feature presence
    feature_counts = {feat.replace('has_', ''): feature_df[feat].sum() 
                     for feat in bool_features}
    feature_counts = dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))
    
    axes[1, 1].barh(list(feature_counts.keys()), list(feature_counts.values()), color='mediumpurple')
    axes[1, 1].set_title('SQL Feature Frequency', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Queries')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/sql_complexity.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/visualizations/sql_complexity.png")
    plt.show()

print()

# ============================================================================
# STEP 6: TOKEN LENGTH ANALYSIS
# ============================================================================

print("STEP 6: TOKEN LENGTH ANALYSIS")
print("-"*80)

# Simple approximation: 4 characters ≈ 1 token
def estimate_tokens(text):
    """Rough token estimation (actual tokenization would be more accurate)"""
    if pd.isna(text):
        return 0
    return len(str(text)) // 4

print("Estimating token lengths (rough approximation)...")

if 'question' in train_df.columns and 'SQL' in train_df.columns:
    train_df['question_tokens'] = train_df['question'].apply(estimate_tokens)
    train_df['sql_tokens'] = train_df['SQL'].apply(estimate_tokens)
    
    # Calculate total tokens (including schema would add more)
    # Assume schema adds ~200 tokens on average
    train_df['estimated_total_tokens'] = train_df['question_tokens'] + train_df['sql_tokens'] + 200
    
    print("\nToken Statistics:")
    print("Question tokens:")
    print(f"  Mean:   {train_df['question_tokens'].mean():.1f}")
    print(f"  Median: {train_df['question_tokens'].median():.1f}")
    print(f"  95th percentile: {train_df['question_tokens'].quantile(0.95):.1f}")
    print()
    
    print("SQL tokens:")
    print(f"  Mean:   {train_df['sql_tokens'].mean():.1f}")
    print(f"  Median: {train_df['sql_tokens'].median():.1f}")
    print(f"  95th percentile: {train_df['sql_tokens'].quantile(0.95):.1f}")
    print()
    
    print("Estimated total tokens (question + SQL + schema):")
    print(f"  Mean:   {train_df['estimated_total_tokens'].mean():.1f}")
    print(f"  Median: {train_df['estimated_total_tokens'].median():.1f}")
    print(f"  95th percentile: {train_df['estimated_total_tokens'].quantile(0.95):.1f}")
    
    # Context window implications
    print()
    print("Context window fit:")
    fit_512 = (train_df['estimated_total_tokens'] <= 512).sum()
    fit_1024 = (train_df['estimated_total_tokens'] <= 1024).sum()
    print(f"  Fit in 512 tokens:  {fit_512:5d} ({fit_512/len(train_df)*100:.1f}%)")
    print(f"  Fit in 1024 tokens: {fit_1024:5d} ({fit_1024/len(train_df)*100:.1f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Question tokens
    axes[0].hist(train_df['question_tokens'], bins=50, color='lightblue', edgecolor='black')
    axes[0].axvline(train_df['question_tokens'].median(), color='red', linestyle='--', 
                   label=f"Median: {train_df['question_tokens'].median():.0f}")
    axes[0].set_title('Question Token Length', fontweight='bold')
    axes[0].set_xlabel('Tokens')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # SQL tokens
    axes[1].hist(train_df['sql_tokens'], bins=50, color='lightcoral', edgecolor='black')
    axes[1].axvline(train_df['sql_tokens'].median(), color='red', linestyle='--',
                   label=f"Median: {train_df['sql_tokens'].median():.0f}")
    axes[1].set_title('SQL Token Length', fontweight='bold')
    axes[1].set_xlabel('Tokens')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # Total tokens
    axes[2].hist(train_df['estimated_total_tokens'], bins=50, color='lightgreen', edgecolor='black')
    axes[2].axvline(512, color='orange', linestyle='--', label='512 (model limit)')
    axes[2].axvline(1024, color='purple', linestyle='--', label='1024 (extended)')
    axes[2].axvline(train_df['estimated_total_tokens'].median(), color='red', linestyle='--',
                   label=f"Median: {train_df['estimated_total_tokens'].median():.0f}")
    axes[2].set_title('Estimated Total Tokens', fontweight='bold')
    axes[2].set_xlabel('Tokens')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/token_distributions.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/visualizations/token_distributions.png")
    plt.show()

print()

# ============================================================================
# STEP 7: DIFFICULTY vs COMPLEXITY CORRELATION
# ============================================================================

print("STEP 7: DIFFICULTY vs COMPLEXITY")
print("-"*80)

if 'difficulty' in train_df.columns and 'sql_features' in train_df.columns:
    # Merge feature data
    train_analysis = train_df.copy()
    feature_cols = pd.DataFrame(train_df['sql_features'].tolist())
    train_analysis = pd.concat([train_analysis, feature_cols], axis=1)
    
    # Analyze by difficulty
    print("Average SQL length by difficulty:")
    length_by_diff = train_analysis.groupby('difficulty')['length'].agg(['mean', 'median', 'max'])
    print(length_by_diff)
    print()
    
    print("Average number of joins by difficulty:")
    joins_by_diff = train_analysis.groupby('difficulty')['num_joins'].mean()
    print(joins_by_diff)
    print()
    
    print("Average number of tables by difficulty:")
    tables_by_diff = train_analysis.groupby('difficulty')['num_tables'].mean()
    print(tables_by_diff)
    print()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Box plot of query length by difficulty
    train_analysis.boxplot(column='length', by='difficulty', ax=axes[0])
    axes[0].set_title('SQL Length by Difficulty', fontweight='bold')
    axes[0].set_xlabel('Difficulty')
    axes[0].set_ylabel('Query Length (characters)')
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # Bar plot of avg joins
    joins_by_diff.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Average JOINs by Difficulty', fontweight='bold')
    axes[1].set_xlabel('Difficulty')
    axes[1].set_ylabel('Average Number of JOINs')
    axes[1].tick_params(axis='x', rotation=0)
    
    # Bar plot of avg tables
    tables_by_diff.plot(kind='bar', ax=axes[2], color='darkorange')
    axes[2].set_title('Average Tables by Difficulty', fontweight='bold')
    axes[2].set_xlabel('Difficulty')
    axes[2].set_ylabel('Average Number of Tables')
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/difficulty_complexity_correlation.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: outputs/visualizations/difficulty_complexity_correlation.png")
    plt.show()

print()

# ============================================================================
# STEP 8: SAVE SUMMARY STATISTICS
# ============================================================================

print("STEP 8: SAVING SUMMARY STATISTICS")
print("-"*80)

# Compile statistics
summary_stats = {
    "dataset": "BIRD-SQL",
    "total_train_examples": len(train_df),
    "total_dev_examples": len(dev_df) if not dev_df.empty else 0,
    "unique_databases": train_df['db_id'].nunique() if 'db_id' in train_df.columns else 0,
}

if 'difficulty' in train_df.columns:
    summary_stats["difficulty_distribution"] = train_df['difficulty'].value_counts().to_dict()

if 'sql_features' in train_df.columns:
    summary_stats["sql_statistics"] = {
        "avg_query_length": float(feature_df['length'].mean()),
        "median_query_length": float(feature_df['length'].median()),
        "queries_with_joins": int(feature_df['has_join'].sum()),
        "queries_with_subqueries": int(feature_df['has_subquery'].sum()),
        "queries_with_group_by": int(feature_df['has_group_by'].sum()),
    }

if 'question_tokens' in train_df.columns:
    summary_stats["token_statistics"] = {
        "question_median_tokens": float(train_df['question_tokens'].median()),
        "question_95th_percentile": float(train_df['question_tokens'].quantile(0.95)),
        "sql_median_tokens": float(train_df['sql_tokens'].median()),
        "sql_95th_percentile": float(train_df['sql_tokens'].quantile(0.95)),
        "fit_in_512_tokens": int((train_df['estimated_total_tokens'] <= 512).sum()),
        "fit_in_1024_tokens": int((train_df['estimated_total_tokens'] <= 1024).sum()),
    }

# Save to JSON
import os
os.makedirs('outputs', exist_ok=True)
with open('outputs/dataset_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("✓ Saved: outputs/dataset_statistics.json")

# Save processed data
train_df.to_json('data/processed/train_analyzed.jsonl', orient='records', lines=True)
print("✓ Saved: data/processed/train_analyzed.jsonl")

if not dev_df.empty:
    dev_df.to_json('data/processed/dev_analyzed.jsonl', orient='records', lines=True)
    print("✓ Saved: data/processed/dev_analyzed.jsonl")

print()

# ============================================================================
# STEP 9: RECOMMENDATIONS FOR EXPERIMENTS
# ============================================================================

print("STEP 9: RECOMMENDATIONS FOR EXPERIMENTS")
print("="*80)

if 'difficulty' in train_df.columns:
    easy_count = (train_df['difficulty'] == 'easy').sum()
    medium_count = (train_df['difficulty'] == 'medium').sum()
    hard_count = (train_df['difficulty'] == 'hard').sum()
    
    print("Suggested experiment data splits:")
    print()
    print(f"Experiment 1 (Baseline - Easy only):")
    print(f"  Use: {easy_count} easy examples")
    print(f"  Expected execution rate: 70-80%")
    print()
    
    print(f"Experiment 2-3 (Easy + Medium):")
    print(f"  Use: {easy_count + medium_count} examples")
    print(f"  Expected execution rate: 55-65%")
    print()
    
    if hard_count > 0:
        print(f"Optional Experiment (All difficulties):")
        print(f"  Use: {len(train_df)} examples")
        print(f"  Expected execution rate: 40-50%")
        print(f"  ⚠️ Warning: Hard queries may not fit in 512 token limit")

if 'estimated_total_tokens' in train_df.columns:
    print()
    print("Token length recommendations:")
    fit_512 = (train_df['estimated_total_tokens'] <= 512).sum()
    fit_1024 = (train_df['estimated_total_tokens'] <= 1024).sum()
    
    print(f"  Start with max_seq_length=512: {fit_512} examples fit ({fit_512/len(train_df)*100:.1f}%)")
    print(f"  Can extend to 1024 if needed: {fit_1024} examples fit ({fit_1024/len(train_df)*100:.1f}%)")

print()
print("="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print()
print("Next steps:")
print("1. Review visualizations in outputs/visualizations/")
print("2. Check summary statistics in outputs/dataset_statistics.json")
print("3. Proceed to data preprocessing (Day 2-3)")
print("4. Start with Easy examples for Experiment 1")