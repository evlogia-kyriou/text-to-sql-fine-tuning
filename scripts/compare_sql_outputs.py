import json
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher
import re
from collections import defaultdict

def analyze_sql_differences(sql1, sql2):
    """Analyze differences between two SQL queries"""
    
    differences = {
        'similarity': 0.0,
        'structural_differences': [],
        'keyword_differences': [],
        'table_differences': [],
        'column_differences': []
    }
    
    # Calculate similarity
    differences['similarity'] = SequenceMatcher(None, sql1.lower(), sql2.lower()).ratio()
    
    # Extract SQL components
    sql1_upper = sql1.upper()
    sql2_upper = sql2.upper()
    
    # Check for structural differences
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'DISTINCT']
    
    for kw in keywords:
        in_sql1 = kw in sql1_upper
        in_sql2 = kw in sql2_upper
        
        if in_sql1 != in_sql2:
            if in_sql1:
                differences['structural_differences'].append(f"Your model uses {kw}, XiYanSQL doesn't")
            else:
                differences['structural_differences'].append(f"XiYanSQL uses {kw}, your model doesn't")
    
    # Extract table names (simple regex)
    def extract_tables(sql):
        # Match patterns like "FROM table" or "JOIN table"
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        return set(re.findall(pattern, sql, re.IGNORECASE))
    
    tables1 = extract_tables(sql1)
    tables2 = extract_tables(sql2)
    
    only_in_1 = tables1 - tables2
    only_in_2 = tables2 - tables1
    
    if only_in_1:
        differences['table_differences'].append(f"Your model uses tables: {', '.join(only_in_1)}")
    if only_in_2:
        differences['table_differences'].append(f"XiYanSQL uses tables: {', '.join(only_in_2)}")
    
    # Count JOINs
    joins1 = sql1_upper.count('JOIN')
    joins2 = sql2_upper.count('JOIN')
    
    if joins1 != joins2:
        differences['keyword_differences'].append(f"JOINs: Your model={joins1}, XiYanSQL={joins2}")
    
    # Check for aggregations
    agg_funcs = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
    for func in agg_funcs:
        in_sql1 = func in sql1_upper
        in_sql2 = func in sql2_upper
        
        if in_sql1 != in_sql2:
            if in_sql1:
                differences['keyword_differences'].append(f"Your model uses {func}")
            else:
                differences['keyword_differences'].append(f"XiYanSQL uses {func}")
    
    return differences


def categorize_sql_approach(sql):
    """Categorize the SQL approach"""
    sql_upper = sql.upper()
    
    categories = []
    
    # Complexity
    if 'JOIN' in sql_upper:
        join_count = sql_upper.count('JOIN')
        if join_count == 1:
            categories.append('Simple JOIN')
        elif join_count >= 2:
            categories.append('Multiple JOINs')
    else:
        categories.append('Single table')
    
    # Aggregation
    if any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
        categories.append('Aggregation')
    
    # Grouping
    if 'GROUP BY' in sql_upper:
        categories.append('Grouping')
    
    # Ordering
    if 'ORDER BY' in sql_upper:
        categories.append('Ordering')
    
    # Filtering
    if 'WHERE' in sql_upper:
        categories.append('Filtering')
    
    # Subquery
    if sql.count('SELECT') > 1:
        categories.append('Subquery')
    
    # DISTINCT
    if 'DISTINCT' in sql_upper:
        categories.append('DISTINCT')
    
    return categories if categories else ['Simple SELECT']


def compare_sql_side_by_side(your_sql, xiyan_sql, question, ground_truth=None):
    """Create a side-by-side comparison"""
    
    comparison = {
        'question': question,
        'your_model': {
            'sql': your_sql,
            'approach': categorize_sql_approach(your_sql),
            'length': len(your_sql),
            'tables_count': your_sql.upper().count('FROM') + your_sql.upper().count('JOIN'),
            'has_subquery': your_sql.count('SELECT') > 1
        },
        'xiyan_model': {
            'sql': xiyan_sql,
            'approach': categorize_sql_approach(xiyan_sql),
            'length': len(xiyan_sql),
            'tables_count': xiyan_sql.upper().count('FROM') + xiyan_sql.upper().count('JOIN'),
            'has_subquery': xiyan_sql.count('SELECT') > 1
        },
        'differences': analyze_sql_differences(your_sql, xiyan_sql),
        'ground_truth': ground_truth
    }
    
    return comparison


def print_sql_comparison(comparison, example_num, show_ground_truth=True):
    """Pretty print SQL comparison"""
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {example_num}: SQL COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nQuestion: {comparison['question']}")
    
    print(f"\n{'─'*80}")
    print("YOUR MODEL:")
    print(f"{'─'*80}")
    print(f"SQL: {comparison['your_model']['sql']}")
    print(f"Approach: {', '.join(comparison['your_model']['approach'])}")
    print(f"Complexity: {comparison['your_model']['tables_count']} table(s), {comparison['your_model']['length']} chars")
    
    print(f"\n{'─'*80}")
    print("XiYanSQL MODEL:")
    print(f"{'─'*80}")
    print(f"SQL: {comparison['xiyan_model']['sql']}")
    print(f"Approach: {', '.join(comparison['xiyan_model']['approach'])}")
    print(f"Complexity: {comparison['xiyan_model']['tables_count']} table(s), {comparison['xiyan_model']['length']} chars")
    
    if show_ground_truth and comparison.get('ground_truth'):
        print(f"\n{'─'*80}")
        print("GROUND TRUTH:")
        print(f"{'─'*80}")
        print(f"SQL: {comparison['ground_truth']}")
    
    # Show differences
    diff = comparison['differences']
    
    print(f"\n{'─'*80}")
    print("ANALYSIS:")
    print(f"{'─'*80}")
    print(f"Similarity: {diff['similarity']*100:.1f}%")
    
    if diff['structural_differences']:
        print("\n📐 Structural Differences:")
        for d in diff['structural_differences']:
            print(f"  • {d}")
    
    if diff['table_differences']:
        print("\n🗄️ Table Usage:")
        for d in diff['table_differences']:
            print(f"  • {d}")
    
    if diff['keyword_differences']:
        print("\n🔑 Keyword Differences:")
        for d in diff['keyword_differences']:
            print(f"  • {d}")
    
    # Determine which is more complex
    if comparison['your_model']['length'] > comparison['xiyan_model']['length'] * 1.2:
        print("\n💡 Your model's query is more verbose")
    elif comparison['xiyan_model']['length'] > comparison['your_model']['length'] * 1.2:
        print("\n💡 XiYanSQL's query is more verbose")
    else:
        print("\n💡 Both queries have similar complexity")


def analyze_common_patterns(comparisons):
    """Analyze patterns across all comparisons"""
    
    patterns = {
        'your_model_prefers': defaultdict(int),
        'xiyan_prefers': defaultdict(int),
        'common_mistakes_yours': defaultdict(int),
        'common_mistakes_xiyan': defaultdict(int)
    }
    
    for comp in comparisons:
        # Analyze approaches
        your_approach = set(comp['your_model']['approach'])
        xiyan_approach = set(comp['xiyan_model']['approach'])
        
        only_yours = your_approach - xiyan_approach
        only_xiyan = xiyan_approach - your_approach
        
        for approach in only_yours:
            patterns['your_model_prefers'][approach] += 1
        
        for approach in only_xiyan:
            patterns['xiyan_prefers'][approach] += 1
    
    return patterns


def generate_comparison_report(comparisons, output_file='sql_comparison_report.json'):
    """Generate comprehensive comparison report"""
    
    patterns = analyze_common_patterns(comparisons)
    
    report = {
        'summary': {
            'total_comparisons': len(comparisons),
            'avg_similarity': sum(c['differences']['similarity'] for c in comparisons) / len(comparisons),
            'your_model_avg_length': sum(c['your_model']['length'] for c in comparisons) / len(comparisons),
            'xiyan_avg_length': sum(c['xiyan_model']['length'] for c in comparisons) / len(comparisons)
        },
        'patterns': patterns,
        'detailed_comparisons': comparisons
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPARISON PATTERNS ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total comparisons: {report['summary']['total_comparisons']}")
    print(f"  Average similarity: {report['summary']['avg_similarity']*100:.1f}%")
    print(f"  Your model avg length: {report['summary']['your_model_avg_length']:.0f} chars")
    print(f"  XiYanSQL avg length: {report['summary']['xiyan_avg_length']:.0f} chars")
    
    if patterns['your_model_prefers']:
        print(f"\n🎯 Your Model Tends To Use:")
        for approach, count in sorted(patterns['your_model_prefers'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {approach}: {count} times")
    
    if patterns['xiyan_prefers']:
        print(f"\n🎯 XiYanSQL Tends To Use:")
        for approach, count in sorted(patterns['xiyan_prefers'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {approach}: {count} times")
    
    print(f"\n💾 Detailed report saved to: {output_file}")
    
    return report


# Integration with existing comparison script
def enhanced_evaluate_model_pair(your_model, xiyan_model, dev_data, schema_dict, db_dir, num_examples=10):
    """
    Evaluate both models and compare their SQL generation approaches
    """
    
    print("="*80)
    print("ENHANCED SQL COMPARISON ANALYSIS")
    print("="*80)
    print("\nComparing SQL generation strategies between:")
    print("  1. Your Model (Qwen2.5-Coder-3B LoRA r=16)")
    print("  2. XiYanSQL-QwenCoder-3B-2504")
    print("="*80)
    
    comparisons = []
    
    for i, example in enumerate(dev_data[:num_examples], 1):
        schema = schema_dict.get(example['db_id'])
        if not schema:
            continue
        
        print(f"\n{'🔍'*40}")
        print(f"Testing Example {i}/{num_examples}")
        print(f"{'🔍'*40}")
        
        # Generate SQL from both models
        try:
            your_sql = your_model.generate(
                example['question'],
                schema,
                example.get('evidence', '')
            )
        except Exception as e:
            print(f"⚠️ Your model failed: {e}")
            continue
        
        try:
            xiyan_sql = xiyan_model.generate(
                example['question'],
                schema,
                example.get('evidence', '')
            )
        except Exception as e:
            print(f"⚠️ XiYanSQL failed: {e}")
            continue
        
        # Compare
        comparison = compare_sql_side_by_side(
            your_sql,
            xiyan_sql,
            example['question'],
            example['SQL']
        )
        
        # Test execution for both
        db_path = Path(db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
        
        if db_path.exists():
            # Test your model
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(your_sql)
                cursor.fetchall()
                conn.close()
                comparison['your_model']['executable'] = True
                print("\n✅ Your model: EXECUTABLE")
            except Exception as e:
                comparison['your_model']['executable'] = False
                comparison['your_model']['error'] = str(e)
                print(f"\n❌ Your model: FAILED - {e}")
            
            # Test XiYanSQL
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(xiyan_sql)
                cursor.fetchall()
                conn.close()
                comparison['xiyan_model']['executable'] = True
                print("✅ XiYanSQL: EXECUTABLE")
            except Exception as e:
                comparison['xiyan_model']['executable'] = False
                comparison['xiyan_model']['error'] = str(e)
                print(f"❌ XiYanSQL: FAILED - {e}")
        
        # Print detailed comparison
        print_sql_comparison(comparison, i, show_ground_truth=True)
        
        comparisons.append(comparison)
    
    # Generate report
    report = generate_comparison_report(comparisons)
    
    return report


if __name__ == "__main__":
    # This can be called from the main comparison script
    print("SQL Comparison Module")
    print("Import this module in compare_models.py for detailed SQL analysis")