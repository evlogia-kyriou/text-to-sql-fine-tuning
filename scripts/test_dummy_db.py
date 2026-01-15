import sqlite3
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from compare_models import YourModel

def test_on_dummy_database(model_path):
    """Test model on dummy database"""
    
    print("="*80)
    print("TESTING ON DUMMY DATABASE")
    print("="*80)
    print("\nClient Criteria:")
    print("  1. ✅ Basic: Can generate SQL query")
    print("  2. ✅ Bonus: SQL is executable")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = YourModel(model_path)
    
    # Load dummy database info
    dummy_dir = Path('data/dummy')
    
    with open(dummy_dir / 'schema.json', 'r') as f:
        schema = json.load(f)
    
    with open(dummy_dir / 'test_questions.json', 'r') as f:
        questions = json.load(f)
    
    db_path = dummy_dir / 'company.sqlite'
    
    print(f"Database: {db_path}")
    print(f"Test questions: {len(questions)}")
    
    # Test each question
    results = {
        'total': len(questions),
        'sql_generated': 0,
        'sql_executable': 0,
        'exact_match': 0,
        'examples': []
    }
    
    print("\n" + "="*80)
    print("TESTING QUESTIONS")
    print("="*80)
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'='*80}")
        print(f"Difficulty: {q['difficulty'].upper()}")
        print(f"Question: {q['question']}")
        
        # Generate SQL
        try:
            generated_sql = model.generate(q['question'], schema, q.get('evidence', ''))
            results['sql_generated'] += 1
            
            print(f"\n✓ Generated SQL:")
            print(f"  {generated_sql}")
            
            # Test execution
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(generated_sql)
                result = cursor.fetchall()
                conn.close()
                
                results['sql_executable'] += 1
                print(f"\n✓ Execution SUCCESSFUL!")
                print(f"  Returned {len(result)} rows")
                
                # Show first few results
                if result:
                    print(f"  Sample output: {result[:3]}")
                
            except Exception as e:
                print(f"\n✗ Execution FAILED:")
                print(f"  Error: {e}")
        
        except Exception as e:
            print(f"\n✗ Generation FAILED:")
            print(f"  Error: {e}")
        
        # Show ground truth
        print(f"\nGround truth SQL:")
        print(f"  {q['SQL']}")
        
        # Check exact match
        if generated_sql.strip().lower() == q['SQL'].strip().lower():
            results['exact_match'] += 1
            print("\n💯 EXACT MATCH!")
        
        results['examples'].append({
            'question': q['question'],
            'difficulty': q['difficulty'],
            'generated_sql': generated_sql,
            'ground_truth': q['SQL'],
            'executable': results['sql_executable'] > len(results['examples'])
        })
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    gen_rate = results['sql_generated'] / results['total']
    exec_rate = results['sql_executable'] / results['total']
    match_rate = results['exact_match'] / results['total']
    
    print(f"\n📊 Performance:")
    print(f"   Total questions:     {results['total']}")
    print(f"   SQL Generated:       {results['sql_generated']}/{results['total']} ({gen_rate*100:.1f}%)")
    print(f"   SQL Executable:      {results['sql_executable']}/{results['total']} ({exec_rate*100:.1f}%)")
    print(f"   Exact Match:         {results['exact_match']}/{results['total']} ({match_rate*100:.1f}%)")
    
    # Client criteria
    print("\n" + "="*80)
    print("CLIENT CRITERIA ASSESSMENT")
    print("="*80)
    
    print(f"\n✅ CRITERION 1: Can generate SQL query")
    print(f"   Result: {results['sql_generated']}/{results['total']} ({'✅ PASS' if gen_rate >= 0.9 else '⚠️ PARTIAL'})")
    
    print(f"\n🎁 CRITERION 2: SQL is executable (BONUS)")
    print(f"   Result: {results['sql_executable']}/{results['total']} ({exec_rate*100:.1f}%)")
    if exec_rate >= 0.8:
        print(f"   Status: ✅ EXCELLENT - Bonus criteria MET!")
    elif exec_rate >= 0.6:
        print(f"   Status: ✅ GOOD - Bonus criteria PARTIALLY met")
    else:
        print(f"   Status: ⚠️ NEEDS IMPROVEMENT")
    
    # By difficulty
    print("\n📊 Performance by Difficulty:")
    by_difficulty = {'easy': {'total': 0, 'exec': 0}, 
                     'medium': {'total': 0, 'exec': 0}, 
                     'hard': {'total': 0, 'exec': 0}}
    
    for ex in results['examples']:
        diff = ex['difficulty']
        by_difficulty[diff]['total'] += 1
        if ex['executable']:
            by_difficulty[diff]['exec'] += 1
    
    for diff in ['easy', 'medium', 'hard']:
        if by_difficulty[diff]['total'] > 0:
            rate = by_difficulty[diff]['exec'] / by_difficulty[diff]['total']
            print(f"   {diff.capitalize():8s}: {by_difficulty[diff]['exec']}/{by_difficulty[diff]['total']} ({rate*100:.0f}%)")
    
    # Save results
    output_file = 'dummy_db_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    MODEL_PATH = "models/qwen3b_baseline_r16_20260114"
    
    # Check if dummy DB exists
    if not Path('data/dummy/company.sqlite').exists():
        print("⚠️  Dummy database not found!")
        print("   Creating it now...\n")
        import create_dummy_db
        create_dummy_db.create_dummy_database()
        print()
    
    # Test model
    results = test_on_dummy_database(MODEL_PATH)