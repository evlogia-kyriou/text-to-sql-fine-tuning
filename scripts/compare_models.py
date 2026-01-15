import torch
import json
import sqlite3
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import time
from compare_sql_outputs import enhanced_evaluate_model_pair, print_sql_comparison, compare_sql_side_by_side

# Try to import vLLM (for XiYanSQL)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not installed. Will only test your model.")
    print("   Install with: pip install vllm")


class YourModel:
    """Your fine-tuned Qwen model"""
    
    def __init__(self, model_path):
        print("Loading YOUR model...")
        base_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Your model loaded!")
    
    def format_schema(self, schema):
        """Format schema same as training"""
        schema_text = "Tables:\n"
        for table_idx, table_name in enumerate(schema['table_names']):
            schema_text += f"\n{table_name}:\n"
            for col_idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                if tbl_idx == table_idx:
                    col_type = schema['column_types'][col_idx]
                    is_pk = col_idx in schema.get('primary_keys', [])
                    pk_marker = " [PRIMARY KEY]" if is_pk else ""
                    schema_text += f"  • {col_name} ({col_type}){pk_marker}\n"
        return schema_text
    
    def generate(self, question, schema, evidence=""):
        """Generate SQL"""
        schema_text = self.format_schema(schema)
        evidence_text = f"\n\nHint: {evidence}" if evidence else ""
        
        prompt = f"""<|im_start|>system
You are an expert at writing SQL queries. Generate a valid SQL query that answers the given question based on the provided database schema.<|im_end|>
<|im_start|>user
Database: {schema['db_id']}

{schema_text}

Question: {question}{evidence_text}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        sql = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        if '<|im_end|>' in sql:
            sql = sql.split('<|im_end|>')[0]
        
        return sql.strip()


class XiYanSQLModel:
    """XiYanSQL-QwenCoder-3B-2504"""
    
    def __init__(self):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available")
        
        print("Loading XiYanSQL model...")
        model_path = "XGenerationLab/XiYanSQL-QwenCoder-3B-2504"
        
        # Adapted for single GPU
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Changed from 8 to 1
            gpu_memory_utilization=0.85,  # Use 85% of GPU
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.1,
            max_tokens=1024
        )
        print("✓ XiYanSQL model loaded!")
    
    def format_schema(self, schema):
        """Format schema for XiYanSQL"""
        # XiYanSQL uses specific format
        schema_lines = []
        for table_idx, table_name in enumerate(schema['table_names']):
            cols = []
            for col_idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                if tbl_idx == table_idx:
                    col_type = schema['column_types'][col_idx]
                    cols.append(f"{col_name} {col_type}")
            schema_lines.append(f"CREATE TABLE {table_name} ({', '.join(cols)})")
        return "\n".join(schema_lines)
    
    def generate(self, question, schema, evidence=""):
        """Generate SQL using XiYanSQL"""
        db_schema = self.format_schema(schema)
        
        # XiYanSQL prompt template (SQLite dialect)
        prompt = f"""You are a SQLite expert. Given an input question, create a syntactically correct SQLite query.

Database Schema:
{db_schema}

Question: {question}

Evidence: {evidence if evidence else "None"}

Generate the SQL query:"""
        
        message = [{'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.generate([text], self.sampling_params)
        sql = outputs[0].outputs[0].text.strip()
        
        return sql

def compare_with_sql_analysis(your_model_path, num_examples=10):
    """
    Enhanced comparison with detailed SQL analysis
    """
    
    print("="*80)
    print("DETAILED SQL COMPARISON: YOUR MODEL vs XiYanSQL")
    print("="*80)
    print("\nThis analysis will show:")
    print("  1. ✅ SQL generation (both models)")
    print("  2. ✅ Execution success (both models)")
    print("  3. 🔍 Side-by-side SQL comparison")
    print("  4. 📊 Pattern analysis (how they differ)")
    print("="*80)
    
    # Load dev data
    print("\nLoading dev data...")
    with open('data/raw/dev/dev.json', 'r') as f:
        dev_data = json.load(f)
    
    with open('data/raw/dev/dev_tables.json', 'r') as f:
        dev_schemas = json.load(f)
        schema_dict = {s['db_id']: s for s in dev_schemas}
    
    dev_data = dev_data[:num_examples]
    db_dir = Path('data/raw/dev/dev_databases')
    
    # Load models
    print("\nLoading your model...")
    your_model = YourModel(your_model_path)
    
    if not VLLM_AVAILABLE:
        print("\n⚠️ vLLM not available. Cannot load XiYanSQL for comparison.")
        print("   Install with: pip install vllm")
        return None
    
    print("Loading XiYanSQL model...")
    try:
        xiyan_model = XiYanSQLModel()
    except Exception as e:
        print(f"\n⚠️ Could not load XiYanSQL: {e}")
        return None
    
    # Run enhanced comparison
    report = enhanced_evaluate_model_pair(
        your_model,
        xiyan_model,
        dev_data,
        schema_dict,
        db_dir,
        num_examples
    )
    
    return report

def test_sql_execution(sql, db_path):
    """Test if SQL executes"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, len(results), None
    except Exception as e:
        return False, 0, str(e)


def evaluate_model(model, dev_data, schema_dict, db_dir, model_name):
    """Evaluate one model on dev set"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    results = {
        'model_name': model_name,
        'total': 0,
        'sql_generated': 0,
        'sql_executable': 0,
        'generation_failed': 0,
        'execution_errors': [],
        'examples': []
    }
    
    start_time = time.time()
    
    for example in tqdm(dev_data, desc=f"Testing {model_name}"):
        results['total'] += 1
        
        # Get schema
        schema = schema_dict.get(example['db_id'])
        if not schema:
            continue
        
        # Generate SQL
        try:
            generated_sql = model.generate(
                example['question'],
                schema,
                example.get('evidence', '')
            )
            sql_generated = True
            results['sql_generated'] += 1
        except Exception as e:
            sql_generated = False
            generated_sql = None
            results['generation_failed'] += 1
            print(f"⚠️  Generation failed: {e}")
        
        # Test execution
        executable = False
        error = None
        if sql_generated and generated_sql:
            db_path = Path(db_dir) / example['db_id'] / f"{example['db_id']}.sqlite"
            
            if db_path.exists():
                success, row_count, error = test_sql_execution(generated_sql, db_path)
                if success:
                    executable = True
                    results['sql_executable'] += 1
                else:
                    results['execution_errors'].append({
                        'question': example['question'],
                        'sql': generated_sql,
                        'error': error
                    })
        
        # Save example result
        results['examples'].append({
            'db_id': example['db_id'],
            'question': example['question'],
            'evidence': example.get('evidence', ''),
            'ground_truth': example['SQL'],
            'generated_sql': generated_sql,
            'sql_generated': sql_generated,
            'sql_executable': executable,
            'error': error
        })
    
    elapsed = time.time() - start_time
    results['elapsed_time'] = elapsed
    
    return results


def compare_models(your_model_path, num_examples=None):
    """
    Compare your model vs XiYanSQL on dev set
    """
    
    print("="*80)
    print("MODEL COMPARISON: YOUR MODEL vs XiYanSQL")
    print("="*80)
    print("\nClient Criteria:")
    print("  1. ✅ Basic: Can generate SQL query")
    print("  2. ✅ Bonus: SQL is executable")
    print("="*80)
    
    # Load dev data
    print("\nLoading dev data...")
    with open('data/raw/dev/dev.json', 'r') as f:
        dev_data = json.load(f)
    
    with open('data/raw/dev/dev_tables.json', 'r') as f:
        dev_schemas = json.load(f)
        schema_dict = {s['db_id']: s for s in dev_schemas}
    
    if num_examples:
        dev_data = dev_data[:num_examples]
        print(f"Using {num_examples} examples for quick test")
    else:
        print(f"Using all {len(dev_data)} examples")
    
    db_dir = Path('data/raw/dev/dev_databases')
    
    # Load your model
    your_model = YourModel(your_model_path)
    
    # Evaluate your model
    your_results = evaluate_model(
        your_model, 
        dev_data, 
        schema_dict, 
        db_dir,
        "Your Model (Qwen2.5-Coder-3B LoRA r=16)"
    )
    
    # Try to load and evaluate XiYanSQL
    xiyan_results = None
    if VLLM_AVAILABLE:
        try:
            xiyan_model = XiYanSQLModel()
            xiyan_results = evaluate_model(
                xiyan_model,
                dev_data,
                schema_dict,
                db_dir,
                "XiYanSQL-QwenCoder-3B-2504"
            )
        except Exception as e:
            print(f"\n⚠️  Could not load XiYanSQL: {e}")
            print("   Continuing with your model only...")
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    # Your model results
    print(f"\n📊 YOUR MODEL:")
    print(f"   Total examples:      {your_results['total']}")
    print(f"   SQL Generated:       {your_results['sql_generated']}/{your_results['total']} ({your_results['sql_generated']/your_results['total']*100:.1f}%)")
    print(f"   SQL Executable:      {your_results['sql_executable']}/{your_results['total']} ({your_results['sql_executable']/your_results['total']*100:.1f}%)")
    print(f"   Generation Failed:   {your_results['generation_failed']}")
    print(f"   Time:                {your_results['elapsed_time']:.1f}s")
    
    # XiYanSQL results
    if xiyan_results:
        print(f"\n📊 XiYanSQL-QwenCoder-3B-2504:")
        print(f"   Total examples:      {xiyan_results['total']}")
        print(f"   SQL Generated:       {xiyan_results['sql_generated']}/{xiyan_results['total']} ({xiyan_results['sql_generated']/xiyan_results['total']*100:.1f}%)")
        print(f"   SQL Executable:      {xiyan_results['sql_executable']}/{xiyan_results['total']} ({xiyan_results['sql_executable']/xiyan_results['total']*100:.1f}%)")
        print(f"   Generation Failed:   {xiyan_results['generation_failed']}")
        print(f"   Time:                {xiyan_results['elapsed_time']:.1f}s")
    
    # Client criteria assessment
    print("\n" + "="*80)
    print("CLIENT CRITERIA ASSESSMENT")
    print("="*80)
    
    print("\n✅ CRITERION 1: Can generate SQL query")
    print(f"   Your Model:  {your_results['sql_generated']}/{your_results['total']} ({'PASS' if your_results['sql_generated']/your_results['total'] > 0.9 else 'PARTIAL'})")
    if xiyan_results:
        print(f"   XiYanSQL:    {xiyan_results['sql_generated']}/{xiyan_results['total']} ({'PASS' if xiyan_results['sql_generated']/xiyan_results['total'] > 0.9 else 'PARTIAL'})")
    
    print("\n🎁 CRITERION 2: SQL is executable (BONUS)")
    your_exec_rate = your_results['sql_executable']/your_results['total']
    print(f"   Your Model:  {your_results['sql_executable']}/{your_results['total']} ({your_exec_rate*100:.1f}%) {'✅ EXCELLENT' if your_exec_rate > 0.7 else '✅ GOOD' if your_exec_rate > 0.5 else '⚠️ NEEDS IMPROVEMENT'}")
    if xiyan_results:
        xiyan_exec_rate = xiyan_results['sql_executable']/xiyan_results['total']
        print(f"   XiYanSQL:    {xiyan_results['sql_executable']}/{xiyan_results['total']} ({xiyan_exec_rate*100:.1f}%) {'✅ EXCELLENT' if xiyan_exec_rate > 0.7 else '✅ GOOD' if xiyan_exec_rate > 0.5 else '⚠️ NEEDS IMPROVEMENT'}")
        
        gap = xiyan_exec_rate - your_exec_rate
        print(f"\n   Gap:         {gap*100:+.1f} percentage points")
        if abs(gap) < 0.05:
            print(f"   Assessment:  ✅ Very competitive!")
        elif abs(gap) < 0.10:
            print(f"   Assessment:  ✅ Good performance")
        else:
            print(f"   Assessment:  ⚠️  Significant gap")
    
    # Save detailed results
    output = {
        'your_model': your_results,
        'xiyan_model': xiyan_results,
        'comparison': {
            'your_generation_rate': your_results['sql_generated']/your_results['total'],
            'your_execution_rate': your_results['sql_executable']/your_results['total'],
        }
    }
    
    if xiyan_results:
        output['comparison']['xiyan_generation_rate'] = xiyan_results['sql_generated']/xiyan_results['total']
        output['comparison']['xiyan_execution_rate'] = xiyan_results['sql_executable']/xiyan_results['total']
        output['comparison']['gap'] = xiyan_results['sql_executable']/xiyan_results['total'] - your_results['sql_executable']/your_results['total']
    
    with open('model_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: model_comparison_results.json")
    
    return output


if __name__ == "__main__":
    YOUR_MODEL_PATH = "models/qwen3b_baseline_r16_20260114"
    
    print("Choose evaluation mode:")
    print("1. Quick test (10 examples, ~5 minutes)")
    print("2. Moderate test (100 examples, ~30 minutes)")
    print("3. Full dev set (1534 examples, ~3-4 hours)")
    print("4. Detailed SQL comparison (10 examples, shows how SQL differs)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "4":
        results = compare_with_sql_analysis(YOUR_MODEL_PATH, num_examples=10)
    elif choice == "1":
        results = compare_models(YOUR_MODEL_PATH, num_examples=10, verbose=True)
    elif choice == "2":
        results = compare_models(YOUR_MODEL_PATH, num_examples=100, verbose=False)
    else:
        results = compare_models(YOUR_MODEL_PATH, num_examples=None, verbose=False)