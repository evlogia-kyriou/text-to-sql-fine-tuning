import json
from pathlib import Path

class BIRDFormatter:
    """
    Simple but effective BIRD data formatter
    Focus: Get the format RIGHT, not fancy features
    """
    
    def __init__(self, schema_info, sample_values=None):
        self.schema_info = schema_info
        self.sample_values = sample_values or {}
    
    def format_schema(self, schema, include_samples=True):
        """
        Format schema clearly and completely
        """
        lines = []
        lines.append("Tables:")
        
        # For each table
        for table_idx, table_name in enumerate(schema['table_names']):
            lines.append(f"\n{table_name}:")
            
            # Get columns for this table
            for col_idx, (tbl_idx, col_name) in enumerate(schema['column_names']):
                if tbl_idx == table_idx:
                    col_type = schema['column_types'][col_idx]
                    
                    # Mark primary keys
                    is_pk = col_idx in schema.get('primary_keys', [])
                    pk_marker = " [PRIMARY KEY]" if is_pk else ""
                    
                    line = f"  • {col_name} ({col_type}){pk_marker}"
                    
                    # Add sample values if available
                    if include_samples and schema['db_id'] in self.sample_values:
                        db_samples = self.sample_values[schema['db_id']]
                        if table_name in db_samples and col_name in db_samples[table_name]:
                            samples = db_samples[table_name][col_name]['samples']
                            if samples and len(samples) > 0:
                                # Show 2-3 examples
                                sample_str = ", ".join(str(s) for s in samples[:3])
                                line += f" — Examples: {sample_str}"
                    
                    lines.append(line)
        
        # Add foreign keys
        if schema.get('foreign_keys'):
            lines.append("\nForeign Keys:")
            for fk in schema['foreign_keys']:
                from_idx, to_idx = fk
                from_tbl, from_col = schema['column_names'][from_idx]
                to_tbl, to_col = schema['column_names'][to_idx]
                
                from_table = schema['table_names'][from_tbl]
                to_table = schema['table_names'][to_tbl]
                
                lines.append(f"  • {from_table}.{from_col} → {to_table}.{to_col}")
        
        return "\n".join(lines)
    
    def format_example(self, row, schema):
        """
        Format one training example
        
        This is THE most important function!
        Get this right and your model will perform well.
        """
        
        # Format schema
        schema_text = self.format_schema(schema, include_samples=True)
        
        # Format evidence (if exists)
        evidence_text = ""
        if row.get('evidence') and row['evidence'].strip():
            evidence_text = f"\n\nHint: {row['evidence']}"
        
        # Build complete prompt
        prompt = f"""<|im_start|>system
You are an expert at writing SQL queries. Generate a valid SQL query that answers the given question based on the provided database schema.<|im_end|>
<|im_start|>user
Database: {row['db_id']}

{schema_text}

Question: {row['question']}{evidence_text}<|im_end|>
<|im_start|>assistant
{row['SQL']}<|im_end|>"""
        
        return {
            'text': prompt,
            'db_id': row['db_id'],
            'question': row['question'],
            'SQL': row['SQL']
        }
    
    def create_training_dataset(self, train_data, output_path):
        """
        Create formatted training data
        """
        formatted = []
        
        for row in train_data:
            schema = self.schema_info[row['db_id']]
            example = self.format_example(row, schema)
            formatted.append(example)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in formatted:
                f.write(json.dumps(ex) + '\n')
        
        print(f"Created {len(formatted)} examples in {output_path}")
        return formatted


# Usage
if __name__ == "__main__":
    # Load data
    with open('data/raw/train/train.json', 'r') as f:
        train_data = json.load(f)
    
    with open('data/raw/train/train_tables.json', 'r') as f:
        train_schema_list = json.load(f)
        train_schema_info = {s['db_id']: s for s in train_schema_list}
    
    # Create formatter
    formatter = BIRDFormatter(train_schema_info, {})
    formatter.create_training_dataset(
        train_data,
        'data/processed/formatted_train.jsonl'
    )
    
    # Also format dev data
    with open('data/raw/dev/dev.json', 'r') as f:
        dev_data = json.load(f)

    with open('data/raw/dev/dev_tables.json', 'r') as f:  # <- Different file!
        dev_schema_list = json.load(f)
        dev_schema_info = {s['db_id']: s for s in dev_schema_list}
    
    with open('data/raw/dev/dev_tables.json', 'r') as f:  # <- Different file!
        dev_schema_list = json.load(f)
        dev_schema_info = {s['db_id']: s for s in dev_schema_list}