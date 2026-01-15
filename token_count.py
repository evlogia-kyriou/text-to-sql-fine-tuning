from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")

# Load one example
import json
with open('data/processed/formatted_train.jsonl', 'r') as f:
    example = json.loads(f.readline())

# Count tokens
tokens = tokenizer.encode(example['text'])
print(f"Token count: {len(tokens)}")
print(f"Max allowed: 512")
print(f"Percentage used: {len(tokens)/512*100:.1f}%")

# Check a few more
with open('data/processed/formatted_train.jsonl', 'r') as f:
    token_counts = []
    for i, line in enumerate(f):
        if i >= 100:  # Check first 100
            break
        ex = json.loads(line)
        token_counts.append(len(tokenizer.encode(ex['text'])))

print(f"\nFirst 100 examples:")
print(f"  Min tokens: {min(token_counts)}")
print(f"  Max tokens: {max(token_counts)}")
print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
print(f"  Over 512: {sum(1 for t in token_counts if t > 512)}")