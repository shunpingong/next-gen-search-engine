from datasets import load_dataset
import hashlib
import base64
import pandas as pd
from pathlib import Path

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


# Create output directory if it doesn't exist
output_dir = Path("decrypted_datasets")
output_dir.mkdir(exist_ok=True)

# Load the encrypted datasets
print("Loading encrypted datasets...")
long_context_dataset = load_dataset("openai/BrowseCompLongContext", split="train")

print("\n" + "=" * 80)
print("DECRYPTING LONG CONTEXT DATASET...")
print("=" * 80)
print(f"Total examples to decrypt: {len(long_context_dataset)}")

decrypted_long_context_data = []
for i, row in enumerate(long_context_dataset):
    try:
        decrypted_row = {
            "problem": decrypt(row["problem"], row["canary"]),
            "answer": decrypt(row["answer"], row["canary"]),
            "urls": decrypt(row["urls"], row["canary"]),
        }
        decrypted_long_context_data.append(decrypted_row)
        
        if (i + 1) % 50 == 0:
            print(f"Decrypted {i + 1}/{len(long_context_dataset)} examples...")
    except Exception as e:
        print(f"Error decrypting row {i}: {e}")

print(f"✓ Successfully decrypted {len(decrypted_long_context_data)} examples from long context dataset")

# Save long context dataset to CSV
long_context_csv_path = output_dir / "browsecomp_openai.csv"
long_context_df = pd.DataFrame(decrypted_long_context_data)
long_context_df.to_csv(long_context_csv_path, index=False, encoding='utf-8')
print(f"✓ Saved to: {long_context_csv_path}")

print(f"\nLong Context Dataset (Decrypted):")
print(f"  - Rows: {len(long_context_df)}")
print(f"  - Columns: {list(long_context_df.columns)}")
print(f"  - File: {long_context_csv_path}")

# Show first example from each dataset
if len(long_context_df) > 0:
    print("\n" + "-" * 80)
    print("Long Context Dataset (First Example):")
    print(f"\nProblem:\n{long_context_df.iloc[0]['problem'][:500]}...")
    print(f"\nAnswer:\n{long_context_df.iloc[0]['answer'][:500]}...")
    print(f"\nURLs:\n{long_context_df.iloc[0]['urls'][:500]}...")


# Print first 10 rows
print(long_context_df.head(10))