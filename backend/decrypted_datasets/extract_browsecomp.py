import csv
import random 

def extract_browsecomp(input_file, output_file, num_samples=50):
    # open with a known encoding and ignore/replace bad bytes to avoid
    # UnicodeDecodeError on non‑cp1252 characters
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
        reader = csv.DictReader(infile)
        # normalize headers to lowercase so 'URL', 'Url', etc. all map to 'url'
        data = []
        for row in reader:
            data.append({k.strip().lower(): v for k, v in row.items()})

    # Randomly sample num_samples rows from the data
    sampled_data = random.sample(data, min(num_samples, len(data)))

    filtered = []
    for row in sampled_data:
        filtered.append({
            'problem': row.get('problem', ''),
            'answer': row.get('answer', ''),
            'url': row.get('url', ''),
        })

    # Write the sampled question/answer pairs to the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['problem', 'answer', 'url'])
        writer.writeheader()
        writer.writerows(filtered)

def print_sampled_data(output_file):
    with open(output_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for raw in reader:
            # make sure the keys are lowercase, matches what extract_browsecomp writes
            row = {k.strip().lower(): v for k, v in raw.items()}
            print(f"Q: {row.get('problem', '')}")
            print(f"A: {row.get('answer', '')}\n")

if __name__ == "__main__":
    input_file = 'browsecomp_openai.csv'  # Path to the input CSV file
    output_file = 'sampled_browsecomp.csv'  # Path to the output CSV file
    # extract_browsecomp(input_file, output_file)
    print_sampled_data(output_file)

