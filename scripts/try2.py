import ir_datasets
import tqdm
import os

# Load the dataset
dataset = ir_datasets.load("lsr42/msmarco-passage-doct5query")

# Define the output folder and make sure it exists
output_dir = 'output_docs'
os.makedirs(output_dir, exist_ok=True)

# Parameters for output files
max_docs_per_file = 1768365
file_count = 0
doc_count = 0

# Initialize file for writing
output_file = open(os.path.join(output_dir, f'docs_{file_count}.tsv'), 'w', encoding='utf-8')

# Iterate over each document in the dataset
for doc in tqdm.tqdm(dataset.docs_iter(), desc="Loading doc collection from ir_datasets"):
    doc_id = doc.doc_id
    text = doc.text.replace('\n', ' ').replace('\t', ' ')  # Ensure no newlines or tabs

    # Write the doc ID and text to the current file
    output_file.write(f'{doc_id}\t{text}\n')
    doc_count += 1

    # If the current file has reached its limit, start a new file
    if doc_count >= max_docs_per_file:
        output_file.close()
        file_count += 1
        doc_count = 0
        output_file = open(os.path.join(output_dir, f'docs_{file_count}.tsv'), 'w', encoding='utf-8')

# Close the last file if it was being written to
if not output_file.closed:
    output_file.close()

print(f"Documents have been processed and written into {file_count + 1} files.")
