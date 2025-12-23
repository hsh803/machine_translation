# This code consists of two parts: 1) training SentencePiece Tokenizer with BPE, 2) tokenize original datasets (Training, Dev, Test)

import os
import sentencepiece as spm
import re
#tokenizes the parallel file into seperate files
# File names
# Extracted files have following format: 'id': 1, 'teo_text': 'Ipu ainapeta nuka adumun abar kotoma akoru.', 'eng_text': 'There are a number of wealth creation programs around agriculture.'}

input_files = [
    'sunbird_dev_extracted.txt',
    'sunbird_test_extracted.txt',
    'sunbird_training_extracted.txt'
]

output_src_files = [
    'sunbird_dev_tokenized.en_XX',
    'sunbird_test_tokenized.en_XX',
    'sunbird_training_tokenized.en_XX'
]

output_tgt_files = [
    'sunbird_dev_tokenized.teo_XX',
    'sunbird_test_tokenized.teo_XX',
    'sunbird_training_tokenized.teo_XX'
]

# Train SentencePiece model if not already trained
model_prefix = 'spm'
model_path = f'{model_prefix}.model'
input_text = 'combined_input.txt'

if not os.path.exists(model_path):
    # Combine input files for training the SentencePiece model
    with open(input_text, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = re.sub(r"(?<!\\)'", '"', line)
                    match = re.search(r'"id":\s*(\d+),.*?"teo_text":\s*"(.*?)".*?"eng_text":\s*"(.*?)"', line)
                    if match:
                        teo_text = match.group(2).strip()
                        eng_text = match.group(3).strip()
                        if teo_text and eng_text:  # Only write non-empty lines
                            outfile.write(f"{teo_text}\n")
                            outfile.write(f"{eng_text}\n")

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_text,
        model_prefix=model_prefix,
        vocab_size=8000,
        character_coverage=1.0,
        model_type='bpe'
    )

# After training SentencePiece Tokenizer, spm.model and spm.vocab are generated and saved.

# Initialize the SentencePieceProcessor
sp = spm.SentencePieceProcessor(model_file=model_path)

# Function to tokenize parallel data using SentencePiece
def tokenize_parallel_data(input_file, output_src_file, output_tgt_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_src_file, 'w', encoding='utf-8') as src_out, \
         open(output_tgt_file, 'w', encoding='utf-8') as tgt_out:
        for line in infile:
            try:
                line = re.sub(r"(?<!\\)'", '"', line)
                match = re.search(r'"id":\s*(\d+),.*?"teo_text":\s*"(.*?)".*?"eng_text":\s*"(.*?)"', line)
                if match:
                    teo_text = match.group(2).strip()
                    eng_text = match.group(3).strip()
                    if teo_text and eng_text:  # Only process non-empty lines
                        # Tokenize teo_text and eng_text using SentencePiece
                        tokenized_teo_sp = sp.encode(teo_text, out_type=str)
                        tokenized_eng_sp = sp.encode(eng_text, out_type=str)
                        # Write the tokenized data to the output files
                        tgt_out.write(' '.join(tokenized_teo_sp) + '\n')
                        src_out.write(' '.join(tokenized_eng_sp) + '\n')
                else:
                    print(f"No match found in file {input_file}, line: {line}")
            except Exception as e:
                print(f"Error processing line in file {input_file}, line: {line}, error: {e}")

# Iterate through all files and extract data
for input_file, output_src_file, output_tgt_file in zip(input_files, output_src_files, output_tgt_files):
    tokenize_parallel_data(input_file, output_src_file, output_tgt_file)
    print(f"Extracted parallel data from {input_file} to {output_src_file} and {output_tgt_file}")

