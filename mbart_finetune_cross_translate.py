import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import sacrebleu

# Load the tokenizer and the model
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
model = MBartForConditionalGeneration.from_pretrained("/ateso-project/mbart-scripts/cross_learning_5_dir/checkpoint-11974")   # change to cross_learning_10_dir and best model checkpoint directory for epoch 10 model. Best model checkpoint is given under "best_model_checkpoint" in trainer_state.json

# Adding the language code manually if it's not already available
if "teo_UG" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["teo_UG"] = len(tokenizer.lang_code_to_id)
    tokenizer.id_to_lang_code[tokenizer.lang_code_to_id["teo_UG"]] = "teo_UG"

tokenizer.src_lang = "teo_UG"   # Source: Ateso-only test set
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]     # Target: English

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test dataset
def load_test_sentences(file_path):
    test_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line)  # Convert line (str) to dictionary
            source_text = data['teo_text']  # Ateso source sentence
            target_text = data['eng_text']  # English reference translation
            test_sentences.append((source_text, target_text))
    return test_sentences

test_file = "/proj/uppmax2024-2-13/ateso-project/sentence_piece_tokenizer/data/sunbird_test_extracted.txt"  # Update with your actual test file path
test_sentences = load_test_sentences(test_file)

# File to write the output
output_file_path = "cross10full.txt"   # "cross5.txt" for epoch 5 training
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Translate the test sentences
    generated_translations = []
    reference_translations = []

    for source_text, reference_text in test_sentences:        
        inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = inputs.to(device)  # Move inputs to the same device as the model

        # Generate the translation with a limit on the number of tokens
        translated_tokens = model.generate(
            **inputs,
            max_length=50,  # Set the maximum number of tokens for the generated translation
            num_beams=5,    # Optional: Use beam search to improve the quality of the translation
            early_stopping=True  # Stop early if all beams reach the end
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # Append translations for BLEU calculation
        generated_translations.append(translated_text)
        reference_translations.append(reference_text)

        # Write the original, reference, and translated texts to the output file
        output_file.write(f"Source (Ateso): {source_text}\n")
        output_file.write(f"Reference (English): {reference_text}\n")
        output_file.write(f"Generated Translation: {translated_text}\n\n")

    # Calculate BLEU score
    print("Calculating BLEU score...")
    bleu = sacrebleu.corpus_bleu(generated_translations, [reference_translations])
    output_file.write(f"\nBLEU score: {bleu.score:.2f}\n")
    print(f"BLEU score: {bleu.score:.2f}")

