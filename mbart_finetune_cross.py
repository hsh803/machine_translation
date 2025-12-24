import os
import json
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

train_file = "/ateso-project/sentence_piece_tokenizer/data/concat/sunbird_training_concat.txt"
validation_file = "/ateso-project/sentence_piece_tokenizer/data/concat/sunbird_dev_concat.txt"
test_file = "/ateso-project/sentence_piece_tokenizer/data/concat/sunbird_test_concat.txt"

# Define Hyperparameters and Training Configurations
learning_rate = 3e-5
batch_size = 4
num_train_epochs = 5   # change to 10 for epoch 10
fp16 = True

# Load the mBART Model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

# Load the tokenizer
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
if "teo_UG" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["teo_UG"] = len(tokenizer.lang_code_to_id)
    tokenizer.id_to_lang_code[tokenizer.lang_code_to_id["teo_UG"]] = "teo_UG"
if "ach_UG" not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id["ach_UG"] = len(tokenizer.lang_code_to_id)
    tokenizer.id_to_lang_code[tokenizer.lang_code_to_id["ach_UG"]] = "ach_UG"

model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]  # Adjust this to match the target language
    
def load_and_tokenize_dataset(file_path, tokenizer):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line)  # Use eval because the dataset isn't strictly JSON
            target_text = data['eng']  # English (target language)
            tgt_lang = "en_XX"

            # Check for Ateso or Acholi source text and set source text and language code accordingly
            if "teo" in data:
                source_text = data['teo']  # Ateso source text
                src_lang = "teo_UG"
            elif "ach" in data:
                source_text = data['ach']  # Acholi source text
                src_lang = "ach_UG"
            else:
                # Skip the entry if neither Ateso nor Acholi source text is present
                continue
                
            # Set tokenizer language codes
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang

            # Tokenize source and target text
            source_ids = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]
            
            target_text_with_lang = f"<en_XX> {target_text}"
            target_ids = tokenizer(target_text_with_lang, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]

            # Append the source-target pair to the dataset
            dataset.append({
                'input_ids': source_ids,
                'labels': target_ids,
                'source_text': source_text,
                'target_text': target_text,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang
            })

    return Dataset.from_list(dataset)


# Load the datasets with tokenization
train_dataset = load_and_tokenize_dataset(train_file, tokenizer)
validation_dataset = load_and_tokenize_dataset(validation_file, tokenizer)
test_dataset = load_and_tokenize_dataset(test_file, tokenizer)

# Define Data Collator
# DataCollatorForSeq2Seq will pad inputs dynamically to the longest sequence in the batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./cross_learning_5_dir",   # Change to "./cross_learning_10_dir" for epoch 10 training
    eval_strategy="epoch",  # Updated based on the deprecation warning
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    fp16=fp16,  # Enable mixed precision training if supported
    report_to="none",  # Disable W&B integration by using --report_to="none"
    run_name="mbart_finetuning_run",  # Set a descriptive run name here to avoid W&B warning
    load_best_model_at_end=True,  # Load the best model found during training
    metric_for_best_model="eval_loss",  # Use validation loss as the metric for determining the best model
    greater_is_better=False,  # Lower loss is better
    save_strategy="epoch",
    save_steps=500,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

print("training has started")
# Create the Trainer Object
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the Model
trainer.train()


