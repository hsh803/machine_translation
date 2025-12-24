import os
import json
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

train_file = "/ateso-project/sentence_piece_tokenizer/data/sunbird_training_extracted.txt"
validation_file = "/ateso-project/sentence_piece_tokenizer/data/sunbird_dev_extracted.txt"
test_file = "/ateso-project/sentence_piece_tokenizer/data/sunbird_test_extracted.txt"

# Define Hyperparameters and Training Configurations
learning_rate = 3e-5
batch_size = 4
num_train_epochs = 5
fp16 = True

# Load the mBART Model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

# Load the tokenizer
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
tokenizer.src_lang = "sw_KE"

model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]  # Adjust this to match the target language

# Load the datasets with tokenization
train_dataset = load_and_tokenize_dataset(train_file, tokenizer)
validation_dataset = load_and_tokenize_dataset(validation_file, tokenizer)
test_dataset = load_and_tokenize_dataset(test_file, tokenizer)

def load_and_tokenize_dataset(file_path, tokenizer):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line)  # Use eval because the dataset isn't strictly JSON
            source_text = data['teo_text']  # Ateso (source language)
            target_text = data['eng_text']  # English (target language)

            # Tokenize source and target texts
            source_ids = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]
            
            # Target: Add <en_XX> manually
            target_text_with_lang = f"<en_XX> {target_text}"
            target_ids = tokenizer(target_text_with_lang, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids[0]

            dataset.append({
                'input_ids': source_ids,
                'labels': target_ids,
                'source': source_text,
                'target': target_text
            })

    return Dataset.from_list(dataset)


# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./swahili_output5_dir",
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

