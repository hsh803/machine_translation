# Machine translation
- Neural Machine Translation for low-resource language, Ateso
- NOTE: Training codes are slightly adjusted from the initial ones used for our expeirments to correct some incorrections and to clean unused snippets in practice. These adjustments could give improved performance compared to the results of the initial ones.

## Data
- Sentences in English, Ateso, Acholi
- Train(23 947), Dev(496), Test (500)
- Source: https://github.com/SunbirdAI/salt

## Tokenizer
1. Fairseq (fairseq==0.10.0)
- SentencePiece Tokenizer (https://github.com/google/sentencepiece)
- Train SententcePiece Tokenizer with BPE (Byte Pair Encoding)
- spm.model, spm.vocal used for tokenze the original input sentences.

2. mBart-50
- mBart-50 SentencePiece Tokenizer

## Models
1. Fairseq (fairseq==0.10.0), Baseline, epoch 5 and 10
2. mBart-50, Baseline, epoch 5: Trained on training dataset containing Ateso and English paried sentences
3. mBart-50, fine-tuned using Swahili language code, epoch 5
4. mBart-50, fine-tuned using cross-lingual learning with Acholi, epoch 5 and 10
5. mBart-50, fine-tuned using Swahili language code and cross-lingual learning with Acholi, epoch 5

## Experimental process
1. Fairseq
1) Train SentencePiece Tokenizer on the train, dev and test datasets with BPE. -> sentencepiece_tokenizer.py
2) Tokenize English, Ateso and Acholi sentences with the trained SentencePiece Tokenizer. ->  sentencepiece_tokenizer.py
3) Train Fairseq model and generate translation for test the trained model. ->  fairseq_baseline.py
4) Evaluate using Sacrebleu. -> fairseq_baseline_translate.sh

2. mBart-50
1) Train mBart-50 model. -> mbart50_baseline.py
2) Generate translation for test the baseline model -> mbart50_baseline_translate.py
3) Train mBart-50 model using Swahili language code. -> mbart50_baseline_sw.py
4) Generate translation for test the baseline model using Swahili language code. -> mbart_sw_translate.py
7) Fine-tune mBart-50 model using cross lingual learning with acholi -> mbart50_finetune_cross.py
8) Generate translation for test the fine-tuned model using cross lingual learning. -> mbart_finetune_cross_translate.py
9) Fine-tune mBart-50 model using Swahili language cod and cross lingual learning with acholi -> mbart50_finetune_cross_sw.py
10) Generate translation for test the fine-tuned model using Swahili language code and cross lingual learning. -> mbart_sw_translate.py

## Results
<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/f1dc130a-0b2e-481f-a7b9-1f601f8a1236" />


