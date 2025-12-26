# Machine translation
- Neural Machine Translation for low-resource language, Ateso
- NOTE: mBart 50 training codes shared here are slightly adjusted in the initial ones used for our expeirments for practical reasons, such as  in order to correct some incorrections and to clean unused snippets. These adjustments could give improved performance compared to the results of the initial ones.

## Data
- Sentences in English, Ateso, Acholi
- Train(23 947), Dev(496), Test (500)
- Source: https://github.com/SunbirdAI/salt

## Tokenizer
- Fairseq (fairseq==0.10.0)
: SentencePiece Tokenizer (https://github.com/google/sentencepiece)
: Train SententcePiece Tokenizer with BPE (Byte Pair Encoding)
: spm.model, spm.vocal used for tokenize the original input sentences.
- mBart-50-large: mBart-50 SentencePiece Tokenizer

## Models
- Fairseq, Baseline: Trained on English, Ateso paried sentences in epoch 5 and 10 respectively
- mBart-50, Baseline: Trained on English, Ateso paried sentences in epoch 5 and 10 respectively
- mBart-50, fine-tuned: Trained on English, Ateso paried sentences using Swahili language code for Ateso in epoch 5
- mBart-50, fine-tuned: Trained on English, Ateso, Acholi paried sentences using cross-lingual learning in epoch 5 and 10 respectively
- mBart-50, fine-tuned: Trained on English, Ateso, Acholi paried sentences using Swahili language code for Ateso and Acholi as well as cross-lingual learning in epoch 5

## Experiments
- Fairseq
: Train SentencePiece Tokenizer on the train, dev and test datasets with BPE. -> sentencepiece_tokenizer.py  
: Tokenize English, Ateso and Acholi sentences with the trained SentencePiece Tokenizer. ->  sentencepiece_tokenizer.py  
: Train Fairseq model and generate translation for test the trained model. ->  fairseq_baseline.py  
: Evaluate using Sacrebleu. -> fairseq_baseline_translate.sh  

- mBart-50
: Train mBart-50 model. -> mbart50_baseline.py  
: Generate translation for test the baseline model -> mbart50_baseline_translate.py  
: Train mBart-50 model using Swahili language code. -> mbart50_baseline_sw.py  
: Generate translation for test the baseline model using Swahili language code. -> mbart_sw_translate.py  
: Fine-tune mBart-50 model using cross lingual learning with acholi -> mbart50_finetune_cross.py  
: Generate translation for test the fine-tuned model using cross lingual learning. -> mbart_finetune_cross_translate.py  
: Fine-tune mBart-50 model using Swahili language cod and cross lingual learning with acholi -> mbart50_finetune_cross_sw.py  
: Generate translation for test the fine-tuned model using Swahili language code and cross lingual learning. -> mbart_sw_translate.py  

## Results
- SacreBLEU scores for the NMT model translations from Ateso to English
<img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/30750cc8-5372-40f7-9ef8-f65125e40d9a" />


## Highlights in conclusions
- The mBart models show much better performance than the fairseq model.
- we obtained the best performance using cross-lingual training on Acholi for 5 epochs (21.38 SacreBLEU).
- The case using Swahili language code for tokenization in training gave an interesting improvement of the model’s performance, which outperformed the baseline model. The influences of Swahili to Ateso in vocabulary might have contributed to improved translations.
- Another point that was surprising, was the case using both Swahili language code and cross-lingual technique in training (16.14 SacreBLEU). This case underperformed the cases using only Swahili language code (16.41 SacreBLEU) or only cross-lingual technique (21.38 SacreBLEU).
- These results suggest that it can confuse the model to learn new language, Ateso in consistent and coherent contexts when combining Swahili language code for tokenization with cross-lingual transfer in training.
- Based on those results, we demonstrate that using cross-lingual transfer in training, the model promises better performance than using Swahili language code for tokenization.


