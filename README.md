# Machine translation
- Neural Machine Translation for low-resource language, Ateso

## Data
- Paried sentences in English, Ateso, Acholi
- Train(23 947), Dev(496), Test (500)
- Source: https://github.com/SunbirdAI/salt


## Tokenizer
1. Fairseq (fairseq==0.10.0)
- SentencePiece Tokenizer (https://github.com/google/sentencepiece)
- Train SententcePiece Tokenizer with BPE (Byte Pair Encoding)
- spm.model, spm.vocal used for tokenze the original input sentences.
- tokenizer.py

2. mBart-50
- mBart-50 tokenizer

## Models
1. Fairseq (fairseq==0.10.0), Baseline, epoch 5 and 10: fairseq_baseline.py
2. mBart-50, Baseline, epoch 5: mbart50_baseline.py
3. mBart-50, fine-tuned using Swahili language code, epoch 5: mbart50_baseline_sw.py
4. mBart-50, fine-tuned using cross-lingual learning with Acholi, epoch 5 and 10: mbart50_cross.py
5. mBart-50, fine-tuned using Swahili language code and cross-lingual learning with Acholi, epoch 5: mbart50_cross_sw.py
