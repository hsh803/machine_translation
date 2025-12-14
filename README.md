# Machine translation
- Neural Machine Translation for low-resource language, Ateso

## Data
- Paried sentences in English, Ateso, Acholi
- Train(23 947), Dev(496), Test (500)
- Source: https://github.com/SunbirdAI/salt


## Tokenizer
- SentencePiece Tokenizer (https://github.com/google/sentencepiece)
- Train SententcePiece Tokenizer with BPE (Byte Pair Encoding)
- spm.model, spm.vocal used for tokenze the original input sentences.
- tokenizer.py

## Models

