#!/bin/sh
#SBATCH -A placeholder
#SBATCH -p core -n 4 
#SBATCH -M snowy
#SBATCH -t 20:00:00  
#SBATCH -J ateso-baseline
#SBATCH --gres=gpu:1 

# https://mklimasz.github.io/blog/2023/fariseq-101-train-a-model/ 

source ~/.bashrc
conda activate mt24


CUDA_VISIBLE_DEVICES=0

DATA_DIR=sentence_piece_tokenizer/tokenized-target_source/
WORK_DIR=fairseq-model
TRAIN_DIR=$WORK_DIR/checkpoints-fairseq
mkdir -p $TRAIN_DIR
mkdir -p $DATA_DIR

TEXT=/ateso-project/sentence_piece_tokenizer/tokenized-target_source/

echo "preparing data"

# Process vocabulary file
cut -f1 $TEXT/spm.vocab | tail -n +5 | sed "s/$/ 100/g" > $TEXT/dict.txt

# Binarization
fairseq-preprocess \
    --trainpref $TEXT/sunbird_training_tokenized \
    --validpref $TEXT/sunbird_dev_tokenized \
    --testpref $TEXT/sunbird_test_tokenized \
    --destdir $DATA_DIR \
    --srcdict $TEXT/dict.txt \
    --source-lang "teo" \
    --target-lang "en" \
    --bpe sentencepiece \
    --workers 20

echo "Running training"
# dropout changed from .3 to .4
fairseq-train $DATA_DIR/ \
   --arch transformer \
   --share-decoder-input-output-embed \
   --optimizer adam --adam-betas '(0.9, 0.98)' \
   --clip-norm 0.0 \
   --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
   --dropout 0.4 --weight-decay 0.0001 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   --max-tokens 4096 \
   --eval-bleu \
   --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
   --eval-bleu-detok space \
   --eval-bleu-remove-bpe sentencepiece \
   --eval-bleu-print-samples \
   --save-dir $TRAIN_DIR \
   --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \

