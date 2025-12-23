#!/bin/sh
#SBATCH -A placeholder
#SBATCH -p core -n 4 # resource, 4 cpu cores
#SBATCH -M snowy # cluster name
#SBATCH -t 01:00:00
#SBATCH -J ateso-baseline
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate mt24


CUDA_VISIBLE_DEVICES=0

# Translating
fairseq-generate /ateso-project/sentence_piece_tokenizer/tokenized-target_source/ --path fairseq-model/checkpoints-fairseq/checkpoint_best.pt --batch-size 128 --beam 5 --results-path fairseq-translations
