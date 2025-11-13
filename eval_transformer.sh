#!/bin/bash
#$ -P cs599dg      # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -l mem_per_core=16G
#$ -l gpus=1

source ~/.bashrc
mkdir -p logs

module load miniconda
conda activate fusion_model

python scripts/eval.py --fusion-model transformer \
    --fusion-ckpt weights/fusion_transformer_zs/transformer_fusion_final.pt \
    --output-dir eval_results/fusion_transformer_zs \
    --test-csv data/triplets_eval.csv