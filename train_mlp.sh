#!/bin/bash
#$ -P cs599dg      # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -l mem_per_core=16G
#$ -l gpus=1

source ~/.bashrc
mkdir -p logs

module load miniconda
conda activate fusion_model

python scripts/train.py --fusion-model mlp \
    --checkpoint-dir weights/fusion_mlp_za