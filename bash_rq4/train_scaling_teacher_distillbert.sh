#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=splade
#SBATCH --output=./log/rankllama_splade_msmarco_distilbert.output
#SBATCH --error=./log/rankllama_splade_msmarco_distilbert.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=rankllama_splade_msmarco_distilbert \
    training_arguments.fp16=True \
    wandb.resume=False