#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=splade
#SBATCH --output=./log/mlm_encoder_only_distll_splade_max.output
#SBATCH --error=./log/mlm_encoder_only_distll_splade_max.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=mlm_encoder_only_distll_splade_max \
    training_arguments.fp16=True \
    wandb.resume=False

