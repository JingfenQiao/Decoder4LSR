#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mlm_encoder_only_t5_base
#SBATCH --output=./log/mlm_encoder_only_t5_base.output
#SBATCH --error=./log/mlm_encoder_only_t5_base.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1


export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python -m lsr.train +experiment=mlm_encoder_only_t5_base \
    training_arguments.fp16=True wandb.resume=False
