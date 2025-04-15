#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mlm_decoder_only_opt3.5
#SBATCH --output=./log/mlm_decoder_only_opt3.5.output
#SBATCH --error=./log/mlm_decoder_only_opt3.5.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0 python -m lsr.train +experiment=mlm_decoder_only_opt3.5 \
    training_arguments.fp16=True wandb.resume=False
