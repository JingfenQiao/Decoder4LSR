#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mlm_decoder_only_opt13
#SBATCH --output=./log/mlm_decoder_only_opt13.output
#SBATCH --error=./log/mlm_decoder_only_opt13.output
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=4


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m lsr.train +experiment=mlm_decoder_only_opt13 \
    training_arguments.fp16=True \
    wandb.resume=False