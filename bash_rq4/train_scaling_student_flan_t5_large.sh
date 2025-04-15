#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mlm_encoder_decoder_multi_t5_large
#SBATCH --output=./log/mlm_encoder_decoder_multi_t5_large.output
#SBATCH --error=./log/mlm_encoder_decoder_multi_t5_large.output
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=4



export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m lsr.train +experiment=mlm_encoder_decoder_multi_t5_large \
    training_arguments.fp16=True \
    wandb.resume=False
