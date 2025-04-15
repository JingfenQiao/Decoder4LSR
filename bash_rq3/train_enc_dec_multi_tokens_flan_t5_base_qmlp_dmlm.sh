#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=qmlp_dmlm_encoder_decoder_multi_t5_base
#SBATCH --output=./log/qmlp_dmlm_encoder_decoder_multi_t5_base.output
#SBATCH --error=./log/qmlp_dmlm_encoder_decoder_multi_t5_base.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1


export HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python -m lsr.train +experiment=qmlp_dmlm_encoder_decoder_multi_t5_base \
    training_arguments.fp16=True wandb.resume=False
