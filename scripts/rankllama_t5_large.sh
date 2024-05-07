#!/bin/sh
#SBATCH  --partition gpu
#SBATCH --job-name=large
#SBATCH --mem 120G
#SBATCH --time=250:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_encoder_decoder_t5_large_0.01.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_encoder_decoder_t5_large_0.01.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4  # Request one GPU per task

export HYDRA_FULL_ERROR=1
conda activate xformers

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m lsr.train +experiment=mlm_encoder_decoder_multi_t5_large_rankllama_teacher0.01_resume \
    training_arguments.fp16=True \
    wandb.resume=False 