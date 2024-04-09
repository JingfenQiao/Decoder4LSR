#!/bin/sh
#SBATCH  --partition gpu
#SBATCH --job-name=opt67
#SBATCH --mem 120G
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/opt67.001.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/opt67.001.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4  # Request one GPU per task


export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m lsr.train +experiment=mlm_decoder_only_opt67_lora_0.001 \
    training_arguments.fp16=True \
    wandb.resume=False