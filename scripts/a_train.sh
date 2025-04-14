#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ss
#SBATCH --mem=90G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/splade_msmarco_distil_flops_0.1_0.08.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/splade_msmarco_distil_flops_0.1_0.08.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request one GPU per task

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lsr.train +experiment=splade_msmarco_distil_flops_0.1_0.08 training_arguments.fp16=True wandb.resume=False