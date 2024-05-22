#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ss
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/test.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/test.output
#SBATCH --gres=gpu   # Request one GPU per task


CUDA_VISIBLE_DEVICES=0 python -m lsr.train +experiment=qmlp_dmlm_tripclick_hard_negative_0.0_0.0 training_arguments.fp16=True wandb.resume=False


