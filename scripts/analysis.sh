#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=analysis
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/analysis.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/analysis.output
#SBATCH --gres=gpu   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
python -m lsr.inference_analysis \
+experiment=mlm_encoder_decoder_multi_t5_base_0.01