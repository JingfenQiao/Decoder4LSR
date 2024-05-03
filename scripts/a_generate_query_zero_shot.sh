#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zero_shot
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/zero_shot.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/zero_shot.output
#SBATCH --gres=gpu   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=64
type='query'
python -m lsr.inference_zeroshot2 \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
+experiment=a_zero_shot