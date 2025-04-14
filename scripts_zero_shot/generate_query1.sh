#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zs
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/zero_shot_opt_6.7b.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/zero_shot_opt_6.7b.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
experiment=zero_shot_opt_6.7b

input_path=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=32
type='query'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=10000 \
+experiment=$experiment 