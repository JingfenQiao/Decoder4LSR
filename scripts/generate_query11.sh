#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=mlm_decoder_only_opt13_lora_0.001
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_lora_0.001.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_lora_0.001.output
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=64
type='query'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=mlm_decoder_only_opt13_lora_0.001