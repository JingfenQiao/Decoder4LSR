#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=nq
#SBATCH --mem=30G
#SBATCH --time=100:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/nq.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/nq.output
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

export HYDRA_FULL_ERROR=1
experiment=mlm_encoder_decoder_multi_t5_base_0.01
input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/nq/contriever_retrieved.jsonl
output_file_name=contriever_retrieved.jsonl
batch_size=128
type='doc'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.input_format=jsonl \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-100  \
+experiment=$experiment