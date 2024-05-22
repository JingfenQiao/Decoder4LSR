#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08.output
#SBATCH --gres=gpu   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/dev_queries/raw.tsv
# output_file_name=raw.tsv
# batch_size=64
# type='query'
# python -m lsr.inference \
# inference_arguments.input_path=$input_path \
# inference_arguments.output_file=$output_file_name \
# inference_arguments.type=$type \
# inference_arguments.batch_size=$batch_size \
# inference_arguments.scale_factor=100 \
# inference_arguments.top_k=-1 \
# +experiment=qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08

output_file_name=flops.tsv
batch_size=64
type='query'
python -m lsr.inference2 \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08