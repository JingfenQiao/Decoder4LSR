#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zs
#SBATCH --mem=30G
#SBATCH --time=20:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/search_zero_shot_flan_t5_base%a.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/search_zero_shot_flan_t5_base%a.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:1   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
experiment=zero_shot_flan_t5_base

input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=32
type='not_query'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=1000 \
+experiment=$experiment 


# declare -a FILE_LIST=("raw_split_aa"  "raw_split_ab"  "raw_split_ac"  "raw_split_ad"  "raw_split_ae")  # replace with your actual filenames
# FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"
# experiment=zero_shot_flan_t5_base
# input_path="data/msmarco/full_collection/split/$FILE_NAME"
# output_file_name=$FILE_NAME
# batch_size=16
# type='doc'
# python -m lsr.inference_zeroshot \
# inference_arguments.input_path=$input_path \
# inference_arguments.output_file=$output_file_name \
# inference_arguments.type=$type \
# inference_arguments.batch_size=$batch_size \
# inference_arguments.scale_factor=100 \
# inference_arguments.in_text_only=True \
# inference_arguments.top_k=128 \
# +experiment=$experiment


