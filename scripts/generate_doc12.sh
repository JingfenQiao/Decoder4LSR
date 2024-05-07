#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=mlm_decoder_only_opt13_0.01
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_0.01%a.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_0.01%a.output
#SBATCH --array=1-5%3   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000
#SBATCH --exclude=ilps-cn108

export HYDRA_FULL_ERROR=1
declare -a FILE_LIST=("raw_split_aa"  "raw_split_ab"  "raw_split_ac"  "raw_split_ad"  "raw_split_ae")  # replace with your actual filenames

FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"

# Updating the input path to use the selected FILE_NAME
input_path="data/msmarco/full_collection/split/$FILE_NAME"

output_file_name=$FILE_NAME
batch_size=32
type='doc'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name.tsv \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=mlm_decoder_only_opt13_0.01