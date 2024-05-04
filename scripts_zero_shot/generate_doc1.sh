#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zs
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/zero_shot_opt_6.7b%a.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/zero_shot_opt_6.7b%a.output
#SBATCH --array=1-5%4   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --exclude=ilps-cn108

export HYDRA_FULL_ERROR=1
declare -a FILE_LIST=("raw_split_aa.tsv"  "raw_split_ab.tsv"  "raw_split_ac.tsv"  "raw_split_ad.tsv"  "raw_split_ae.tsv")  # replace with your actual filenames

FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"
experiment=zero_shot_opt_6.7b

# Updating the input path to use the selected FILE_NAME
input_path="data/msmarco/full_collection/split/$FILE_NAME"

output_file_name=$FILE_NAME
batch_size=8
type='doc'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=128 \
+experiment=$experiment