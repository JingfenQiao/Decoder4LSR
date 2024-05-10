#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=rankllama_splade_msmarco_distil_flops_0.1_0.08_affine
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/rankllama_splade_msmarco_distil_flops_0.1_0.08_affine%a.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/rankllama_splade_msmarco_distil_flops_0.1_0.08_affine%a.output
#SBATCH --array=1-5   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --exclude=ilps-cn108

export HYDRA_FULL_ERROR=1
declare -a FILE_LIST=("raw_split_aa.tsv"  "raw_split_ab.tsv"  "raw_split_ac.tsv"  "raw_split_ad.tsv"  "raw_split_ae.tsv")  # replace with your actual filenames

FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"

# Updating the input path to use the selected FILE_NAME
input_path="data/msmarco/full_collection/split/$FILE_NAME"

output_file_name=$FILE_NAME
batch_size=64
type='doc'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name.tsv \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=rankllama_splade_msmarco_distil_flops_0.1_0.08