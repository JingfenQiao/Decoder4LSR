#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=splade
#SBATCH --mem=30G
#SBATCH --time=5:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/splade%a.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/splade%a.output
#SBATCH --array=1   # Assuming you have 5 files
#SBATCH --gres=gpu

export HYDRA_FULL_ERROR=1
MODEL_NAME='distilbert-base-uncased'
declare -a FILE_LIST=(raw_split_aa.tsv  raw_split_ab.tsv  raw_split_ac.tsv  raw_split_ad.tsv  raw_split_ae.tsv)  # replace with your actual filenames
FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"
INPUT_DIR=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/outputs/rankllama_splade_msmarco_distil_flops_0.1_0.08/inference

# mkdir -p "$INPUT_DIR/flop_doc"
# mkdir -p "$INPUT_DIR/flop_query"

# python /ivi/ilps/personal/tnguyen5/jf/lsr_eval/convert_token2id.py \
#     $MODEL_NAME \
#     $INPUT_DIR/doc/$FILE_NAME \
#     $INPUT_DIR/flop_doc/$FILE_NAME

# mkdir -p "$INPUT_DIR/flop_query"
# python /ivi/ilps/personal/tnguyen5/jf/lsr_eval/convert_token2id.py \
#     $MODEL_NAME \
#     $INPUT_DIR/query/flops.tsv \
#     $INPUT_DIR/flop_query/raw.tsv

MODEL_NAME='distilbert-base-uncased'
INPUT_DIR=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/outputs/rankllama_splade_msmarco_distil_flops_0.1_0.08/inference
python /ivi/ilps/personal/tnguyen5/jf/lsr_eval/flops.py \
    $MODEL_NAME \
    $INPUT_DIR/flop_doc \
    $INPUT_DIR/flop_query \
    $INPUT_DIR/flops.json