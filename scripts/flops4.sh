#!/bin/sh
#SBATCH -p cpu
#SBATCH --job-name=flops_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine
#SBATCH --mem=40G
#SBATCH --time=5:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/flops_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine%a.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/flops_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine%a.output
#SBATCH --array=1   # Assuming you have 5 files

export HYDRA_FULL_ERROR=1
MODEL_NAME='google/flan-t5-base'
# declare -a FILE_LIST=(raw_split_aa.tsv  raw_split_ab.tsv  raw_split_ac.tsv  raw_split_ad.tsv  raw_split_ae.tsv)  # replace with your actual filenames
# FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"
INPUT_DIR=/ivi/ilps/personal/jqiao/lsr_eval/outputs/rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine/inference

# mkdir -p "$INPUT_DIR/flop_doc"
# mkdir -p "$INPUT_DIR/flop_query"

# python /ivi/ilps/personal/jqiao/lsr_eval/convert_token2id.py \
#     $MODEL_NAME \
#     $INPUT_DIR/doc/$FILE_NAME \
#     $INPUT_DIR/flop_doc/$FILE_NAME

# mkdir -p "$INPUT_DIR/flop_query"
# python /ivi/ilps/personal/jqiao/lsr_eval/convert_token2id.py \
#     $MODEL_NAME \
#     $INPUT_DIR/query/flops.tsv \
#     $INPUT_DIR/flop_query/raw.tsv

python /ivi/ilps/personal/jqiao/lsr_eval/flops.py \
    $MODEL_NAME \
    $INPUT_DIR/flop_doc \
    $INPUT_DIR/flop_query \
    $INPUT_DIR/flops.json