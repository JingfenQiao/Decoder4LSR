#!/bin/bash
#SBATCH --job-name=re_ranking_preprocess
#SBATCH --mem=30G
#SBATCH --time=8:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/re_ranking_preprocess%a.out
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/re_ranking_preprocess%a.out
#SBATCH --array=1
#SBATCH --partition=cpu

CUDA_VISIBLE_DEVICES=0 python re_ranking_preprocess.py \
    --queries_path /var/scratch/jkang/project/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/raw.tsv \
    --collections_path /var/scratch/jkang/project/lsr_eval/data/msmarco/full_collection/raw.tsv \
    --passage_output_path /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_passage.json \
    --query_output_path  /var/scratch/jkang/project/lsr_eval/data/msmarco/re_ranking/claude_generated_query.json \
    > test.log 2>&1 &