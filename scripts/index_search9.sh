#!/bin/bash
#SBATCH --job-name=search_mlm_encoder_only_t5_base_0.1_0.08
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/search_mlm_encoder_only_t5_base_0.1_0.08%a.out
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/search_mlm_encoder_only_t5_base_0.1_0.08%a.out
#SBATCH --array=1
#SBATCH --partition=cpu


export HYDRA_FULL_ERROR=1
MODLE_NAME=mlm_encoder_only_t5_base_0.1_0.08
ANSERINI_PATH=/ivi/ilps/personal/jqiao/anserini-lsr
INPUT_DIR=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference
OUTPUT_INDEX=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference

# mkdir -p "$OUTPUT_INDEX" # Create the OUTPUT_DIR if it doesn't exist
# echo "start index "$MODLE_NAME
# $ANSERINI_PATH/target/appassembler/bin/IndexCollection \
# -collection JsonSparseVectorCollection \
# -input $INPUT_DIR/doc  \
# -index $OUTPUT_INDEX/index \
# -generator SparseVectorDocumentGenerator \
# -threads 60 -impact -pretokenized -storePositions -storeDocvectors -storeRaw

echo "start search query" $MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index $OUTPUT_INDEX/index  \
-topics $INPUT_DIR/query/raw.tsv \
-topicreader TsvString \
-output $OUTPUT_INDEX/msmarco.trec \
-hits 1000 -impact  -pretokenized  -parallelism 60

ANSERINI_PATH=/ivi/ilps/personal/jqiao/anserini-lsr

ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  $OUTPUT_INDEX/msmarco.trec MRR@10 R@1000 NDCG@10



# RR@10   0.3102
# R@1000  0.9742
# nDCG@10 0.3717