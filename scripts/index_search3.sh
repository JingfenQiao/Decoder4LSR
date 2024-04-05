#!/bin/bash
#SBATCH --job-name=search_mlp_encoder_decoder_single_t5_base
#SBATCH --mem=30G
#SBATCH --time=8:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/search_mlp_encoder_decoder_single_t5_base%a.out
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/search_mlp_encoder_decoder_single_t5_base%a.out
#SBATCH --array=1
#SBATCH --partition=cpu

export HYDRA_FULL_ERROR=1
MODLE_NAME=mlp_encoder_decoder_single_t5_base
ANSERINI_PATH=/ivi/ilps/personal/jqiao/anserini-lsr
INPUT_DIR=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference
OUTPUT_INDEX=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference

# mkdir -p "$OUTPUT_INDEX" # Create the OUTPUT_DIR if it doesn't exist
echo "start index "$MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/IndexCollection \
-collection JsonSparseVectorCollection \
-input $INPUT_DIR/doc  \
-index $OUTPUT_INDEX/index \
-generator SparseVectorDocumentGenerator \
-threads 60 -impact -pretokenized -storePositions -storeDocvectors -storeRaw

echo "start search query" $MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index $OUTPUT_INDEX/index  \
-topics $INPUT_DIR/query/raw.tsv \
-topicreader TsvString \
-output $OUTPUT_INDEX/msmarco.trec \
-hits 1000 -impact  -pretokenized  -parallelism 60

ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  $OUTPUT_INDEX/msmarco.trec MRR@10 R@1000 NDCG@10