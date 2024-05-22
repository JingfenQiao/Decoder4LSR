#!/bin/bash
#SBATCH --job-name=search_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/search_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine%a.out
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/search_rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine%a.out
#SBATCH --array=1
#SBATCH --partition=cpu

export HYDRA_FULL_ERROR=1
MODLE_NAME=rankllama_mlm_encoder_decoder_multi_t5_base_0.01_affine
ANSERINI_PATH=/ivi/ilps/personal/jqiao/anserini-lsr
INPUT_DIR=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference
OUTPUT_INDEX=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference

mkdir -p "$OUTPUT_INDEX" # Create the OUTPUT_DIR if it doesn't exist
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
-output $OUTPUT_INDEX/msmarco2.trec \
-hits 10 -impact  -pretokenized  -parallelism 60

ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  $OUTPUT_INDEX/msmarco.trec MRR@10 NDCG@10
