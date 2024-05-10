#!/bin/bash
#SBATCH --job-name=search
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/search_zero_shot_flan_t5_xl%a.out
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/search_zero_shot_flan_t5_xl%a.out
#SBATCH --array=1
#SBATCH --partition=cpu

export HYDRA_FULL_ERROR=1
MODLE_NAME=zero_shot_flan_t5_xl
ANSERINI_PATH=/ivi/ilps/personal/tnguyen5/jf/anserini-lsr
INPUT_DIR=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/outputs/$MODLE_NAME/inference
OUTPUT_INDEX=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/outputs/$MODLE_NAME/inference

mkdir -p "$OUTPUT_INDEX" # Create the OUTPUT_DIR if it doesn't exist
echo "start index "$MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/IndexCollection \
  -collection JsonVectorCollection \
  -input $INPUT_DIR/doc  \
  -index $OUTPUT_INDEX/index \
  -generator DefaultLuceneDocumentGenerator \
  -threads 60 -impact -pretokenized -storePositions -storeDocvectors -storeRaw

echo "start search query" $MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
    -index $OUTPUT_INDEX/index  \
    -topics $INPUT_DIR/query/raw.tsv \
    -topicreader TsvString \
    -output $OUTPUT_INDEX/msmarco.trec \
    -hits 1000 -impact  -pretokenized  -parallelism 60

ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  $OUTPUT_INDEX/msmarco.trec MRR@10 R@1000 R@100 NDCG@10


# ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  msmarco.trec MRR@10 R@1000 R@100 NDCG@10
# RR@10   0.1245
# R@1000  0.7045
# R@100   0.4938
# nDCG@10 0.1547