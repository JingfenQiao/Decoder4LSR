#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zero_shot_flan_t5_xxl
#SBATCH --mem=30G
#SBATCH --time=20:00:00
#SBATCH --output=./log/zero_shot_flan_t5_xxl%a.output
#SBATCH --error=./log/zero_shot_flan_t5_xxl%a.output
#SBATCH --array=1-5   # We have 5 files
#SBATCH --gres=gpu:1   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1

# Generate sparse representation for queries
experiment=zero_shot_flan_t5_xxl
input_path=./data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=32
type='not_query'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=1000 \
+experiment=$experiment

# Generate sparse representation for documents
declare -a FILE_LIST=("raw_split_aa"  "raw_split_ab"  "raw_split_ac"  "raw_split_ad"  "raw_split_ae")  # replace with your actual filenames
FILE_NAME="${FILE_LIST[$SLURM_ARRAY_TASK_ID - 1]}"
experiment=zero_shot_flan_t5_xxl
input_path="data/msmarco/full_collection/split/$FILE_NAME"
output_file_name=$FILE_NAME
batch_size=16
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

# Build index based on generated documents sparse representation
MODLE_NAME=zero_shot_flan_t5_xxl
ANSERINI_PATH=./anserini-lsr
INPUT_DIR=./outputs/$MODLE_NAME/inference
OUTPUT_INDEX=./outputs/$MODLE_NAME/inference

mkdir -p "$OUTPUT_INDEX" # Create the OUTPUT_DIR if it doesn't exist
echo "start index "$MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/IndexCollection \
  -collection JsonVectorCollection \
  -input $INPUT_DIR/doc  \
  -index $OUTPUT_INDEX/index \
  -generator DefaultLuceneDocumentGenerator \
  -threads 60 -impact -pretokenized -storePositions -storeDocvectors -storeRaw

# Search the index with the generated queries
echo "start search query" $MODLE_NAME
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
    -index $OUTPUT_INDEX/index  \
    -topics $INPUT_DIR/query/raw.tsv \
    -topicreader TsvString \
    -output $OUTPUT_INDEX/msmarco.trec \
    -hits 1000 -impact  -pretokenized  -parallelism 60

# Evaluate the search results
ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt  $OUTPUT_INDEX/msmarco.trec MRR@10 R@1000 R@100 NDCG@10

