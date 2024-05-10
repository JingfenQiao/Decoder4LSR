#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=splade_affine_dl19
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/splade_affine_querywise_dl19.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/splade_affine_querywise_dl19.output
#SBATCH --gres=gpu   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
MODLE_NAME=splade_affine
ANSERINI_PATH=/ivi/ilps/personal/tnguyen5/jf/anserini-lsr
input_path=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/raw.tsv
output_path=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/outputs/$MODLE_NAME/inference
output_file_name=raw_dl19.tsv
batch_size=64
type='query'
mkdir -p "$output_path/results"

echo "start to generate query"
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_dir=$output_path/query \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=rankllama_splade_msmarco_distil_flops_0.1_0.08

echo "start search"
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index $output_path/index  \
-topics $output_path/query/$output_file_name \
-topicreader TsvString \
-output $output_path/results/$output_file_name.result \
-impact -pretokenized -hits 10 -parallelism 60

ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.dl19-passage.txt $output_path/results/$output_file_name.result MRR@10 NDCG@10

