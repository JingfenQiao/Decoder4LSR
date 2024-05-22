#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=search_dl19_20_mlm_decoder_only_opt13_0.01.output
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/search_dl19_20_mlm_decoder_only_opt13_0.01.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/search_dl19_20_mlm_decoder_only_opt13_0.01.output
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

export variable HYDRA_FULL_ERROR=1
MODLE_NAME=mlm_decoder_only_opt13_0.01
ANSERINI_PATH=/ivi/ilps/personal/jqiao/anserini-lsr
input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/TREC_DL_2019/queries_2019/raw.tsv
output_path=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference
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
+experiment=mlm_decoder_only_opt13_0.01

echo "start search"
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index $output_path/index  \
-topics $output_path/query/$output_file_name \
-topicreader TsvString \
-output $output_path/results/$output_file_name.result \
-impact -pretokenized -hits 10 -parallelism 60

echo "dl19"
ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.dl19-passage.txt $output_path/results/$output_file_name.result MRR@10 NDCG@10


input_path=/ivi/ilps/personal/jqiao/lsr_eval/data/msmarco/TREC_DL_2020/queries_2020/raw.tsv
output_path=/ivi/ilps/personal/jqiao/lsr_eval/outputs/$MODLE_NAME/inference
output_file_name=raw_dl20.tsv
batch_size=64
type='query'

echo "start to generate query"
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_dir=$output_path/query \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=mlm_decoder_only_opt13_0.01

echo "start search"
$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index $output_path/index  \
-topics $output_path/query/$output_file_name \
-topicreader TsvString \
-output $output_path/results/$output_file_name.result \
-impact -pretokenized -hits 10 -parallelism 60

echo "dl20"
ir_measures $ANSERINI_PATH/tools/topics-and-qrels/qrels.dl20-passage.txt $output_path/results/$output_file_name.result MRR@10 NDCG@10

