#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=zero_shot
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/beir_zero_shot_llama3_instruct_8b.output
#SBATCH --error=/ivi/ilps/personal/tnguyen5/jf/lsr_eval/log/beir_zero_shot_llama3_instruct_8b.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

export HYDRA_FULL_ERROR=1
ANSERINI_PATH=/ivi/ilps/personal/tnguyen5/jf/anserini-lsr
experiment=zero_shot_llama3_instruct_8b

dataset=nfcorpus
if [[ "$dataset" == "msmarco" ]]
then
    EXTRA="/dev"
elif [[ "$dataset" == "fiqa" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "dbpedia-entity" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "fever" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "hotpotqa" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "nfcorpus" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "quora" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "scifact" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "webis-touche2020" ]]
then
    EXTRA="/v2"
else
    EXTRA=""
fi

mkdir -p outputs/$experiment/inference/
mkdir -p outputs/$experiment/inference/doc/
mkdir -p outputs/$experiment/inference/doc/$dataset/
mkdir -p outputs/$experiment/inference/runs/

input_path=beir/${dataset}${EXTRA}
output_file_name=$dataset.tsv
batch_size=16
type='query'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.input_format=ir_datasets \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=10 \
+experiment=$experiment 

input_path=beir/${dataset}${EXTRA}
output_file_name=$dataset/test.jsonl
batch_size=16
type='doc'
python -m lsr.inference_zeroshot \
inference_arguments.input_path=$input_path \
inference_arguments.input_format=ir_datasets \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.in_text_only=True \
inference_arguments.top_k=10 \
+experiment=$experiment

# rm -r outputs/$experiment/index/$dataset/

$ANSERINI_PATH/target/appassembler/bin/IndexCollection \
  -collection JsonVectorCollection \
  -input outputs/$experiment/inference/doc/$dataset \
  -index outputs/$experiment/index/$dataset/ \
  -generator DefaultLuceneDocumentGenerator \
  -threads 1 -impact -pretokenized \

$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index outputs/$experiment/index/$dataset  \
-topics outputs/$experiment/inference/query/$dataset.tsv \
-topicreader TsvString \
-output outputs/$experiment/inference/runs/$dataset.trec \
-impact -pretokenized -hits 10 -parallelism 60

python clean.py outputs/$experiment/inference/runs/$dataset.trec

mkdir outputs/$experiment/inference/results/

ir_measures beir/${dataset}${EXTRA} outputs/$experiment/inference/runs/$dataset.trec.fixed MRR@10 NDCG@10 > outputs/$experiment/inference/results/$dataset

cat outputs/$experiment/inference/results/$dataset