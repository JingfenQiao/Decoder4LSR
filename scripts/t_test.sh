#!/bin/bash
#SBATCH --job-name=t_test
#SBATCH --mem=20G
#SBATCH --time=20:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/t_test%a.out
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/t_test%a.out
#SBATCH --array=1
#SBATCH --partition=cpu

export HYDRA_FULL_ERROR=1

msmarco_qrels=/ivi/ilps/personal/jqiao/anserini-lsr/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt
splade_msmarco=/ivi/ilps/personal/jqiao/lsr/outputs/splade_t5_decoder_distil_multi_decoding_multi_gpus_uppercase/inference600k/msmarco.trec

run_qmlp=/ivi/ilps/personal/jqiao/lsr_eval/outputs/qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08/inference/msmarco.trec
python /ivi/ilps/personal/jqiao/lsr_eval/t_test.py $splade_msmarco $run_qmlp $msmarco_qrels


run_qmlm=/ivi/ilps/personal/jqiao/lsr_eval/outputs/qmlp_dmlm_encoder_decoder_multi_t5_base_0.0_0.08/inference/msmarco.trec
python /ivi/ilps/personal/jqiao/lsr_eval/t_test.py $splade_msmarco $run_qmlp $msmarco_qrels

# dl19_qrels=/ivi/ilps/personal/jqiao/anserini-lsr/tools/topics-and-qrels/qrels.dl19-passage.txt
# dl19=/ivi/ilps/personal/jqiao/anserini-lsr/t_test/dl19
# echo "processing dl19" $run2
# python /ivi/ilps/personal/jqiao/lsr/scripts/t_test.py $dl19/run.dl19-distill-splade-max.tsv $dl19/$run2 $dl19_qrels

# dl20_qrels=/ivi/ilps/personal/jqiao/anserini-lsr/tools/topics-and-qrels/qrels.dl20-passage.txt
# dl20=/ivi/ilps/personal/jqiao/anserini-lsr/t_test/dl20
# echo "processing dl20" $run2
# python /ivi/ilps/personal/jqiao/lsr/scripts/t_test.py $dl20/run.dl20-distill-splade-max.tsv $dl20/$run2 $dl20_qrels
