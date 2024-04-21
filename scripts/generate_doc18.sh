#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=mlm_encoder_decoder_multi_t5_base_0.01_doc2query
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_encoder_decoder_multi_t5_base_0.01_doc2query%a.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_encoder_decoder_multi_t5_base_0.01_doc2query%a.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu
#SBATCH --exclude=ilps-cn108
# nvidia_rtx_a6000
export HYDRA_FULL_ERROR=1

# Updating the input path to use the selected FILE_NAME
input_path=hfds:lsr42/msmarco-passage-doct5query

output_file_name=$FILE_NAME
batch_size=32
type='doc'
python -m lsr.inference \
inference_arguments.input_format=hfds \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name.tsv \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-1 \
+experiment=mlm_encoder_decoder_multi_t5_base_0.01_doc2query