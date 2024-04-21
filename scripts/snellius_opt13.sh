#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=mlm_decoder_only_opt13
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_0.01.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr_eval/log/mlm_decoder_only_opt13_0.01.output
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=4


source activate lsr

echo "Node Information:"
echo "-----------------"
# Print the name of the node
echo "Node Name: $SLURM_NODELIST"
# Print the number of CPUs
echo "Number of CPUs: $SLURM_JOB_CPUS_PER_NODE"
# Print the total memory
echo "Total Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Printing device information..."
nvidia-smi

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m lsr.train +experiment=mlm_decoder_only_opt13_0.01 \
    training_arguments.fp16=True \
    wandb.resume=False