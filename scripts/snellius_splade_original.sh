#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=splade
#SBATCH --output=/home/span/lsr_eval/log/splade.output
#SBATCH --error=/home/span/lsr_eval/log/splade.output
#SBATCH --time=120:00:00
#SBATCH --gpus=1

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

CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=rankllama_splade_msmarco_distil_flops_0.1_0.08 \
    training_arguments.fp16=True \
    wandb.resume=False