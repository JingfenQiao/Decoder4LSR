#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=t5_large
#SBATCH --output=/home/jqiao/project/lsr_eval/log/t5_large1.output
#SBATCH --error=/home/jqiao/project/lsr_eval/log/t5_large1.output
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

CUDA_VISIBLE_DEVICES=0 nohup python -m lsr.train +experiment=mlm_encoder_decoder_multi_t5_large_rankllama_teacher0.01 \
    training_arguments.fp16=True \
    wandb.resume=False