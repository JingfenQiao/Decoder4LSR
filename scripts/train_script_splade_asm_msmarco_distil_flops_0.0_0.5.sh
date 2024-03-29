
CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_only_distil_t5_base training_arguments.fp16=True wandb.resume=False \
> log/lsr_test.log 2>&1 &

# MLM

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_only_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_decoder_only_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_decoder_multi_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlm_encoder_decoder_single_t5_base_0.01 training_arguments.fp16=True wandb.resume=False > log/lsr_test.log 2>&1 &

# MLP
CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_only_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_only_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_decoder_only_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_decoder_only_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_decoder_multi_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_decoder_multi_t5_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python -m lsr.train +experiment=mlp_encoder_decoder_single_t5_base training_arguments.fp16=True wandb.resume=False > log/mlp_encoder_decoder_single_t5_base.log 2>&1 &