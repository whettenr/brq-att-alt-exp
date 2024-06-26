#!/bin/bash

#SBATCH --job-name=mem_t   # nom du job
#SBATCH --account=nkp@a100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/train_est_%j.log  # log file

module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp


# fastformer
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/fastformer.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/ff_train/ \
    --d_model 616 --encoder_module conformer --nhead 8 --transformer_dropout 0.2 --num_encoder_layers 12
rm -r results/toy/ff_train/save


torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/fastformer_lg.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/ff_lg_train/ \
    --lr 0.00008
rm -r results/toy/ff_lg_train/save


# summary mixing 94.6M
do
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        train.py hparams/brq_summarymixing.yaml --find_unused_parameters \
        --precision bf16 --optimizer_step_limit 1000 \
        --nhead 1 --d_model 536 \
        --output_folder results/toy/summary_mix_train/
    rm -r results/toy/summary_mix_train/save

# 313.4M
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        train.py hparams/brq_summarymixing_lg.yaml --find_unused_parameters \
        --precision bf16 --optimizer_step_limit 1000 \
        --nhead 1 --d_model 848 \
        --output_folder results/toy/summary_mix_lg_train/
    rm -r results/toy/summary_mix_lg_train/save
done

# brq 

torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/brq.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/brq_train/
rm -r results/toy/brq_train/save


torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/brq_lg.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/brq_lg_train/
rm -r results/toy/brq_lg_train/save

# hyper conformer
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/hyperconformer.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/hc_train/
rm -r results/toy/hc_train/save


torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
    train.py hparams/hyperconformer_lg.yaml --find_unused_parameters \
    --precision bf16 --optimizer_step_limit 1000 \
    --output_folder results/toy/hc_lg_train/
rm -r results/toy/hc_lg_train/save
