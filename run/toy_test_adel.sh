#!/bin/bash

#SBATCH --job-name=mem_t   # nom du job
#SBATCH --account=uul@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos=qos_gpu-t4
#SBATCH --exclusive
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=mem_test_%j.log  # log file

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0
module load ffmpeg/4.2.2
conda deactivate
conda activate mamba_ssl

cd /gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/brq-att-alt-exp


# brq 
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_mamba_bidirectional.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/brq/
    rm -r $SCRATCH/results/toy/brq/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_mamba_bidirectional_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/brq_lg/
    rm -r $SCRATCH/results/toy/brq_lg/save
done


# fastformer
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/fastformer.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/ff/ \
        --d_model 616 --encoder_module conformer --nhead 8 --transformer_dropout 0.2 --num_encoder_layers 12 \
        --mask_prob .05
    rm -r $SCRATCH/results/toy/ff/save
done


for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/fastformer_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/ff_lg/ \
        --lr 0.00008
    rm -r $SCRATCH/results/toy/ff_lg/save
done


# summary mixing 94.6M
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_summarymixing.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --nhead 1 --d_model 536 \
        --output_folder $SCRATCH/results/toy/summary_mix/
    rm -r $SCRATCH/results/toy/summary_mix/save
done

# 313.4M
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_summarymixing_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --nhead 1 --d_model 848 \
        --output_folder $SCRATCH/results/toy/summary_mix_lg/
    rm -r $SCRATCH/results/toy/summary_mix_lg/save
done

# brq 
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/brq/
    rm -r $SCRATCH/results/toy/brq/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/brq_lg/
    rm -r $SCRATCH/results/toy/brq_lg/save
done

# hyper conformer
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/hyperconformer.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/hc/
    rm -r $SCRATCH/results/toy/hc/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/hyperconformer_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder $SCRATCH/results/toy/hc_lg/
    rm -r $SCRATCH/results/toy/hc_lg/save
done


# for sim_test_time in 10 20 30 40 50 60 70 80 90
# do
#     torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
#         memory_test.py hparams/hyperbranchformer.yaml --find_unused_parameters \
#         --seconds_per_batch 100 --train_num_buckets 50 \
#         --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
#         --sim_test_time $sim_test_time --sim_batch_size 2 \
#         --log_interval 10 \
#         --output_folder results/toy/hbf/
#     rm -r results/toy/hbf/save
# done



# for sim_test_time in 10 20 30 40 50 60 70 80 90
# do
#     torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
#         memory_test.py hparams/hyperbranchformer_lg.yaml --find_unused_parameters \
#         --seconds_per_batch 100 --train_num_buckets 50 \
#         --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
#         --sim_test_time $sim_test_time --sim_batch_size 10 \
#         --log_interval 10 \
#         --output_folder results/toy/hbf_lg/
#     rm -r results/toy/hbf_lg/save
# done
