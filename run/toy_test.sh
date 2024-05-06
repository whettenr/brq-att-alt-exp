#!/bin/bash

#SBATCH --job-name=mem_t   # nom du job
#SBATCH --account=dha@v100
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=10:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/mem_test_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp


# summary mixing 
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_summarymixing.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/summary_mix/
    rm -r results/toy/summary_mix/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_summarymixing_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/summary_mix_lg/
    rm -r results/toy/summary_mix_lg/save
done

# brq 
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/brq/
    rm -r results/toy/brq/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/brq_lg/
    rm -r results/toy/brq_lg/save
done

# hyper conformer
for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/hyperconformer.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/hc/
    rm -r results/toy/hc/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/hyperconformer_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 10 \
        --output_folder results/toy/hc_lg/
    rm -r results/toy/hc_lg/save
done

# # fastformer
# for sim_test_time in 10 20 30 40 50 60 70 80 90
# do
#     torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
#         memory_test.py hparams/fastformer.yaml --find_unused_parameters \
#         --seconds_per_batch 100 --train_num_buckets 50 \
#         --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
#         --sim_test_time $sim_test_time --sim_batch_size 2 \
#         --log_interval 10 \
#         --output_folder results/toy/ff/ \
#         --d_model 624 --encoder_module conformer --nhead 16 --transformer_dropout 0.2 --num_encoder_layers 12
#     rm -r results/toy/ff/save
# done


# for sim_test_time in 10 20 30 40 50 60 70 80 90
# do
#     torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
#         memory_test.py hparams/fastformer_lg.yaml --find_unused_parameters \
#         --seconds_per_batch 100 --train_num_buckets 50 \
#         --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
#         --sim_test_time $sim_test_time --sim_batch_size 2 \
#         --log_interval 10 \
#         --output_folder results/toy/ff_lg/ \
#         --d_model 1472 --encoder_module conformer --nhead 32 --transformer_dropout 0.2 --num_encoder_layers 12 
#     rm -r results/toy/ff_lg/save
# done


# for sim_test_time in 10 20 30 40 50 60 70 80 90
# do
#     torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
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
#     torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
#         memory_test.py hparams/hyperbranchformer_lg.yaml --find_unused_parameters \
#         --seconds_per_batch 100 --train_num_buckets 50 \
#         --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 100 \
#         --sim_test_time $sim_test_time --sim_batch_size 10 \
#         --log_interval 10 \
#         --output_folder results/toy/hbf_lg/
#     rm -r results/toy/hbf_lg/save
# done
