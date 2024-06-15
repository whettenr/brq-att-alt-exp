#!/bin/bash

#SBATCH --job-name=mem_t_mamba   # nom du job
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=15:00:00          # temps d'exécution maximum demande (HH:MM:SS)
#SBATCH --output=test-mamba-ls%j.log # fichier de sortie (%j = job ID)
#SBATCH --error=test-mamba-ls%j.log # fichier d’erreur (%j = job ID)

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
        --output_folder /gpfsscratch/rech/uul/ujg45iy/results/toy/brq/
    rm -r results/toy/brq/save
done

for sim_test_time in 10 20 30 40 50 60 70 80 90
do
    torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 \
        memory_test.py hparams/brq_mamba_bidirectional_lg.yaml --find_unused_parameters \
        --seconds_per_batch 100 --train_num_buckets 50 \
        --grad_accumulation_factor 1 --precision fp16 --optimizer_step_limit 1000 \
        --sim_test_time $sim_test_time --sim_batch_size 2 \
        --log_interval 100 \
        --output_folder /gpfsscratch/rech/uul/ujg45iy/results/toy/brq_lg/
    rm -r results/toy/brq_lg/save
done
