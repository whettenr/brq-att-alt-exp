#!/bin/bash

#SBATCH --job-name=summix   # nom du job
#SBATCH --account=dha@v100
#SBATCH -C v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

python -m torch.distributed.run --nproc_per_node=8 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train.py hparams/brq_summarymixing.yaml --find_unused_parameters \
        --seconds_per_batch 200 --train_num_buckets 70 --precision fp16 --nhead 1 --d_model 536