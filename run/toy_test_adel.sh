#!/bin/bash

#SBATCH --job-name=mem_t   # nom du job
#SBATCH --account=uul@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=2:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=mem_test_%j.log  # log file



cd /users/amoumen/machine_learning/research/mamba_speech/speechbrain
git checkout summary-mixing
pip install -e .
cd /users/amoumen/machine_learning/research/mamba_speech/brq-att-alt-exp

# summary mixing 94.6M
python memory_test_v2.py hparams/brq_summarymixing.yaml --nhead 1 --d_model 536
python memory_test_v2.py hparams/brq_summarymixing_lg.yaml --nhead 1 --d_model 848

cd /users/amoumen/machine_learning/research/mamba_speech/speechbrain
git checkout alt-attention
pip install -e .

cd /users/amoumen/machine_learning/research/mamba_speech/brq-att-alt-exp
# mamba
python memory_test_v2.py hparams/brq_mamba_bidirectional.yaml 
python memory_test_v2.py hparams/brq_mamba_bidirectional_lg.yaml


# fastformer
python memory_test_v2.py hparams/fastformer.yaml --d_model 616 --encoder_module conformer --nhead 8 --transformer_dropout 0.2 --num_encoder_layers 12 --mask_prob .05
python memory_test_v2.py hparams/fastformer_lg.yaml

# brq 
python memory_test_v2.py hparams/brq.yaml
python memory_test_v2.py hparams/brq_lg.yaml

# hyper conformer
python memory_test_v2.py hparams/hyperconformer.yaml
python memory_test_v2.py hparams/hyperconformer_lg.yaml