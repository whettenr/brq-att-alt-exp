#!/bin/bash

#SBATCH --job-name=f_br   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=30:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_%j.log  # log file

module load pytorch-gpu/py3/2.0.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp


python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder /gpfsdswork/dataset/LibriSpeechAsrCorpus \
    --pt_model_hub /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/brq/1000/save/CKPT+2024-02-01+05-33-06+00 \
    --output_folder results/ft/brq


python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder /gpfsdswork/dataset/LibriSpeechAsrCorpus \
    --pt_model_hub /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/brq/1000/save/CKPT+2024-02-01+05-33-06+00 \
    --output_folder results/ft/brq \
    --test_only --kenlm_model_path /gpfswork/rech/nkp/uaj64gk/bestrqexp/4-gram.arpa.gz