#!/bin/bash

#SBATCH --job-name=f_brlg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=40:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_lg_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/brq_lg/1000/save/CKPT+2024-02-28+09-05-17+00
encoder_dim='848'


python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder /gpfsdswork/dataset/LibriSpeechAsrCorpus \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --num_encoder_layers 24 \
    --output_folder results/ft/brq_lg


python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder /gpfsdswork/dataset/LibriSpeechAsrCorpus \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --num_encoder_layers 24 \
    --output_folder results/ft/brq_lg \
    --test_only --kenlm_model_path /gpfswork/rech/nkp/uaj64gk/bestrqexp/4-gram.arpa.gz
   
