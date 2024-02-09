#!/bin/bash

#SBATCH --job-name=f_hc   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/hc_%j.log  # log file

module load pytorch-gpu/py3/2.0.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

encoder_dim='512'
attention_type='hypermixing'
encoder_module='conformer'
output_folder='results/ft/hc'
hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/hc/1000/save/CKPT+2024-02-03+16-04-24+00
data_folder=/gpfsdswork/dataset/LibriSpeechAsrCorpus

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --output_folder $output_folder