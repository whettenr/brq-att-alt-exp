#!/bin/bash

#SBATCH --job-name=f_ffm5   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=100:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ff_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

encoder_dim='1472'
attention_type='fastattention'
encoder_module='conformer'
output_folder='results/ft/ff_lg_mask_10'
hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/ff_test/lg_lower_mask_10_wide/save/CKPT+2024-05-16+16-12-10+00
data_folder=/gpfsdswork/dataset/LibriSpeechAsrCorpus

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --output_folder $output_folder \
    --transformer_dropout 0.2 --nhead 32 --num_encoder_layers 12

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --output_folder $output_folder \
    --test_only --kenlm_model_path /gpfswork/rech/nkp/uaj64gk/bestrqexp/4-gram.arpa.gz --use_language_modelling true \
    --transformer_dropout 0.2 --nhead 32 --num_encoder_layers 12

