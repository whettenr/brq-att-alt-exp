#!/bin/bash

#SBATCH --job-name=f_hbf   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=30:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/hbf_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa

cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

encoder_dim='936'
attention_type='hypermixing'
encoder_module='branchformer'
output_folder='results/MP3/hbf'
hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/hbf/1000/save/CKPT+2024-02-19+19-48-51+00
data_folder=/gpfsdswork/dataset/LibriSpeechAsrCorpus

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --output_folder $output_folder

python finetune/ft_brq.py finetune/ft_brq.yaml \
    --data_folder $data_folder \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --attention_type $attention_type --encoder_module $encoder_module \
    --output_folder $output_folder \
    --test_only --kenlm_model_path /gpfswork/rech/nkp/uaj64gk/bestrqexp/4-gram.arpa.gz