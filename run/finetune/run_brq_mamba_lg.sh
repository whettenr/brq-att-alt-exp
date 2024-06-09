#!/bin/bash

#SBATCH --job-name=brq_mamba_ls_lg   # nom du job
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS)
#SBATCH --output=ssl-mamba-ls%j.log # fichier de sortie (%j = job ID)
#SBATCH --error=ssl-mamba-ls%j.log # fichier d’erreur (%j = job ID)

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0
module load ffmpeg/4.2.2
conda deactivate
conda activate mamba_ssl


cd /gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/brq-att-alt-exp
hub=/gpfsscratch/rech/uul/ujg45iy/brq_mamba_bidirectional/save/CKPT+2024-06-07+17-55-16+00
encoder_dim='474'
num_encoder_layers=24

python finetune/ft_brq.py finetune/ft_brq_mamba.yaml \
    --data_folder /gpfsdswork/dataset/LibriSpeech \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --output_folder /gpfsscratch/rech/uul/ujg45iy/FT/LS/mamba_bidir_fp32/ \
    --test_batch_size 4 \ 
    --num_encoder_layers $num_encoder_layers 

ngram='/gpfsscratch/rech/uul/ujg45iy/ngram/ls/4-gram.arpa'
python finetune/ft_brq.py finetune/ft_brq_mamba.yaml \
    --data_folder LibriSpeech \
    --pt_model_hub $hub \
    --pt_model_output_dim $encoder_dim \
    --output_folder /gpfsscratch/rech/uul/ujg45iy/FT/LS/mamba_bidir_fp32 \
    --test_batch_size 4 \
    --use_language_modelling True \
    --kenlm_model_path $ngram \
    --test_only \ 
    --num_encoder_layers $num_encoder_layers 