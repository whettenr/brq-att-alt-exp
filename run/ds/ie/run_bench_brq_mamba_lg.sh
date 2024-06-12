#!/bin/bash
#SBATCH --job-name=ie_brq_lg_mamba   # nom du job
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=brq_lg_ie_%j.log  # log file
#SBATCH --array=0-4%1

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0
module load ffmpeg/4.2.2
conda deactivate
conda activate mamba_ssl

cd /gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/brq-att-alt-exp
hub=/gpfsscratch/rech/uul/ujg45iy/brq_mamba_bidirectional_lg/save/CKPT+2024-06-09+18-01-16+00
num_layers=25
num_encoder_layers=24
encoder_dim=678
output_folder='/gpfsscratch/rech/uul/ujg45iy/FT/IE/brq_mamba_bidir_lg'
benchmark_location=/gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/benchmarks

task='IEMOCAP'
downstream='ecapa_tdnn'

for i in {1..10}
do
echo "on speaker $i"
python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /gpfsscratch/rech/uul/ujg45iy/IEMOCAP/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" \
	--num_encoder_layers "$num_encoder_layers" \
	--ssl_hub "$hub" \
	--encoder_dim "$encoder_dim" \
	--output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--test_spk_id=$i
done

downstream='linear'

for i in {1..10}
do
echo "on speaker $i"
python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /gpfsscratch/rech/uul/ujg45iy/IEMOCAP/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" \
	--num_encoder_layers "$num_encoder_layers" \
	--ssl_hub "$hub" \
	--encoder_dim "$encoder_dim" \
	--output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--test_spk_id=$i
done

