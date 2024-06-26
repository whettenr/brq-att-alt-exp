#!/bin/bash
#SBATCH --job-name=ie_b_mamba   # nom du job
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=brq_ie_%j.log  # log file
#SBATCH --array=0-4%1

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0
module load ffmpeg/4.2.2
conda deactivate
conda activate mamba_ssl

cd /gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/brq-att-alt-exp
hub=/gpfsscratch/rech/uul/ujg45iy/brq_mamba_bidirectional/save/CKPT+2024-06-07+17-55-16+00
num_layers='13'
encoder_dim='474'
output_folder='/gpfsscratch/rech/uul/ujg45iy/FT/IE/brq_mamba_bidir'
benchmark_location=/gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/benchmarks

task='IEMOCAP'
downstream='ecapa_tdnn'

for i in {1..10}
do
echo "on speaker $i"
python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /gpfsscratch/rech/uul/ujg45iy/IEMOCAP/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" \
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
	--ssl_hub "$hub" \
	--encoder_dim "$encoder_dim" \
	--output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--test_spk_id=$i
done

