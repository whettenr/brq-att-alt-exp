#!/bin/bash
#SBATCH --job-name=s_brq_lg_mamba
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=s_mamba_%j.log  # log file
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
output_folder='/gpfsscratch/rech/uul/ujg45iy/FT/SLURP/brq_mamba_bidir_lg'
benchmark_location=/gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/benchmarks

DatasetsFolders=('/gpfsscratch/rech/uul/ujg45iy/SLURP/slurp/dataset/slurp' '/gpfsscratch/rech/uul/ujg45iy/SLURP/slurp/dataset/slurp')
ConsideredTasks=('SLURP' 'SLURP')
DownStreams=('LSTM_linear' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers \
		--num_encoder_layers $num_encoder_layers \
		--ssl_hub $hub \
		--encoder_dim $encoder_dim \
		--output_folder $output_folder/$task/$downstream \
		--data_folder $dataset_folder 
done

