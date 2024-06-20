#!/bin/bash
#SBATCH --job-name=cv_brq_mamba   # nom du job
#SBATCH --account=uul@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=brq_cv_%j.log  # log file
#SBATCH --array=0-2%1

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0
module load ffmpeg/4.2.2
conda deactivate
conda activate mamba_ssl

cd /gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/brq-att-alt-exp
hub=/gpfsscratch/rech/uul/ujg45iy/brq_mamba_bidirectional/save/CKPT+2024-06-07+17-55-16+00
num_layers=13
encoder_dim=474
output_folder='/gpfsscratch/rech/uul/ujg45iy/FT/CV/brq_mamba_bidir'
benchmark_location=/gpfswork/rech/uul/ujg45iy/projects/mamba_ssl/benchmarks
language='cy'

DatasetsFolders=("/gpfsscratch/rech/uul/ujg45iy/cv/cv-corpus-11.0-2022-09-21/$language" "/gpfsscratch/rech/uul/ujg45iy/cv/cv-corpus-11.0-2022-09-21/$language")
ConsideredTasks=('CommonVoice' 'CommonVoice')
DownStreams=('LSTM' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers \
        --ssl_hub $hub \
        --encoder_dim $encoder_dim \
        --output_folder $output_folder/$task/$language/$downstream \
        --data_folder $dataset_folder \
		--language $language 
done


language='eu'
ConsideredTasks=('CommonVoice' 'CommonVoice')
DownStreams=('LSTM' 'linear')
DatasetsFolders=("/gpfsscratch/rech/uul/ujg45iy/cv/cv-corpus-11.0-2022-09-21/$language" "/gpfsscratch/rech/uul/ujg45iy/cv/cv-corpus-11.0-2022-09-21/$language")


for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --ssl_hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$language/$downstream --data_folder $dataset_folder \
		--language $language 
done

