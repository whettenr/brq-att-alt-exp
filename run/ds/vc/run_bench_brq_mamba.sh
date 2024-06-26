#!/bin/bash
#SBATCH --job-name=VC_brq_mamba   # nom du job
#SBATCH --constraint='GPURAM_Min_32GB'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=72:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=brq_VC_%j.log  # log file
#SBATCH --mail-type=BEGIN,END,FAIL

source /etc/profile.d/conda.sh
conda activate mamba_speech

cd /users/amoumen/machine_learning/research/mamba_speech/brq-att-alt-exp
hub=/users/amoumen/machine_learning/research/mamba_speech/saved_models/brq_mamba_bidirectional
num_layers=13
encoder_dim=474
output_folder='/users/amoumen/machine_learning/research/mamba_speech/outputs/FT/VoxCeleb1/brq_mamba_bidir'
benchmark_location=/users/amoumen/machine_learning/research/mamba_speech/benchmarks

DatasetsFolders=("/local_disk/arges/jduret/corpus/voxceleb2/" "/local_disk/arges/jduret/corpus/voxceleb2/")
ConsideredTasks=('VoxCeleb1' 'VoxCeleb1')
DownStreams=('ecapa_tdnn' 'Xvectors')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers \
		--ssl_hub $hub \
		--encoder_dim $encoder_dim \
		--output_folder $output_folder/$task/$downstream \
		--data_folder $dataset_folder 
done

