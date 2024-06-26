#!/bin/bash
#SBATCH --job-name=vc_b   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=72:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_vc_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/brq/1000/save/CKPT+2024-02-01+05-33-06+00
num_layers='13'
encoder_dim='512' # change this to 576
output_folder='results/MP3/brq'
csv_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/MP3S/csv/VoxCeleb1
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks

DatasetsFolders=('/gpfsscratch/rech/nkp/uaj64gk/corpus/voxceleb2' '/gpfsscratch/rech/nkp/uaj64gk/corpus/voxceleb2')
ConsideredTasks=('VoxCeleb1' 'VoxCeleb1')
DownStreams=('ecapa_tdnn' 'Xvectors')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers 
		--ssl_hub $hub 
		--encoder_dim $encoder_dim 
		--output_folder $output_folder/$task/$downstream 
		--data_folder $dataset_folder 
		--debug \
		--csv_location $csv_location \
done


# DatasetsFolders=( '/corpus/LibriSpeech/' '/corpus/LibriSpeech/' '/local_disk/arges/jduret/corpus/voxceleb2' '/users/rwhetten/benchmarks/benchmarks/MP3S/SLURP')
# ConsideredTasks=('LibriSpeech' 'LibriSpeech' 'VoxCeleb1' 'SLURP')
# DownStreams=('contextnet' 'LSTM' 'ecapa_tdnn' 'LSTM_linear')

# for i in "${!ConsideredTasks[@]}"; do
# 	task=${ConsideredTasks[i]}
# 	downstream=${DownStreams[i]}
# 	dataset_folder=${DatasetsFolders[i]}
# 	python /users/rwhetten/benchmarks/benchmarks/MP3S/$task/$downstream/train.py /users/rwhetten/benchmarks/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml --num_layers_ssl $num_layers --ssl_hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$downstream --data_folder $dataset_folder --debug
# done

