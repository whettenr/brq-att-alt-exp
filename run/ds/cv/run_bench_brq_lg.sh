#!/bin/bash
#SBATCH --job-name=cv_blg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=100:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_lg_cv_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/brq_lg/1000/save/CKPT+2024-02-28+09-05-17+00
num_layers='25'
num_encoder_layers='24'
encoder_dim='848'
output_folder='results/MP3/brq_lg'
language='cy'
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks

DatasetsFolders=("/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language" "/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language")
ConsideredTasks=('CommonVoice' 'CommonVoice')
DownStreams=('LSTM' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --num_encoder_layers $num_encoder_layers --ssl_hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$language/$downstream --data_folder $dataset_folder \
		--language $language
done

language='eu'

DatasetsFolders=("/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language" "/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language")
ConsideredTasks=('CommonVoice' 'CommonVoice')
DownStreams=('LSTM' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --num_encoder_layers $num_encoder_layers --ssl_hub $hub --encoder_dim $encoder_dim --output_folder $output_folder/$task/$language/$downstream --data_folder $dataset_folder \
		--language $language
done
