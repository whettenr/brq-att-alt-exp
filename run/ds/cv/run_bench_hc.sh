#!/bin/bash
#SBATCH --job-name=cv_hc   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=35:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/hc_cv_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/hc/1000/save/CKPT+2024-02-18+11-59-20+00
num_layers='13'
encoder_dim='672'
attention_type='hypermixing'
encoder_module='conformer'
output_folder='results/MP3/hc'
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks
language='cy'

DatasetsFolders=("/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language" "/gpfsscratch/rech/nkp/uaj64gk/corpus/cv-corpus-11.0-2022-09-21/$language")
ConsideredTasks=('CommonVoice' 'CommonVoice')
DownStreams=('LSTM' 'linear')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --ssl_hub $hub --encoder_dim $encoder_dim --attention_type $attention_type --encoder_module $encoder_module --output_folder $output_folder/$task/$language/$downstream --data_folder $dataset_folder \
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
		--num_layers_ssl $num_layers --ssl_hub $hub --encoder_dim $encoder_dim --attention_type $attention_type --encoder_module $encoder_module --output_folder $output_folder/$task/$language/$downstream --data_folder $dataset_folder \
		--language $language
done

