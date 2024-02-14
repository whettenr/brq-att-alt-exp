#!/bin/bash
#SBATCH --job-name=ls_hclg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=75:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/hc_lg_ls_%j.log  # log file

module load pytorch-gpu/py3/2.0.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/hc_lg/1000/save/CKPT+2024-02-07+02-50-50+00
num_layers='25'
num_encoder_layers='24'
encoder_dim='768' # change to ???
attention_type='hypermixing'
encoder_module='conformer'
output_folder='results/MP3/hc_lg'
csv_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/MP3S
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks

DatasetsFolders=('/corpus/LibriSpeech/' '/corpus/LibriSpeech/')
ConsideredTasks=('LibriSpeech' 'LibriSpeech')
DownStreams=('contextnet' 'LSTM')

for i in "${!ConsideredTasks[@]}"; do
	task=${ConsideredTasks[i]}
	downstream=${DownStreams[i]}
	dataset_folder=${DatasetsFolders[i]}
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --num_encoder_layers $num_encoder_layers --ssl_hub $hub --encoder_dim $encoder_dim \
		--output_folder $output_folder/$task/$downstream --data_folder $dataset_folder --debug \
		--attention_type $attention_type --encoder_module $encoder_module \
		--csv_location $csv_location
	
	python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
		--num_layers_ssl $num_layers --num_encoder_layers $num_encoder_layers --ssl_hub $hub --encoder_dim $encoder_dim \
		--output_folder $output_folder/$task/$downstream --data_folder $dataset_folder --test_only --language_modelling True \
		--attention_type $attention_type --encoder_module $encoder_module \
		--csv_location $csv_location
done