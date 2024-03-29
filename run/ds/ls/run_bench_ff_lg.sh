#!/bin/bash
#SBATCH --job-name=ls_fflg   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=75:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ff_lg_ls_%j.log  # log file

module load pytorch-gpu/py3/2.1.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp
hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/ff_lg_test/ff_lg_tr_updrop/CKPT+2024-02-17+08-32-23+00
# hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/CKPT+2024-02-16+22-26-35+00
# hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/ff_lg/1000/save/CKPT+2024-02-07+11-51-51+00
num_layers='25'
num_encoder_layers='24'
encoder_dim='1472'
attention_type='fastattention'
encoder_module='transformer'
output_folder='results/MP3/ff_lg_test_updo'
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
		--output_folder $output_folder/$task/$downstream --data_folder $dataset_folder \
		--attention_type $attention_type --encoder_module $encoder_module \
		--csv_location $csv_location
	
	# python $benchmark_location/benchmarks/MP3S/$task/$downstream/train.py $benchmark_location/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
	# 	--num_layers_ssl $num_layers --num_encoder_layers $num_encoder_layers --ssl_hub $hub --encoder_dim $encoder_dim \
	# 	--output_folder $output_folder/$task/$downstream --data_folder $dataset_folder --test_only --language_modelling True \
	# 	--attention_type $attention_type --encoder_module $encoder_module \
	# 	--csv_location $csv_location
done
