#!/bin/bash
#SBATCH --job-name=er_b   # nom du job
#SBATCH --account=dha@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=72:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/brq_er_%j.log  # log file

module load pytorch-gpu/py3/2.0.1
conda activate aa
cd /gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp

hub=/gpfswork/rech/nkp/uaj64gk/attention_alt/brq-att-alt-exp/results/old/brq/1000/save/CKPT+2024-02-01+05-33-06+00
num_layers='13'
encoder_dim='512' # change this to 576
output_folder='results/MP3/brq'
benchmark_location=/gpfswork/rech/nkp/uaj64gk/attention_alt/benchmarks

task='IEMOCAP'
downstream='ecapa_tdnn'

for i in {1..10}
do
echo "on speaker $i"
python /users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S/$task/$downstream/train.py /users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /users/rwhetten/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" --ssl_hub "$hub" --encoder_dim "$encoder_dim" --output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--test_spk_id=$i
done


downstream='linear'

for i in {1..10}
do
echo "on speaker $i"
python /users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S/$task/$downstream/train.py /users/rwhetten/attention_alt/benchmarks/benchmarks/MP3S/$task/$downstream/hparams/ssl_brq.yaml \
    --data_folder /users/rwhetten/IEMOCAP/IEMOCAP_full_release \
	--num_layers_ssl "$num_layers" --ssl_hub "$hub" --encoder_dim "$encoder_dim" --output_folder "$output_folder"/"$task"/"$downstream/$i" \
	--test_spk_id=$i
done

