#!/bin/bash

#SBATCH --job-name=ff_lg   # nom du job
#SBATCH --account=rrg-ravanelm
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=10:00:00          # temps d'exécution maximum demande (HH:MM:SS) 
#SBATCH --output=log/ff_lg_%j.log  # log file

# load env
module load python/3.11
source $HOME/alt_att/aa/bin/activate

# set up data
scp -r $HOME/projects/def-ravanelm/datasets/librispeech $SLURM_TMPDIR/
cd $SLURM_TMPDIR/librispeech 
tar -zxf dev-clean.tar.gz
tar -zxf dev-other.tar.gz
tar -zxf test-clean.tar.gz
tar -zxf test-other.tar.gz
tar -zxf train-clean-100.tar.gz
tar -zxf train-clean-360.tar.gz
tar -zxf train-other-500.tar.gz

# set up run
cd  $HOME/alt_att/brq-att-alt-exp

python -m torch.distributed.run --nproc_per_node=1 --rdzv_backend c10d --rdzv-endpoint=localhost:0 train.py hparams/fastformer_lg.yaml --find_unused_parameters \
    --grad_accumulation_factor 16 --output_folder results/ff_lg_test/ff_lg_branch \
    --d_model 1184 --encoder_module branchformer --transformer_dropout 0.2 \
    --data_folder $SLURM_TMPDIR/librispeech/LibriSpeech 
