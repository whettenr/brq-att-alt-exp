#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

module purge # nettoyer les modules herites par defaut
module load python/3.11.5
module load ffmpeg/4.2.2
conda deactivate # for some reason, this can fail
conda activate mamba_ssl

echo "***************************"
echo "r$SLURM_NODEID SLURM_TMPDIR: $SLURM_TMPDIR"
echo "nnodes=$SLURM_JOB_NUM_NODES"
echo "node_rank=$SLURM_NODEID"
echo "master=$MASTER"
echo "master_port=$MASTER_PORT"
echo "***************************"

torchrun --nproc_per_node=8 \
	--master_port=${MASTER_PORT} \
	--nnodes=${SLURM_JOB_NUM_NODES} \
	--node_rank=${SLURM_NODEID} \
	--master_addr=$(hostname --ip-address)  \
	train.py hparams/brq_mamba_bidirectional.yaml \
	--data_folder=/gpfsdswork/dataset/LibriSpeech \
	--output_folder=/gpfsscratch/rech/uul/ujg45iy/brq_mamba_bidirectional \
	--precision=fp16 \
    --find_unused_parameters