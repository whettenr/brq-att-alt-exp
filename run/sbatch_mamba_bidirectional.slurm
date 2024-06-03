#!/bin/bash
#SBATCH --job-name=ssl-mamba #nom du job
#SBATCH --output=ssl-mamba%j.log # fichier de sortie (%j = job ID)
#SBATCH --error=ssl-mamba%j.log # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
##SBATCH -C v100-32g
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_p2
#SBATCH --cpus-per-task=24
##SBATCH --qos=qos_gpu-dev # QoS; for testing and <2h you may use gpu-dev
#SBATCH --time=20:00:00
#SBATCH --account=uul@v100
#SBATCH --array=0-6%1

set -ex # activer l’echo des commandes

cd ${SLURM_SUBMIT_DIR}

export TORCH_NCCL_BLOCKING_WAIT=1
export MASTER=$(hostname --ip-address)
export MASTER_PORT=$((RANDOM%1000+20000))

srun run_mamba_bidirectional.sh