#!/bin/bash
#SBATCH --job-name=extract_feats
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saandeepaath@usf.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120gb
#SBATCH --cpus-per-task=8
#SBATCH --output=/shares/rra_sarkar-2135-1003-00/faces/OpenFace/TorchOpenFace/slurm/output.%j
#SBATCH --partition=rra
#SBATCH --qos=rra


module load apps/anaconda
source /apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate openface
# cd build && make 
srun python torch_openface.py
