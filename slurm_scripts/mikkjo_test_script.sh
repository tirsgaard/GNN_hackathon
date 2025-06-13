#!/bin/bash

#SBATCH --job-name=<a_descriptive_job-name>
#SBATCH --output=/home/<your_username>/projects/GNN_Hackathon/logs/test-%A-%a.out
#SBATCH --mail-user=%u@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL
#SBATCH --partition=titans
#SBATCH --gres=gpu
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4
#SBATCH --exclude=comp-gpu14,comp-gpu01

cd $PROJECT_DIR
source manage_env.sh setup # This line here should be changed to activate your specific environment.

python3 gnn-hackathon/run-gnn.py