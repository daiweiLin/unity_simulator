#!/bin/bash
#SBATCH --array=0-2
#SBATCH --mem=875M
#SBATCH -t 1-00:00
#SBATCH -o s_random_%job.out
#SBATCH --job-name=vpoint1_multi_0.1_random
source ~/unity_ml/bin/activate
python Test_Simulator.py True Random ddpg 5 0.1 simple $SLURM_ARRAY_TASK_ID