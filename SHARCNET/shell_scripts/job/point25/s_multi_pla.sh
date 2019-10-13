#!/bin/bash
#SBATCH --array=0-1
#SBATCH --mem=1800M
#SBATCH -t 1-00:00
#SBATCH -o s_pla_%job.out
#SBATCH --job-name=point25_multi_0.3_pla
source ~/unity_ml/bin/activate
python Test_Simulator.py True PLA ddpg 5 0.3 $SLURM_ARRAY_TASK_ID