#!/bin/bash
#SBATCH --array=0-1
#SBATCH --mem=2048M
#SBATCH -t 1-00:00
#SBATCH -o s_sara_%job.out
#SBATCH --job-name=one_multi_0.3_sara
source ~/unity_ml/bin/activate
python Test_Simulator.py True SARA ddpg 5 0.3 $SLURM_ARRAY_TASK_ID