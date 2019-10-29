#!/bin/bash
echo Submitting PLA job
sbatch s_single_pla.sh
echo Submitting SARA job
sbatch s_single_sara.sh
echo Submitting Random job
sbatch s_single_random.sh
