#!/bin/bash
echo Submitting PLA job
sbatch s_multi_pla.sh
echo Submitting SARA job
sbatch s_multi_sara.sh
echo Submitting Random job
sbatch s_multi_random.sh
