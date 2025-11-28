#!/bin/bash

#SBATCH --nodes 4
#SBATCH --qos debug
#SBATCH --time 00:30:00
#SBATCH --constraint gpu
#SBATCH --gpus 16
#SBATCH --account m4646_g

srun ./build/test_compression
