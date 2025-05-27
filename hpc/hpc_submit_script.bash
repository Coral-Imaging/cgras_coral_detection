#!/bin/bash

#PBS -N long_train
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=12:ngpus=1:mem=64GB:gpu_id=A100
#PBS -m abe
#PBS -I
