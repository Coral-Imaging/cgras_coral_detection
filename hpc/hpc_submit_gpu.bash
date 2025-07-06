#!/bin/bash

#PBS -N dataset_analysis
#PBS -l walltime=7:00:00
#PBS -l ncpus=8
#PBS -l mem=64gb
#PBS -l ngpus=1
#PBS -l gpu_id=A100
#PBS -m abe
#PBS -I