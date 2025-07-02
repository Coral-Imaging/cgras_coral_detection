#!/bin/bash -l

#PBS -N test
#PBS -l select=4:ncpus=4:ngpus=1:mem=8GB:gpu_id=A100 
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m abe
