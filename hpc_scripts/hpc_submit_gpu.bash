#!/bin/bash

#PBS -N dataset_analysis
#PBS -l walltime=2:00:00
#PBS -l ncpus=16
#PBS -l mem=120gb
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -m abe
#PBS -I