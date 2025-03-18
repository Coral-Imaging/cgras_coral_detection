#!/bin/bash

#PBS -N cslics_train
#PBS -l walltime=3:00:00
#PBS -l ncpus=12
#PBS -l mem=128gb
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -m abe
#PBS -I