#!/bin/bash

#PBS -N long_train
#PBS -l walltime=08:00:00
#PBS -l ncpus=8
#PBS -l mem=32gb
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -m abe
#PBS -I