    #!/bin/bash

    #PBS -N unzip_script
    #PBS -l ncpus=64
    #PBS -l mem=128gb
    #PBS -l walltime=30:00
    #PBS -m abe

    cd $PBS_O_WORKDIR
    
    source mambaforge/bin/activate cgras
    # conda activate /home/tsaid/miniforge3/envs/cslics
    # conda activate cslics

    python3 Corals/cgras_settler_counter/hpc/unzip.py
    # python3 myscript.py

    conda deactivate