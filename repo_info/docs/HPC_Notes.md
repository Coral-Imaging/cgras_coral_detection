# Using QUT HPC  
Apply for an account [here](https://qutvirtual4.qut.edu.au/group/staff/research/conducting/facilities/advanced-research-computing-storage/supercomputing/getting-started-with-hpc)

WIKI [here](https://qut.atlassian.net/wiki/spaces/cyphy/pages/356746658/Working+with+QUT+s+HPC+High+Performance+Computing+Facilities) email [Vicki](v.martin@qut.edu.au) for access if restricted.
 

## Tips 
“Interactive session” - before installing packages  
Interactive session: qsub -I 
 
SSH into Lyra: ssh nxxxxxxx@lyra.qut.edu.au [then enter your password - you will need to have an active account to login to Lyra] 

 
## Mount Drives 
Entitled to lots of storage (multiple TB) on HPC. Can make virtual link between HPC drive and local PC for easier copy and data access using this [link](https://qutvirtual4.qut.edu.au/group/staff/research/conducting/facilities/advanced-research-computing-storage/supercomputing/using-hpc-filesystems) 


## Best inputs to HPC queue
Best GPUs: 
- A100 have large memory (40GB), most and fastest GPUs. However very popular (so longer wait time for job to run)
- P100s less memory (16GB) but still sufficent. 
Can test with `-l gpus=1` and leave gputype empty to see whats allocated
- If there are memory issues either decrease batch size or request gpu with more memory
- 1 GPU is sufficent, more means having to paallelize training which might be non-trivial

Request a couple (3) cpus. They help with dataloading and won't impact queue waitime

Careful with versions of PyTorch, CUDA etc for the different GPUs - the A100s require pretty recent versions of these packages because they are new, whereas the older GPUs can run on older versions of PyTorch, CUDA.  Just create different mamba environments if you need to install different versions of these packages. 

#PBS -m abe sets up mail events. 'a' for when job aborted, 'b' for when begins execution, 'e' for when terminated.


## Enviroment Setup
Will only need to run this process once.
1. Request interactive session
    `qsub -l`
2. Once in an interactive session run 
    ``wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
    export PATH=$HOME/mambaforge/bin:$PATH
    chmod +x cgras_settler_counter/make_cgras_venv.sh
    ./cgras_settler_counter/make_cgras_venv.sh``
3. Now future scripts can just run `mamba activate /mnt/hpccs01/home/<username>/mambaforge`


## Example Jobscript
```
#PBS -N long_train
#PBS -l walltime=12:00:00
#PBS -l ncpus=3
#PBS -l mem=16gb
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -m abe
#PBS -I

cd $PBS_O_WORKDIR
python myscript.py
```
Run the above by typing `qsub jobscriptname.pbs`. wwill output a 'jobname.o' which has the terminal output of the jobscript and a 'jobname.e' file with any errors.
 