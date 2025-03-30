# Start the metric collection process on the Speed cluster

References:
- https://github.com/NAG-DevOps/speed-hpc
- HTML reference: https://nag-devops.github.io/speed-hpc/

## 1. Set up the environment
### 1.1 Verify the current environment:
```sh
# Verify PATH
echo $PATH

# Create your personal space and go to that space
mkdir -p /speed-scratch/$USER && cd /speed-scratch/$USER

# Clone the code repository in the /speed-scratch/$USER folder. Alternatively, you can use scp or rsync.
git clone https://github.com/thanhpd/SOEN691.git
```
More details: https://nag-devops.github.io/speed-hpc/#environment-set-up

### 1.2. Set up your python environment

Details: https://nag-devops.github.io/speed-hpc/#anaconda
```sh
# The first node you connected to is only for job submission, not for computing. You'll need to create a new interactive job:
salloc --mem=10G -p ps -A soen691w25

# Load the anaconda module and create a new env with the name myconda
module load anaconda3/2023.03/default
conda create --prefix /speed-scratch/$USER/myconda

# Verify conda env
conda info --envs

# Activate the new env and install python and dependencies. It may ask you to restart the shell. Just disconnect and reconnect.
conda activate /speed-scratch/$USER/myconda
conda install python=3.12.9
pip install bert-score

conda deactivate
```

### 1.3. Prepare for the job execution
- Determine your parallelization strategy: how many jobs to run in parallel? How many cpu cores are needed for each job? These need to be answered as you'll provide these information. 1 job using 32 cores will achieve the highest speed, but won't be able to leverage parallelism so it doesn't necessarily finish the quickest.
- In `/speed-scratch/$USER/SOEN691`, create a new bash file speed-exec.sh with the following content below. You'll need to change some parameters inside.

```sh
#!/encs/bin/tcsh

#SBATCH -J llmeval              ## Job’s name set to ’llmeval’ -> OPTIONAL: CHANGE THIS
#SBATCH --mail-type=ALL       ## Receive all email type notifications
#SBATCH -D ./                 ## Use current directory as working directory
#SBATCH -N 1                  ## Node count required for the job
#SBATCH -n 1                  ## Number of tasks to be launched
#SBATCH -c 32                  ## Number of CPU core needed for this job. By default you have a max of 32 cores -> CHANGE THIS
#SBATCH --mem=32G             ## Allocate 32G memory per node -> CHANGE THIS

date

module load anaconda3/2023.03/default
conda activate /speed-scratch/ph_thanh/myconda

date

cd measure_script

## Provide only the commands you want to run. The example below run both summary & per line scripts for all metrics
date
srun python run_metrics.py

date
srun python run_bertscore.py

date
srun python run_metrics_per_line.py

date
srun python bertscore_per_line.py
```

Make this file executable:
```sh
chmod +x speed-exec.sh
```

Ref: https://github.com/thanhpd/SOEN691/blob/main/speed-exec.sh

- Clone the folder as many times as you needed for the job execution. For example:
```sh
cp -r SOEN691 perline1
# Many times
cp -r SOEN691 perline10
```

- Go into each folder and upload the TOKENIZED dataset for each job using SCP/RSYNC (alternatively, you can upload the whole dataset before cloning the folder and then go into each folder and delete the files you don't need. I use the latter solution so I can keep the folder structure intact)
The file destination should be: `<code_folder_root>/measure_script/processed_msg`
See more about the evaluation process here: https://github.com/thanhpd/SOEN691/blob/dev/thanh/measure_script/How%20to%20measure.txt
  
```
# Example using RSYNC to copy both folder and files from the current folder on local to the Speed cluster
rsync -av ./ ph_thanh@speed.encs.concordia.ca:/speed-scratch/ph_thanh/SOEN691/summary_without_empty/measure_script/processed_msg/20000_op_filtered
``` 

## 2. Run the script
- Optional: Go into each cloned folder > rename the job name so you can track the job in case it failed (you can also track it by the job id)

### 2.1. Submit the non-interactive jobs
Make sure you're not submitting the job inside the interactive job. Best way is to disconnect and reconnect again:
```
# In each code folder, run:
sbatch -p ps ./speed-exec.sh -A soen691w25
```

You'll receive notifications when the status of the job changes (started, completed, failed)
The execution output is stored in slurm-<jobid>.out of each folder

### 2.2. Job management commands
See: https://nag-devops.github.io/speed-hpc/#common-job-management-commands
