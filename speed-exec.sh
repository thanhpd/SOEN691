#!/encs/bin/tcsh

#SBATCH -J llmeval              ## Job’s name set to ’llmeval’
#SBATCH --mail-type=ALL       ## Receive all email type notifications
#SBATCH -D ./                 ## Use current directory as working directory
#SBATCH -N 1                  ## Node count required for the job
#SBATCH -n 1                  ## Number of tasks to be launched
#SBATCH -c 32                  ## Request 32 cores
#SBATCH --mem=32G             ## Allocate 32G memory per node

date

module load anaconda3/2023.03/default
conda activate /speed-scratch/ph_thanh/myconda

date

cd measure_script

date
srun python run_metrics.py

date
srun python run_bertscore.py

date
srun python run_metrics_per_line.py

date
srun python bertscore_per_line.py
