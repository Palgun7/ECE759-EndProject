#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J Task1

#SBATCH -o Task.out

#SBATCH -e Task.err

#SBATCH --cpus-per-task=20

#SBATCH -t 0-00:10:00

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0
# cd repo759/HW03
g++ task.cpp pca.cpp -o pca -fopenmp -std=c++17

for ((counter=1; counter<=20; counter = counter+1))
do
./task1 1024 $counter
printf "\n"
done