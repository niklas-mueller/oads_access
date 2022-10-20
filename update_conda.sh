#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

module load 2021
cd $HOME/projects/oads_access
conda activate oads
conda update --all
conda update --all
# conda env update -f environment.yml
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

