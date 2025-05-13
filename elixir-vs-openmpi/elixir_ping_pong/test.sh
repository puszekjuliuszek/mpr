#!/bin/bash
#SBATCH -J plgmpr25-cpu
#SBATCH -o plgmpr25-cpu-%a-%j.out
#SBATCH -e plgmpr25-cpu-%a-%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G
#SBATCH -p plgrid
#SBATCH --time=00:05:00

cd ~/mpr/elixir_demo
mix run