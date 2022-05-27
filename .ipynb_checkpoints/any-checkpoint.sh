#!/bin/bash
#PBS -A WYOM0131
#PBS -N plot
#PBS -l walltime=03:00:00
#PBS -q regular
#PBS -m abe
#PBS -M wgeethma@uwyo.edu
#PBS -l select=1:ncpus=36

python GCMwind_M_readmodels.py
