# @Author: Geethma Werapitiya <wgeethma>
# @Date:   2022-05-24T10:24:47-06:00
# @Email:  wgeethma@uwyo.edu
# @Last modified by:   wgeethma
# @Last modified time: 2022-05-25T13:19:13-06:00



#!/bin/bash
#PBS -A WYOM0131
#PBS -N PDF
#PBS -l walltime=03:00:00
#PBS -q regular
#PBS -m abe
#PBS -M wgeethma@uwyo.edu
#PBS -l select=1:ncpus=36

python GCMwind_M_pdf.py
