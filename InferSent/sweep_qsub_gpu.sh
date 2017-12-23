#!/bin/bash
#$ -l h_rt=128:0:0 # Time (h:m:s)
#$ -l tmem=47.1G # Mem (xG or xM)
#$ -l gpu=1
#$ -P gpu

singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg python /home/mmenezes/Dev/COMPM091/InferSent/sweep_job1.py
