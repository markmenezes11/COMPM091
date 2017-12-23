#!/bin/bash
#$ -l h_rt=23:30:0 # Time (h:m:s)
#$ -l tmem=15.5G # Mem (xG or xM)
#$ -l gpu=1
#$ -P gpu

singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg python /home/mmenezes/Dev/COMPM091/InferSent/train_qsub_wrapper.py $@
