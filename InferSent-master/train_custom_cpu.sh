#!/bin/bash
#$ -l h_rt=96:0:0 # Time (h:m:s)
#$ -l tmem=32G -l h_vmem=32G # Mem (xG or xM)

mkdir /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/model_snli_small_cpu
singularity exec /home/mmenezes/Dev/_COMPM091/COMPM091/markmenezes11-COMPM091-master.simg python /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/train_custom.py --gpu_id -1 --wordvecpath /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/dataset/GloVe/glove.840B.300d.txt --nlipath /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/dataset/SNLI_small --outputdir /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/model_snli_small_cpu |& tee /home/mmenezes/Dev/_COMPM091/COMPM091/InferSent-master/model_snli_small_cpu/output.txt
