#$ -S /bin/bash
#$ -l h_rt=23:30:0 # Time (h:m:s)
#$ -l tmem=15.5G # Mem (xG or xM)
#$ -l gpu=1
#$ -P gpu

hostname
# $@ --gpu_id $(nvidia-smi dmon -c 1 | grep -v "#" | awk '{ print $4 " " $1 }' | sort -n | awk '{ print $2 }' | tr -d "\n" | head -c 1)
$@ --gpu_id $CUDA_VISIBLE_DEVICES

